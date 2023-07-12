import os
import time
import json
import datetime as datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import transforms


from dataloaders.train_datasets import ViSha
import dataloaders.video_transforms as tr

from utils.meters import AverageMeter
from utils.image import label2colormap, masked_image, save_image
from utils.checkpoint import load_network, save_network
from utils.learning import adjust_learning_rate, get_trainable_params
from utils.metric import pytorch_iou
from utils.ema import ExponentialMovingAverage, get_param_buffer_for_ema

from networks.models import build_vos_model
from networks.engines import build_engine


class Trainer(object):
    def __init__(self, rank, cfg, enable_amp=True):
        self.gpu = rank
        self.gpu_num = cfg.TRAIN_GPUS 
        self.rank = rank
        self.cfg = cfg

        self.print_log("Exp {}:".format(cfg.EXP_NAME))
        self.print_log(json.dumps(cfg.__dict__, indent=4, sort_keys=True))
        self.log_save = os.path.join(cfg.DIR_RESULT, 'train.txt')


        print("Use GPU {} for training LSTN.".format(self.gpu))
        torch.cuda.set_device(self.gpu)
        torch.backends.cudnn.benchmark = True if cfg.DATA_RANDOMCROP[0]==cfg.DATA_RANDOMCROP[1] and 'swin' not in cfg.MODEL_ENCODER else False


        # build model
        self.model = build_vos_model(cfg.MODEL_VSD, cfg).cuda(self.gpu)
        print("Params: ", sum(p.numel() for p in self.model.parameters())/1e6)
        self.model_encoder = self.model.encoder
        self.engine = build_engine(cfg.MODEL_ENGINE, 'train', lstn_model=self.model, gpu_id=self.gpu, long_term_mem_gap=cfg.TRAIN_LONG_TERM_MEM_GAP)
        if cfg.MODEL_FREEZE_BACKBONE:
            for param in self.model_encoder.parameters():
                param.requires_grad = False

        # init ddp
        if cfg.DIST_ENABLE:
            dist.init_process_group(backend=cfg.DIST_BACKEND,  # nccl
                                    init_method=cfg.DIST_URL,  # tcp://
                                    world_size=cfg.TRAIN_GPUS,
                                    rank=rank,
                                    timeout=datetime.timedelta(seconds=300))
            self.model.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.model.encoder).cuda(self.gpu)
            self.dist_engine = torch.nn.parallel.DistributedDataParallel(self.engine,
                                                                        device_ids=[self.gpu],
                                                                        output_device=self.gpu,
                                                                        find_unused_parameters=True,
                                                                        broadcast_buffers=False)
        else:
            self.dist_engine = self.engine


        # about batch norm
        self.use_frozen_bn = False
        if 'swin' in cfg.MODEL_ENCODER:
            self.print_log('Use LN in Encoder!')
        elif not cfg.MODEL_FREEZE_BN:
            if cfg.DIST_ENABLE:
                self.print_log('Use Sync BN in Encoder!')
            else:
                self.print_log('Use BN in Encoder!')
        else:
            self.use_frozen_bn = True
            self.print_log('Use Frozen BN in Encoder!')


        # init train params
        if self.rank == 0:
            total_steps = float(cfg.TRAIN_TOTAL_STEPS)
            ema_decay = 1. - 1. / (total_steps * cfg.TRAIN_EMA_RATIO)  # 0.9999
            self.ema_params = get_param_buffer_for_ema(self.model, update_buffer=(not cfg.MODEL_FREEZE_BN))
            self.ema = ExponentialMovingAverage(self.ema_params, decay=ema_decay)

        self.print_log('Build optimizer.')
        trainable_params = get_trainable_params(
            model=self.dist_engine,
            base_lr=cfg.TRAIN_LR,
            use_frozen_bn=self.use_frozen_bn,
            weight_decay=cfg.TRAIN_WEIGHT_DECAY,
            exclusive_wd_dict=cfg.TRAIN_WEIGHT_DECAY_EXCLUSIVE,
            no_wd_keys=cfg.TRAIN_WEIGHT_DECAY_EXEMPTION)


        # optimizer
        if cfg.TRAIN_OPT == 'sgd':
            self.optimizer = optim.SGD(trainable_params, lr=cfg.TRAIN_LR, momentum=cfg.TRAIN_SGD_MOMENTUM, nesterov=True)
        else:
            self.optimizer = optim.AdamW(trainable_params, lr=cfg.TRAIN_LR, weight_decay=cfg.TRAIN_WEIGHT_DECAY)

        self.enable_amp = enable_amp
        if enable_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self.prepare_dataset()
        self.process_pretrained_model()


    def prepare_dataset(self):
        cfg = self.cfg
        self.print_log('Process dataset...')
        composed_transforms = transforms.Compose([
            tr.RandomScale(cfg.DATA_MIN_SCALE_FACTOR, cfg.DATA_MAX_SCALE_FACTOR, cfg.DATA_SHORT_EDGE_LEN),  # random scale，ensure image scale>DATA_SHORT_EDGE_ELN
            tr.BalancedRandomCrop(cfg.DATA_RANDOMCROP, max_obj_num=cfg.MODEL_MAX_OBJ_NUM),  # compare hw with 465, choose the small one and crop it
            tr.Resize(cfg.DATA_RANDOMCROP, use_padding=True),  # true: padding with 0; false: interpolation
            tr.ToTensor()
        ])

        train_dataset = ViSha(root=cfg.DIR_VISHA,
                            transform=composed_transforms,
                            repeat_time=cfg.DATA_VISHA_REPEAT,
                            seq_len=cfg.DATA_SEQ_LEN,
                            rand_gap=cfg.DATA_RANDOM_GAP_DAVIS,
                            rand_reverse=cfg.DATA_RANDOM_REVERSE_SEQ,
                            max_obj_n=cfg.MODEL_MAX_OBJ_NUM)

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if self.cfg.DIST_ENABLE else None
        self.train_loader = DataLoader(train_dataset, batch_size=int(cfg.TRAIN_BATCH_SIZE / cfg.TRAIN_GPUS),
                                       shuffle=False if self.cfg.DIST_ENABLE else True, num_workers=cfg.DATA_WORKERS,
                                       pin_memory=True, sampler=self.train_sampler, drop_last=True, prefetch_factor=16)


    def process_pretrained_model(self):
        cfg = self.cfg
        model_encoder, removed_dict = load_network(self.model_encoder, cfg.PRETRAIN_MODEL, self.gpu)
        if len(removed_dict) > 0:
            self.print_log('Remove {} from pretrained model.'.format(removed_dict))
        self.print_log('Load pretrained backbone model from {}.'.format(cfg.PRETRAIN_MODEL))


    def sequential_training(self):
        cfg = self.cfg
        frame_names = ['Ref(Prev)']
        for i in range(cfg.DATA_SEQ_LEN - 1):
            frame_names.append('Curr{}'.format(i + 1))
        seq_len = len(frame_names)


        running_losses = []
        running_ious = []
        for _ in range(seq_len):
            running_losses.append(AverageMeter())
            running_ious.append(AverageMeter())
        batch_time = AverageMeter()

        optimizer = self.optimizer
        model = self.dist_engine
        train_sampler = self.train_sampler
        train_loader = self.train_loader

        max_itr = cfg.TRAIN_TOTAL_STEPS
        start_seq_training_step = int(cfg.TRAIN_SEQ_TRAINING_START_RATIO * max_itr)
        use_prev_prob = cfg.MODEL_USE_PREV_PROB

        step = 0
        epoch = 0
        model.train()
        while step < cfg.TRAIN_TOTAL_STEPS:
            if self.cfg.DIST_ENABLE:
                train_sampler.set_epoch(epoch)
            epoch += 1
            last_time = time.time()
            for _, sample in enumerate(train_loader):
                if step > cfg.TRAIN_TOTAL_STEPS:  # total step --> end
                    break
                if step >= start_seq_training_step:
                    use_prev_pred = True
                    freeze_params = cfg.TRAIN_SEQ_TRAINING_FREEZE_PARAMS
                else:
                    use_prev_pred = False
                    freeze_params = []

                if step % cfg.TRAIN_LR_UPDATE_STEP == 0:
                    now_lr = adjust_learning_rate(
                        optimizer=optimizer,
                        base_lr=cfg.TRAIN_LR,
                        p=cfg.TRAIN_LR_POWER,
                        itr=step,
                        max_itr=max_itr,
                        restart=cfg.TRAIN_LR_RESTART,
                        warm_up_steps=cfg.TRAIN_LR_WARM_UP_RATIO * max_itr,
                        is_cosine_decay=cfg.TRAIN_LR_COSINE_DECAY,
                        min_lr=cfg.TRAIN_LR_MIN,
                        encoder_lr_ratio=cfg.TRAIN_LR_ENCODER_RATIO,
                        freeze_params=freeze_params)

                ref_imgs = sample['ref_img'].cuda(self.gpu, non_blocking=True)  # batch_size * 3 * h * w
                prev_imgs = sample['prev_img'].cuda(self.gpu, non_blocking=True)
                curr_imgs = [curr_img.cuda(self.gpu, non_blocking=True) for curr_img in sample['curr_img']]
                ref_labels = sample['ref_label'].cuda(self.gpu, non_blocking=True)  # batch_size * 1 * h * w
                prev_labels = sample['prev_label'].cuda(self.gpu, non_blocking=True)
                curr_labels = [curr_label.cuda(self.gpu, non_blocking=True) for curr_label in sample['curr_label']]
                obj_nums = sample['meta']['obj_num']

                bs, _, h, w = curr_imgs[0].size()
                obj_nums = list(obj_nums)
                obj_nums = [int(obj_num) for obj_num in obj_nums]

                batch_size = ref_imgs.size(0)
                all_frames = torch.cat([ref_imgs, prev_imgs] + curr_imgs, dim=0)
                all_labels = torch.cat([ref_labels, prev_labels] + curr_labels, dim=0)

                self.engine.restart_engine(batch_size, True)
                optimizer.zero_grad(set_to_none=True)

                if self.enable_amp:
                    with torch.cuda.amp.autocast(enabled=True):
                        loss, all_pred, all_loss = model(all_frames,
                                                        all_labels,
                                                        use_prev_pred=use_prev_pred,
                                                        obj_nums=obj_nums,
                                                        step=step,
                                                        enable_prev_frame=False,
                                                        use_prev_prob=use_prev_prob)
                        loss = torch.mean(loss)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN_CLIP_GRAD_NORM)
                    self.scaler.step(optimizer)
                    self.scaler.update()  
                else:
                    loss, all_pred, all_loss = model(all_frames,
                                                    all_labels,
                                                    use_prev_pred=use_prev_pred,
                                                    obj_nums=obj_nums,
                                                    step=step,
                                                    enable_prev_frame=False,
                                                    use_prev_prob=use_prev_prob)
                    loss = torch.mean(loss)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN_CLIP_GRAD_NORM)
                    loss.backward()
                    optimizer.step()

                # log loss
                for idx in range(seq_len):
                    now_pred = all_pred[idx].detach()  # frame 1
                    now_label = all_labels[idx*bs:(idx+1)*bs].detach()  # frame 1
                    now_loss = torch.mean(all_loss[idx].detach())
                    now_iou = pytorch_iou(now_pred.unsqueeze(1), now_label, obj_nums) * 100
                    if self.cfg.DIST_ENABLE:
                        dist.all_reduce(now_loss)
                        dist.all_reduce(now_iou)
                        now_loss /= self.gpu_num
                        now_iou /= self.gpu_num
                    if self.rank == 0:
                        running_losses[idx].update(now_loss.item())
                        running_ious[idx].update(now_iou.item())

                # update ema
                if self.rank == 0:
                    self.ema.update(self.ema_params)
                    curr_time = time.time()
                    batch_time.update(curr_time - last_time)
                    last_time = curr_time

                    # image log
                    if step % cfg.TRAIN_IMGLOG_STEP == 0:
                        self.process_log(ref_imgs, curr_imgs[-2], curr_imgs[-1], 
                                         ref_labels, curr_labels[-2], curr_labels[-1], all_pred[-1], step)

                    # log step
                    if step % cfg.TRAIN_LOG_STEP == 0:
                        strs = 'I:{}, LR:{:.5f}'.format(step, now_lr)
                        batch_time.reset()
                        for idx in range(seq_len):
                            strs += ', {}: L {:.3f} IoU {:.1f}%'.format(
                                frame_names[idx], running_losses[idx].val, running_ious[idx].val)
                            running_losses[idx].reset()
                            running_ious[idx].reset()
                        self.write_print_log(strs)

                step += 1
                # save checkpoints
                if step % cfg.TRAIN_SAVE_STEP == 0 and self.rank == 0:
                    max_mem = torch.cuda.max_memory_allocated(device=self.gpu) / (1024.**3)
                    ETA = str(datetime.timedelta(seconds=int(batch_time.moving_avg * (cfg.TRAIN_TOTAL_STEPS - step))))
                    self.write_print_log('ETA: {}, Max Mem: {:.2f}G.'.format(ETA, max_mem))
                    self.write_print_log('Save CKPT (Step {}).'.format(step))
                    save_network(self.model, optimizer, step, cfg.DIR_CKPT, cfg.TRAIN_MAX_KEEP_CKPT, 
                                    backup_dir='./backup/{}/ckpt'.format(cfg.EXP_NAME), scaler=self.scaler)
                    torch.cuda.empty_cache()
                    self.ema.store(self.ema_params)  # First save original parameters before replacing with EMA version
                    self.ema.copy_to(self.ema_params)  # Copy EMA parameters to model
                    # Save EMA model
                    save_network(self.model, optimizer, step, cfg.DIR_EMA_CKPT, cfg.TRAIN_MAX_KEEP_CKPT,
                        backup_dir='./backup/{}/ema_ckpt'.format(cfg.EXP_NAME), scaler=self.scaler)
                    self.ema.restore(self.ema_params) # Restore original parameters to resume training later
        self.write_print_log('Stop training!')


    def print_log(self, string):
        if self.rank == 0:
            print(string)

    def write_print_log(self, string):
        if self.rank == 0:
            self.print_log(string)
            with open(self.log_save, 'a+') as f:
                f.write(string+'\n')  #文件的写操作


    def process_log(self, ref_imgs, prev_imgs, curr_imgs, 
                    ref_labels, prev_labels, curr_labels, curr_pred, step):
        cfg = self.cfg

        mean = np.array([[[0.485]], [[0.456]], [[0.406]]])
        sigma = np.array([[[0.229]], [[0.224]], [[0.225]]])
        # gt: ref, pre, cur
        # pred: cur
        show_ref_img, show_prev_img, show_curr_img = [
            img.cpu().numpy()[0] * sigma + mean 
            for img in [ref_imgs, prev_imgs, curr_imgs]
        ]

        show_ref_gt, show_prev_gt, show_gt, show_preds_s = [
            label.cpu()[0].squeeze(0).numpy() 
            for label in [ref_labels, prev_labels, curr_labels, curr_pred]
        ]

        show_ref_gtf, show_prev_gtf, show_gtf, show_preds_sf = [
            label2colormap(label).transpose((2, 0, 1)) 
            for label in [show_ref_gt, show_prev_gt, show_gt, show_preds_s]
        ]

        show_ref_img = masked_image(show_ref_img, show_ref_gtf, show_ref_gt)
        save_image(show_ref_img, os.path.join(cfg.DIR_IMG_LOG, '%06d_ref_img.jpeg' % (step)))
        show_prev_img = masked_image(show_prev_img, show_prev_gtf, show_prev_gt)
        save_image(show_prev_img, os.path.join(cfg.DIR_IMG_LOG, '%06d_prev_img.jpeg' % (step)))
        show_curr_img = masked_image(show_curr_img, show_gtf, show_gt)
        save_image(show_curr_img, os.path.join(cfg.DIR_IMG_LOG, '%06d_groundtruth.jpeg' % (step)))
        show_img_pred = masked_image(show_curr_img, show_preds_sf, show_preds_s)
        save_image(show_img_pred, os.path.join(cfg.DIR_IMG_LOG, '%06d_prediction.jpeg' % (step)))

