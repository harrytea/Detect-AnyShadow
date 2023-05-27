import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.math import generate_permute_matrix
from utils.image import one_hot_mask

from networks.layers.basic import seq_to_2d


class LSTNEngine(nn.Module):
    def __init__(self, lstn_model, gpu_id=0, long_term_mem_gap=9999, short_term_mem_skip=1):
        super().__init__()

        self.cfg = lstn_model.cfg
        self.align_corners = lstn_model.cfg.MODEL_ALIGN_CORNERS
        self.lstn = lstn_model

        self.max_obj_num = lstn_model.max_obj_num
        self.gpu_id = gpu_id
        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip
        self.losses = None

        self.restart_engine()

    def forward(self, all_frames, all_masks, obj_nums, step=0,
                use_prev_pred=False, enable_prev_frame=False, use_prev_prob=False):  # only used for training
        if self.losses is None:
            self._init_losses()

        self.freeze_id = True if use_prev_pred else False
        aux_weight = self.aux_weight * max(self.aux_step - step, 0.) / self.aux_step

        self.offline_encoder(all_frames, all_masks)  # extract backbone feature
        self.add_reference_frame(frame_step=0, obj_nums=obj_nums)  # operate on reference frame

        grad_state = torch.no_grad if aux_weight == 0 else torch.enable_grad
        # generate_loss_mask: 返回loss以及预测的<离散化>的mask
        with grad_state():
            ref_aux_loss, ref_aux_mask = self.generate_loss_mask(self.offline_masks[self.frame_step], step)

        aux_losses = [ref_aux_loss]
        aux_masks = [ref_aux_mask]

        curr_losses, curr_masks = [], []
        self.match_propogate_one_frame()  # 找到第2个frame，并进行long-short term attention
        curr_loss, curr_mask, curr_prob = self.generate_loss_mask(self.offline_masks[self.frame_step], step, return_prob=True)
        self.update_short_term_memory(curr_mask if not use_prev_prob else curr_prob, None if use_prev_pred else self.assign_identity(self.offline_one_hot_masks[self.frame_step]))
        curr_losses.append(curr_loss)
        curr_masks.append(curr_mask)


        self.match_propogate_one_frame()
        curr_loss, curr_mask, curr_prob = self.generate_loss_mask(self.offline_masks[self.frame_step], step, return_prob=True)
        curr_losses.append(curr_loss)
        curr_masks.append(curr_mask)


        for _ in range(self.total_offline_frame_num - 3):
            self.update_short_term_memory(curr_mask if not use_prev_prob else curr_prob,
                None if use_prev_pred else self.assign_identity(self.offline_one_hot_masks[self.frame_step]))
            self.match_propogate_one_frame()
            curr_loss, curr_mask, curr_prob = self.generate_loss_mask(self.offline_masks[self.frame_step], step, return_prob=True)
            curr_losses.append(curr_loss)
            curr_masks.append(curr_mask)

        aux_loss = torch.cat(aux_losses, dim=0).mean(dim=0)
        pred_loss = torch.cat(curr_losses, dim=0).mean(dim=0)

        loss = aux_weight*aux_loss + pred_loss
        all_pred_mask = aux_masks+curr_masks
        all_frame_loss = aux_losses+curr_losses

        return loss, all_pred_mask, all_frame_loss

    def _init_losses(self):
        cfg = self.cfg

        from networks.layers.loss import CrossEntropyLoss, SoftJaccordLoss
        bce_loss = CrossEntropyLoss(cfg.TRAIN_TOP_K_PERCENT_PIXELS, cfg.TRAIN_HARD_MINING_RATIO * cfg.TRAIN_TOTAL_STEPS)
        iou_loss = SoftJaccordLoss()

        losses = [bce_loss, iou_loss]
        loss_weights = [0.5, 0.5]

        self.losses = nn.ModuleList(losses)
        self.loss_weights = loss_weights
        self.aux_weight = cfg.TRAIN_AUX_LOSS_WEIGHT
        self.aux_step = cfg.TRAIN_TOTAL_STEPS * cfg.TRAIN_AUX_LOSS_RATIO + 1e-5


    def offline_encoder(self, all_frames, all_masks=None):
        self.enable_offline_enc = True
        self.offline_frames = all_frames.size(0) // self.batch_size  # 5

        # image embeddings
        self.offline_enc_embs = self.split_frames(self.lstn.encode_image(all_frames), self.batch_size)
        self.total_offline_frame_num = len(self.offline_enc_embs)  # 5

        # mask embeddings
        if all_masks is not None:
            # extract mask embeddings
            offline_one_hot_masks = one_hot_mask(all_masks, self.max_obj_num)  # [10, 11, 456, 456]
            self.offline_masks = list(torch.split(all_masks, self.batch_size, dim=0))  # [[2, 1, 456, 456], ..., [2, 1, 456, 456]]
            self.offline_one_hot_masks = list(torch.split(offline_one_hot_masks, self.batch_size, dim=0))  # [[2, 11, 456, 456], ..., [2, 11, 456, 456]]

        # update image and embed size
        if self.input_size_2d is None:
            self.update_size(all_frames.size()[2:], self.offline_enc_embs[0][-1].size()[2:])


    def split_frames(self, xs, chunk_size):
        new_xs = []
        for x in xs:  # xs四个尺度特征，每个特征有十个frame
            all_x = list(torch.split(x, chunk_size, dim=0))
            new_xs.append(all_x)
        return list(zip(*new_xs))

    def add_reference_frame(self, img=None, mask=None, frame_step=-1, obj_nums=None, img_embs=None):
        if self.obj_nums is None and obj_nums is None:
            print('No objects for reference frame!')
            exit()
        elif obj_nums is not None:
            self.obj_nums = obj_nums

        if frame_step == -1:
            frame_step = self.frame_step

        # get enc_embs and mask_embs
        if img_embs is None:  # 提取ref的所有embedding和ref的mask
            curr_enc_embs, curr_one_hot_mask = self.encode_one_img_mask(img, mask, frame_step)
        else:
            _, curr_one_hot_mask = self.encode_one_img_mask(None, mask, frame_step)
            curr_enc_embs = img_embs

        if curr_enc_embs is None or curr_one_hot_mask is None:
            print('No image/mask for reference frame!')
            exit()

        if self.input_size_2d is None:
            self.update_size(img.size()[2:], curr_enc_embs[-1].size()[2:])

        self.curr_enc_embs = curr_enc_embs
        self.curr_one_hot_mask = curr_one_hot_mask

        # pos and id embedding
        if self.pos_emb is None:
            self.pos_emb = self.lstn.get_pos_emb(curr_enc_embs[-1])  \
                .expand(self.batch_size, -1, -1,-1).view(self.batch_size, -1, self.enc_hw).permute(2, 0, 1)  # [900, 2, 256]
 
        curr_id_emb = self.assign_identity(curr_one_hot_mask)  # curr_id_emb, 这个是将mask进行ID编码了
        self.curr_id_embs = curr_id_emb  # [900, 2, 256]

        # self matching and propagation   # curr_id_emb和pos_emb分别是ID编码和pos编码
        self.curr_lsab_output = self.lstn.LSAB_forward(curr_enc_embs, None, None, curr_id_emb, pos_emb=self.pos_emb, size_2d=self.enc_size_2d)
        lsab_embs, lsab_curr_memories, lsab_long_memories, lsab_short_memories = self.curr_lsab_output

        if self.long_term_memories is None:
            self.long_term_memories = lsab_long_memories
        else:
            self.update_long_term_memory(lsab_long_memories)

        self.last_mem_step = self.frame_step
        self.short_term_memories_list = [lsab_short_memories]
        self.short_term_memories = lsab_short_memories


    def encode_one_img_mask(self, img=None, mask=None, frame_step=-1):
        if frame_step == -1:
            frame_step = self.frame_step

        if self.enable_offline_enc:  # 一开始经过offline_encoder这块就会变为True
            curr_enc_embs = self.offline_enc_embs[frame_step]
        elif img is None:
            curr_enc_embs = None
        else:
            curr_enc_embs = self.lstn.encode_image(img)

        if mask is not None:
            curr_one_hot_mask = one_hot_mask(mask, self.max_obj_num)
        elif self.enable_offline_enc:
            curr_one_hot_mask = self.offline_one_hot_masks[frame_step]
        else:
            curr_one_hot_mask = None

        return curr_enc_embs, curr_one_hot_mask


    def assign_identity(self, one_hot_mask):
        if self.enable_id_shuffle:  # [2, 11, 465, 465] * [2, 11, 11](这是一个每行只有一个1的矩阵)
            one_hot_mask = torch.einsum('bohw, bot->bthw', one_hot_mask, self.id_shuffle_matrix)
        # [900, 2, 256]
        id_emb = self.lstn.get_id_emb(one_hot_mask).view(self.batch_size, -1, self.enc_hw).permute(2, 0, 1)  # [900, 2, 256]

        if self.training and self.freeze_id:
            id_emb = id_emb.detach()

        return id_emb


    def generate_loss_mask(self, gt_mask, step, return_prob=False):
        self.decode_current_logits()  # 网络输出最终的预测值，是float的连续变量
        loss = self.calculate_current_loss(gt_mask, step)
        if return_prob:
            mask, prob = self.predict_current_mask(return_prob=True)
            return loss, mask, prob
        else:
            mask = self.predict_current_mask()  # 根据当前的预测返回obj编号，离散变量
            return loss, mask


    def decode_current_logits(self, output_size=None):
        curr_enc_embs = self.curr_enc_embs
        curr_lsab_embs = self.curr_lsab_output[0]
        pred_id_logits = self.lstn.decode_id_logits(curr_lsab_embs, curr_enc_embs)  # [2, 11, 117, 117]
        if self.enable_id_shuffle:  # reverse shuffle
            pred_id_logits = torch.einsum('bohw,bto->bthw', pred_id_logits, self.id_shuffle_matrix)  # [2, 11, 117, 117]

        # remove unused identities
        for batch_idx, obj_num in enumerate(self.obj_nums):  # 将无目标的地方设为-1e+4表示预测概率为0
            pred_id_logits[batch_idx, (obj_num+1):] = -1e+10 if pred_id_logits.dtype == torch.float32 else -1e+4
        self.pred_id_logits = pred_id_logits

        if output_size is not None:
            pred_id_logits = F.interpolate(pred_id_logits, size=output_size, mode="bilinear", align_corners=self.align_corners)

        return pred_id_logits


    def calculate_current_loss(self, gt_mask, step):
        pred_id_logits = self.pred_id_logits
        pred_id_logits = F.interpolate(pred_id_logits, size=gt_mask.size()[-2:], mode="bilinear", align_corners=self.align_corners)

        label_list = []
        logit_list = []
        for batch_idx, obj_num in enumerate(self.obj_nums):
            now_label = gt_mask[batch_idx].long()  # [1, 456, 456]
            now_logit = pred_id_logits[batch_idx, :(obj_num+1)].unsqueeze(0)  # [1, obj_num+1, 456, 456]
            label_list.append(now_label.long())
            logit_list.append(now_logit)

        total_loss = 0
        for loss, loss_weight in zip(self.losses, self.loss_weights):
            total_loss = total_loss + loss_weight*loss(logit_list, label_list, step)
        return total_loss


    def predict_current_mask(self, output_size=None, return_prob=False):
        if output_size is None:
            output_size = self.input_size_2d

        pred_id_logits = F.interpolate(self.pred_id_logits, size=output_size, mode="bilinear", align_corners=self.align_corners)
        pred_mask = torch.argmax(pred_id_logits, dim=1)  # [2, 11, 465, 465]

        if not return_prob:
            return pred_mask
        else:
            pred_prob = torch.softmax(pred_id_logits, dim=1)
            return pred_mask, pred_prob


    def match_propogate_one_frame(self, img=None, img_embs=None):
        self.frame_step += 1
        if img_embs is None:
            curr_enc_embs, _ = self.encode_one_img_mask(img, None, self.frame_step)
        else:
            curr_enc_embs = img_embs
        self.curr_enc_embs = curr_enc_embs
        self.curr_lsab_output = self.lstn.LSAB_forward(curr_enc_embs, self.long_term_memories, self.short_term_memories, None,
                                                      pos_emb=self.pos_emb, size_2d=self.enc_size_2d)


    def update_short_term_memory(self, curr_mask, curr_id_emb=None):
        if curr_id_emb is None:
            if len(curr_mask.size()) == 3 or curr_mask.size()[0] == 1:
                curr_one_hot_mask = one_hot_mask(curr_mask, self.max_obj_num)
            else:
                curr_one_hot_mask = curr_mask
            curr_id_emb = self.assign_identity(curr_one_hot_mask)

        lsab_curr_memories = self.curr_lsab_output[1]  # [[900, 2, 256], [900, 2, 256]]
        lsab_curr_memories_2d = []
        for layer_idx in range(len(lsab_curr_memories)):
            curr_k, curr_v = lsab_curr_memories[layer_idx][0], lsab_curr_memories[layer_idx][1]
            curr_k, curr_v = self.lstn.LSAB.layers[layer_idx].fuse_key_value_id(curr_k, curr_v, curr_id_emb)  # k不变，v加上id_embed
            lsab_curr_memories[layer_idx][0], lsab_curr_memories[layer_idx][1] = curr_k, curr_v
            lsab_curr_memories_2d.append([
                seq_to_2d(lsab_curr_memories[layer_idx][0], self.enc_size_2d),
                seq_to_2d(lsab_curr_memories[layer_idx][1], self.enc_size_2d)
            ])

        self.short_term_memories_list.append(lsab_curr_memories_2d)
        self.short_term_memories_list = self.short_term_memories_list[-self.short_term_mem_skip:]
        self.short_term_memories = self.short_term_memories_list[0]

        if self.frame_step - self.last_mem_step >= self.long_term_mem_gap:
            self.update_long_term_memory(lsab_curr_memories)
            self.last_mem_step = self.frame_step







    def update_long_term_memory(self, new_long_term_memories):
        if self.long_term_memories is None:
            self.long_term_memories = new_long_term_memories
        updated_long_term_memories = []
        for new_long_term_memory, last_long_term_memory in zip(new_long_term_memories, self.long_term_memories):
            updated_e = []
            for new_e, last_e in zip(new_long_term_memory, last_long_term_memory):
                if new_e is None or last_e is None:
                    updated_e.append(None)
                else:
                    updated_e.append(torch.cat([new_e, last_e], dim=0))
            updated_long_term_memories.append(updated_e)
        self.long_term_memories = updated_long_term_memories


    def keep_gt_mask(self, pred_mask, keep_prob=0.2):
        pred_mask = pred_mask.float()
        gt_mask = self.offline_masks[self.frame_step].float().squeeze(1)

        shape = [1 for _ in range(pred_mask.ndim)]
        shape[0] = self.batch_size
        random_tensor = keep_prob + torch.rand(shape, dtype=pred_mask.dtype, device=pred_mask.device)
        random_tensor.floor_()  # binarize

        pred_mask = pred_mask * (1 - random_tensor) + gt_mask * random_tensor

        return pred_mask

    def restart_engine(self, batch_size=1, enable_id_shuffle=False):

        self.batch_size = batch_size
        self.frame_step = 0
        self.last_mem_step = -1
        self.enable_id_shuffle = enable_id_shuffle
        self.freeze_id = False

        self.obj_nums = None
        self.pos_emb = None
        self.enc_size_2d = None
        self.enc_hw = None
        self.input_size_2d = None

        self.long_term_memories = None
        self.short_term_memories_list = []
        self.short_term_memories = None

        self.enable_offline_enc = False
        self.offline_enc_embs = None
        self.offline_one_hot_masks = None
        self.offline_frames = -1
        self.total_offline_frame_num = 0

        self.curr_enc_embs = None
        self.curr_memories = None
        self.curr_id_embs = None

        if enable_id_shuffle:
            self.id_shuffle_matrix = generate_permute_matrix(self.max_obj_num+1, batch_size, gpu_id=self.gpu_id)
        else:
            self.id_shuffle_matrix = None

    def update_size(self, input_size, enc_size):
        self.input_size_2d = input_size  # [456, 456]
        self.enc_size_2d = enc_size  # [30, 30]
        self.enc_hw = self.enc_size_2d[0] * self.enc_size_2d[1]


class LSTNInferEngine(nn.Module):
    def __init__(self, lstn_model, gpu_id=0, long_term_mem_gap=9999, short_term_mem_skip=1, max_lstn_obj_num=None):
        super().__init__()

        self.cfg = lstn_model.cfg
        self.lstn = lstn_model

        if max_lstn_obj_num is None or max_lstn_obj_num > lstn_model.max_obj_num:
            self.max_lstn_obj_num = lstn_model.max_obj_num
        else:
            self.max_lstn_obj_num = max_lstn_obj_num

        self.gpu_id = gpu_id
        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip

        self.lstn_engines = []

        self.restart_engine()

    def restart_engine(self):
        del (self.lstn_engines)
        self.lstn_engines = []
        self.obj_nums = None

    def separate_mask(self, mask, obj_nums):
        if mask is None:
            return [None] * len(self.lstn_engines)
        if len(self.lstn_engines) == 1:
            return [mask], [obj_nums]

        separated_obj_nums = [self.max_lstn_obj_num for _ in range(len(self.lstn_engines))]
        if obj_nums % self.max_lstn_obj_num > 0:
            separated_obj_nums[-1] = obj_nums % self.max_lstn_obj_num

        if len(mask.size()) == 3 or mask.size()[0] == 1:
            separated_masks = []
            for idx in range(len(self.lstn_engines)):
                start_id = idx * self.max_lstn_obj_num + 1
                end_id = (idx + 1) * self.max_lstn_obj_num
                fg_mask = ((mask >= start_id) & (mask <= end_id)).float()
                separated_mask = (fg_mask * mask - start_id + 1) * fg_mask
                separated_masks.append(separated_mask)
            return separated_masks, separated_obj_nums
        else:
            prob = mask
            separated_probs = []
            for idx in range(len(self.lstn_engines)):
                start_id = idx * self.max_lstn_obj_num + 1
                end_id = (idx + 1) * self.max_lstn_obj_num
                fg_prob = prob[start_id:(end_id + 1)]
                bg_prob = 1. - torch.sum(fg_prob, dim=1, keepdim=True)
                separated_probs.append(torch.cat([bg_prob, fg_prob], dim=1))
            return separated_probs, separated_obj_nums


    def soft_logit_aggregation(self, all_logits):
        if len(all_logits) == 1:
            return all_logits[0]

        fg_probs = []
        bg_probs = []

        for logit in all_logits:
            prob = torch.softmax(logit, dim=1)
            bg_probs.append(prob[:, 0:1])
            fg_probs.append(prob[:, 1:1 + self.max_lstn_obj_num])

        bg_prob = torch.prod(torch.cat(bg_probs, dim=1), dim=1, keepdim=True)
        merged_prob = torch.cat([bg_prob]+fg_probs, dim=1).clamp(1e-5, 1 - 1e-5)
        merged_logit = torch.logit(merged_prob)

        return merged_logit

    def add_reference_frame(self, img, mask, obj_nums, frame_step=-1):
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]
        self.obj_nums = obj_nums
        lstn_num = max(np.ceil(obj_nums / self.max_lstn_obj_num), 1)
        while (lstn_num > len(self.lstn_engines)):
            new_engine = LSTNEngine(self.lstn, self.gpu_id, self.long_term_mem_gap, self.short_term_mem_skip)
            new_engine.eval()
            self.lstn_engines.append(new_engine)

        separated_masks, separated_obj_nums = self.separate_mask(mask, obj_nums)
        img_embs = None
        for lstn_engine, separated_mask, separated_obj_num in zip(self.lstn_engines, separated_masks, separated_obj_nums):
            lstn_engine.add_reference_frame(img, separated_mask, obj_nums=[separated_obj_num], frame_step=frame_step, img_embs=img_embs)
            if img_embs is None:  # reuse image embeddings
                img_embs = lstn_engine.curr_enc_embs

        self.update_size()

    def match_propogate_one_frame(self, img=None):
        img_embs = None
        for lstn_engine in self.lstn_engines:
            lstn_engine.match_propogate_one_frame(img, img_embs=img_embs)
            if img_embs is None:  # reuse image embeddings
                img_embs = lstn_engine.curr_enc_embs

    def decode_current_logits(self, output_size=None):
        all_logits = []
        for lstn_engine in self.lstn_engines:
            all_logits.append(lstn_engine.decode_current_logits(output_size))
        pred_id_logits = self.soft_logit_aggregation(all_logits)
        return pred_id_logits

    def update_memory(self, curr_mask):
        separated_masks, _ = self.separate_mask(curr_mask, self.obj_nums)
        for lstn_engine, separated_mask in zip(self.lstn_engines, separated_masks):
            lstn_engine.update_short_term_memory(separated_mask)

    def update_size(self):
        self.input_size_2d = self.lstn_engines[0].input_size_2d
        self.enc_size_2d = self.lstn_engines[0].enc_size_2d
        self.enc_hw = self.lstn_engines[0].enc_hw
