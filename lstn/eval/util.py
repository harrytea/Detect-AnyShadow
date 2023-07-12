import os
import os.path as osp
from pathlib import Path

import torch
import numpy as np
from pathlib import Path
import torch.distributed as dist

# import yaml
import random
# import imageio as io
from collections import OrderedDict

'''  initial seed  '''
def init_seeds(seed=0):
    random.seed(seed)  # seed for module random
    np.random.seed(seed)  # seed for numpy
    torch.manual_seed(seed)  # seed for PyTorch CPU
    torch.cuda.manual_seed(seed)  # seed for current PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # seed for all PyTorch GPUs
    if seed == 0:
        # if True, causes cuDNN to only use deterministic convolution algorithms.
        torch.backends.cudnn.deterministic = True
        # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
        torch.backends.cudnn.benchmark = False


'''  distributed mode  '''
def initial_distributed():
    '''  initial distributed mode  '''
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training")
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        gpu = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print("os.environ[\"WORLD_SIZE\"]: ", os.environ["WORLD_SIZE"])
        print("os.environ[\"RANK\"]: ", os.environ["RANK"])
        print("os.environ[\"LOCAL_RANK\"]: ", os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
    torch.cuda.set_device(rank)
    dist_url = 'env://'
    dis_backend = 'nccl'  # communication: nvidia GPU recommened nccl
    print('| distributed init (rank {}): {}'.format(rank, dist_url), flush=True)
    dist.init_process_group(backend=dis_backend, init_method=dist_url, world_size=world_size, rank=rank)
    dist.barrier()
    return rank


'''  log info  '''
def load_config(file_name):
    path = Path(__file__).parent.parent/file_name
    return yaml.load(open(path, "r"), Loader=yaml.FullLoader)


'''  conver dict to object  '''
class dic2obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [dic2obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, dic2obj(b) if isinstance(b, dict) else b)


def save_mask(path_dir, step, image_size, img_lists: tuple):
        data, label, predict = img_lists

        # process data and label
        data, label= (data.numpy()*255).astype('uint8'), (label.numpy()*255).astype('uint8')
        label = np.tile(label, (3,1,1))
        h, w = image_size, image_size

        # process predicts
        predicts = []
        for pred in predict:
            pred = (np.tile(pred.cpu().data * 255,(3,1,1))).astype('uint8')
            predicts.append(pred)

        # save image
        gen_num = (2, 1)  # save two example images
        per_img_num = len(predicts)+2
        img = np.zeros((gen_num[0]*h, gen_num[1]*(per_img_num)*w, 3)).astype('uint8')
        for _ in img_lists:
            for i in range(gen_num[0]):  # i row
                row = i * h
                for j in range(gen_num[1]):  # j col
                    idx = i * gen_num[1] + j
                    # save data && gt mask
                    pred_mask_list = [p[idx] for p in predicts]
                    tmp_list = [data[idx], label[idx]] + pred_mask_list
                    for k in range(per_img_num):
                        col = (j * 3 + k) * w
                        tmp = np.transpose(tmp_list[k], (1, 2, 0))
                        # print(tmp.shape)
                        img[row: row+h, col: col+w] = tmp

        img_file = os.path.join(path_dir, '%d.jpg'%(step))
        io.imsave(img_file, img)


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    for dir in dir_name:
        dir = osp.expanduser(dir)
        os.makedirs(dir, mode=mode, exist_ok=True)


def load_checkpoint(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)












def save_networks(epoch, net, path):
    save_filename = '%s_net.pth' % (epoch)
    save_path = os.path.join(path, save_filename)
    torch.save(net.state_dict(), save_path)


def save_results_image(opt, step, input, output, gt, path):
    input = tensor2im(input)
    output = tensor2im(output)
    gt = tensor2im(gt)
    img = np.concatenate((input, output, gt), axis=1)
    mkdir_or_exist(os.path.join(opt['results_dir'], str(step)))
    img_file = os.path.join(opt['results_dir'], str(step), "%s" % (path[0].split('/')[-1]))
    io.imsave(img_file, output)
    # self.logger.info("save: %s" % (img_file))


def save_image(opt, step, input, output, gt, path):
    input = tensor2im(input)
    output = tensor2im(output)
    gt = tensor2im(gt)
    img = np.concatenate((input, output, gt), axis=1)
    img_file = os.path.join(opt['image_dir'], "iter%d_%s"%(step, path[0].split('/')[-1]))
    io.imsave(img_file, img)
    # self.logger.info("save: %s" % (img_file))


def tensor2im(input_image, imtype=np.uint8):
    if len(input_image.shape)<3: return None
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    input_image = input_image.clamp(0,1)
    image_numpy = image_tensor.data[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy[image_numpy<0] = 0
    image_numpy[image_numpy>255] = 255
    return image_numpy.astype(imtype)









