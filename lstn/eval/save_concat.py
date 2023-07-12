# 从每个video中随机抽取
import sys
sys.path.append("../")

import os
import os.path as osp
import numpy as np
from PIL import Image
from util import *
import argparse
from torchvision import transforms
from torchvision import utils

res_path = r"../checkpoints/results"
save_path = r"../checkpoints/simple"

img_path = r'/data4/wangyh/Datasets/shadow/visha/test/images'
gt_path = r'/data4/wangyh/Datasets/shadow/visha/test/labels'

parser = argparse.ArgumentParser()
parser.add_argument('--start_epoch', type=int, default=1000)
parser.add_argument('--end_epoch', type=int, default=15000)
args = parser.parse_args()

to_tensor = transforms.ToTensor()
mkdir_or_exist([save_path])
for epoch in range(args.start_epoch, args.end_epoch+1, 1000):
    mkdir_or_exist([osp.join(save_path, 'predict_{}'.format(epoch))])
    for video in os.listdir(osp.join(res_path, 'predict_{}'.format(epoch))):
        video_path = osp.join(res_path, 'predict_{}'.format(epoch), video)
        clip = sorted(os.listdir(video_path))
        choice_clip = np.random.choice(clip, size=8, replace=False)

        # img and first clip gt
        first_clip = choice_clip[0]
        img = np.array(Image.open(osp.join(img_path, video, first_clip.split('.')[0]+'.jpg')))  # 获取gt img
        gt_mask = np.array(Image.open(os.path.join(gt_path, video, first_clip)))  # 获取gt mask
        clip1 = np.array(Image.open(os.path.join(res_path, 'predict_{}'.format(epoch), video, choice_clip[0])))  # 获取gt mask
        clip2 = np.array(Image.open(os.path.join(res_path, 'predict_{}'.format(epoch), video, choice_clip[1])))  # 获取gt mask
        clip3 = np.array(Image.open(os.path.join(res_path, 'predict_{}'.format(epoch), video, choice_clip[2])))  # 获取gt mask
        clip4 = np.array(Image.open(os.path.join(res_path, 'predict_{}'.format(epoch), video, choice_clip[3])))  # 获取gt mask
        clip5 = np.array(Image.open(os.path.join(res_path, 'predict_{}'.format(epoch), video, choice_clip[4])))  # 获取gt mask
        clip6 = np.array(Image.open(os.path.join(res_path, 'predict_{}'.format(epoch), video, choice_clip[5])))  # 获取gt mask
        clip7 = np.array(Image.open(os.path.join(res_path, 'predict_{}'.format(epoch), video, choice_clip[6])))  # 获取gt mask
        clip8 = np.array(Image.open(os.path.join(res_path, 'predict_{}'.format(epoch), video, choice_clip[7])))  # 获取gt mask

        img = to_tensor(img)
        gt_mask = to_tensor(np.tile(gt_mask, (3,1,1)).transpose(1,2,0))
        clip1 = to_tensor(np.tile(clip1, (3,1,1)).transpose(1,2,0))
        clip2 = to_tensor(np.tile(clip2, (3,1,1)).transpose(1,2,0))
        clip3 = to_tensor(np.tile(clip3, (3,1,1)).transpose(1,2,0))
        clip4 = to_tensor(np.tile(clip4, (3,1,1)).transpose(1,2,0))
        clip5 = to_tensor(np.tile(clip5, (3,1,1)).transpose(1,2,0))
        clip6 = to_tensor(np.tile(clip6, (3,1,1)).transpose(1,2,0))
        clip7 = to_tensor(np.tile(clip7, (3,1,1)).transpose(1,2,0))
        clip8 = to_tensor(np.tile(clip8, (3,1,1)).transpose(1,2,0))

        img_all = torch.stack([img, gt_mask, clip1, clip2, clip3, clip4, clip5, clip6, clip7, clip8], dim=0)
        img_all = utils.make_grid(img_all, nrow=5, padding=20, pad_value=1)
        # img_all = img_all / 2 + 0.5     # unnormalize
        img_all = (img_all.numpy()*255).astype('uint8').transpose(1,2,0)
        img_all = Image.fromarray(img_all)
        img_all.save(osp.join(save_path, 'predict_{}'.format(epoch), str(video)+'.jpg'))