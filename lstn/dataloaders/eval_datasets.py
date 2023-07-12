from __future__ import division
import os
import shutil
import json
import cv2
from PIL import Image

import numpy as np
from torch.utils.data import Dataset


class VSDTest(Dataset):
    def __init__(self, image_root, label_root, seq_name, images, labels, rgb=True, transform=None,):
        self.image_root = image_root
        self.label_root = label_root
        self.seq_name = seq_name
        self.images = images
        self.labels = labels
        self.obj_num = 1
        self.num_frame = len(self.images)
        self.transform = transform
        self.rgb = rgb

        self.obj_nums = []
        self.obj_indices = []

        # 这一部分输出每一帧有多少个obj
        curr_objs = [0]
        for img_name in self.images:
            self.obj_nums.append(len(curr_objs) - 1)
            current_label_name = img_name.split('.')[0] + '.png'
            if current_label_name in self.labels:
                current_label = self.read_label(current_label_name)
                curr_obj = list(np.unique(current_label))
                for obj_idx in curr_obj:
                    if obj_idx not in curr_objs:
                        curr_objs.append(obj_idx)
            self.obj_indices.append(curr_objs.copy())

        self.obj_nums[0] = self.obj_nums[1]

    def __len__(self):
        return len(self.images)

    def read_image(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_root, self.seq_name, img_name)
        img = cv2.imread(img_path)
        img = np.array(img, dtype=np.float32)
        if self.rgb:
            img = img[:, :, [2, 1, 0]]
        return img

    def read_label(self, label_name, squeeze_idx=None):
        label_path = os.path.join(self.label_root, self.seq_name, label_name)
        label = Image.open(label_path)
        label = np.array(label, dtype=np.uint8)
        label = (label > 0).astype(np.uint8)
        
        return label

    def __getitem__(self, idx):
        img_name = self.images[idx]
        current_img = self.read_image(idx)
        height, width, channels = current_img.shape
        current_label_name = img_name.split('.')[0] + '.png'
        obj_num = self.obj_nums[idx]
        obj_idx = self.obj_indices[idx]

        if current_label_name in self.labels:
            current_label = self.read_label(current_label_name, obj_idx)
            sample = {'current_img': current_img, 'current_label': current_label}
        else:
            sample = {'current_img': current_img}

        sample['meta'] = {
            'seq_name': self.seq_name,
            'frame_num': self.num_frame,
            'obj_num': obj_num,
            'current_name': img_name,
            'height': height,
            'width': width,
            'flip': False,
            'obj_idx': obj_idx
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample



class ViSha_Test(object):
    def __init__(self, root='./visha', transform=None, rgb=True, result_root=None):
        self.transform = transform
        self.rgb = rgb
        self.result_root = result_root

        self.image_root = os.path.join(root, 'test', 'images')
        self.label_root = os.path.join(root, 'test', 'labels')
        seq_names = [video for video in os.listdir(self.image_root)]
        self.seqs = list(np.unique(seq_names))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_name = self.seqs[idx]
        images = list(np.sort(os.listdir(os.path.join(self.image_root, seq_name))))  # image list
        labels = [images[0].replace('jpg', 'png')]  # label 0

        # check if the mask of frame is exits, if not, copy to the target folder
        if not os.path.isfile(os.path.join(self.result_root, seq_name, labels[0])):
            seq_result_folder = os.path.join(self.result_root, seq_name)
            if not os.path.exists(seq_result_folder):
                os.makedirs(seq_result_folder)
            source_label_path = os.path.join(self.label_root, seq_name, labels[0])
            result_label_path = os.path.join(self.result_root, seq_name, labels[0])
            shutil.copy(source_label_path, result_label_path)

        seq_dataset = VSDTest(self.image_root,
                              self.label_root,
                              seq_name,
                              images,
                              labels,
                              transform=self.transform,
                              rgb=self.rgb,)
        return seq_dataset