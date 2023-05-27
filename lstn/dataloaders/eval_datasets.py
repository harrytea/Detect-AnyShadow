from __future__ import division
import os
import shutil
import json
import cv2
from PIL import Image

import numpy as np
from torch.utils.data import Dataset


class VSDTest(Dataset):
    def __init__(self, image_list, mask_list, rgb=True, transform=None,):
        self.image_list = image_list
        self.label_list = mask_list
        self.obj_num = 1
        self.transform = transform
        self.rgb = rgb


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_array = self.image_list[idx]
        height, width, channels = img_array.shape
        obj_idx = [0, 1]

        if idx==0:
            label_array = self.label_list[0]
            sample = {'current_img': img_array, 'current_label': label_array}
        else:
            sample = {'current_img': img_array}

        sample['meta'] = {
            'obj_num': 1,
            'height': height,
            'width': width,
            'obj_idx': obj_idx
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample