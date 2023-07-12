from __future__ import division

import os
import cv2
import numpy as np
from torch.utils.data import Dataset
cv2.setNumThreads(0)


class VSDTrain(Dataset):
    def __init__(self,
                 image_root,
                 label_root,
                 imglistdic,
                 transform=None,
                 rgb=True,
                 repeat_time=1,
                 rand_gap=3,
                 seq_len=5,
                 rand_reverse=True,
                 dynamic_merge=True,
                 max_obj_n=10):
        self.image_root = image_root
        self.label_root = label_root
        self.rand_gap = rand_gap
        self.seq_len = seq_len
        self.rand_reverse = rand_reverse
        self.repeat_time = repeat_time
        self.transform = transform
        self.dynamic_merge = dynamic_merge
        self.max_obj_n = max_obj_n
        self.rgb = rgb
        self.imglistdic = imglistdic
        self.seqs = list(self.imglistdic.keys())
        print('Video Num: {} X {}'.format(len(self.seqs), self.repeat_time))

    def __len__(self):
        return int(len(self.seqs) * self.repeat_time)

    def reverse_seq(self, imagelist, lablist):
        if np.random.randint(2) == 1:
            imagelist = imagelist[::-1]
            lablist = lablist[::-1]
        return imagelist, lablist

    def get_ref_index_v2(self, seqname, lablist, min_fg_pixels=200, max_try=20, total_gap=0):
        search_range = len(lablist) - total_gap
        if search_range <= 1:
            return 0
        bad_indices = []
        for _ in range(max_try):
            ref_index = np.random.randint(search_range)
            if ref_index in bad_indices:
                continue
            ref_label =  np.load(os.path.join(self.label_root, seqname, lablist[ref_index]))
            # ref_label = np.array(ref_label, dtype=np.uint8)
            xs, ys = np.nonzero(ref_label)
            if len(xs) > min_fg_pixels:
                break
            bad_indices.append(ref_index)
        return ref_index

    def get_curr_gaps(self, seq_len, max_gap=999, max_try=10, short_video=False):
        temp = self.rand_gap
        if short_video == True:
            self.rand_gap = 1
        for _ in range(max_try):
            curr_gaps = []
            total_gap = 0
            for _ in range(seq_len):
                gap = int(np.random.randint(self.rand_gap)+1)  # gap在 [1, rand_gap] 之间
                total_gap += gap
                curr_gaps.append(gap)
            if total_gap <= max_gap:
                break
        self.rand_gap = temp
        # print(self.rand_gap) # # # # # #  ##########################################################
        return curr_gaps, total_gap

    def check_index(self, total_len, index, allow_reflect=True):
        if total_len <= 1:
            return 0

        if index < 0:
            if allow_reflect:
                index = -index
                index = self.check_index(total_len, index, True)
            else:
                index = 0
        elif index >= total_len:
            if allow_reflect:
                index = 2 * (total_len - 1) - index
                index = self.check_index(total_len, index, True)
            else:
                index = total_len - 1

        return index

    def get_curr_indices(self, lablist, prev_index, gaps):
        total_len = len(lablist)
        curr_indices = []
        now_index = prev_index
        for gap in gaps:
            now_index += gap
            curr_indices.append(self.check_index(total_len, now_index))
        return curr_indices

    def get_image_label(self, seqname, imagelist, lablist, index):
        image =  np.load(os.path.join(self.image_root, seqname, imagelist[index]))
        # image = np.array(image, dtype=np.float32)

        if self.rgb:
            image = image[:, :, [2, 1, 0]]

        label =  np.load(os.path.join(self.label_root, seqname, lablist[index]))
        # label = np.array(label, dtype=np.uint8)

        return image, label

    def sample_sequence(self, idx):
        idx = idx % len(self.seqs)
        seqname = self.seqs[idx]  # choice video
        imagelist, lablist = self.imglistdic[seqname]  # get image and label list
        frame_num = len(imagelist)
        if self.rand_reverse:
            imagelist, lablist = self.reverse_seq(imagelist, lablist)

        is_consistent = False
        max_try = 5
        try_step = 0
        while (is_consistent is False and try_step < max_try):
            try_step += 1

            # generate random gaps
            if seqname=='Bikeshow_ce' or seqname=='MotorRolling':
                curr_gaps, total_gap = self.get_curr_gaps(self.seq_len-1, short_video=True)
            else:
                curr_gaps, total_gap = self.get_curr_gaps(self.seq_len-1)


            # prev frame is next to ref frame
            # get ref frame
            ref_index = self.get_ref_index_v2(seqname, lablist)
            ref_image, ref_label = self.get_image_label(seqname, imagelist, lablist, ref_index)
            ref_label = np.uint8(ref_label / 255.)  # convert to 0-1 # add ------------------------------------
            ref_objs = list(np.unique(ref_label))

            # get curr frames
            curr_indices = self.get_curr_indices(lablist, ref_index, curr_gaps)
            curr_images, curr_labels, curr_objs = [], [], []
            for curr_index in curr_indices:
                curr_image, curr_label = self.get_image_label(seqname, imagelist, lablist, curr_index)
                curr_label = np.uint8(curr_label / 255.)  # convert to 0-1 # add ------------------------------------
                c_objs = list(np.unique(curr_label))
                curr_images.append(curr_image)
                curr_labels.append(curr_label)
                curr_objs.extend(c_objs)

            objs = list(np.unique(curr_objs))
            prev_image, prev_label = curr_images[0], curr_labels[0]
            curr_images, curr_labels = curr_images[1:], curr_labels[1:]

            is_consistent = True
            for obj in objs:
                if obj == 0:
                    continue
                if obj not in ref_objs:
                    is_consistent = False
                    break

        # get meta info
        obj_num = list(np.sort(ref_objs))[-1]

        sample = {
            'ref_img': ref_image,
            'prev_img': prev_image,
            'curr_img': curr_images,
            'ref_label': ref_label,
            'prev_label': prev_label,
            'curr_label': curr_labels
        }
        sample['meta'] = {
            'seq_name': seqname,
            'frame_num': frame_num,
            'obj_num': obj_num
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __getitem__(self, idx):
        sample = self.sample_sequence(idx)
        return sample




class ViSha(VSDTrain):
    def __init__(self,
                 split=['train'],
                 root='./DAVIS',
                 transform=None,
                 rgb=True,
                 repeat_time=1,
                 rand_gap=3,
                 seq_len=5,
                 rand_reverse=True,
                 dynamic_merge=True,
                 max_obj_n=10,):

        image_root = os.path.join(root, split[0], 'images')
        label_root = os.path.join(root, split[0], 'labels')
        seq_names = [video for video in os.listdir(image_root)]
        imglistdic = {}
        for seq_name in seq_names:
            images = list(np.sort(os.listdir(os.path.join(image_root, seq_name))))
            labels = list(np.sort(os.listdir(os.path.join(label_root, seq_name))))
            imglistdic[seq_name] = (images, labels)

        super(ViSha, self).__init__(image_root,
                                    label_root,
                                    imglistdic,
                                    transform,
                                    rgb,
                                    repeat_time,
                                    rand_gap,
                                    seq_len,
                                    rand_reverse,
                                    dynamic_merge,
                                    max_obj_n=max_obj_n)