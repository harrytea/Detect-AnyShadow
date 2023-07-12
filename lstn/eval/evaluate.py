import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
parent_dir_path2 = os.path.abspath(os.path.join(parent_dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
sys.path.insert(0, parent_dir_path2)

import time
import datetime
import argparse
import numpy as np
from PIL import Image
from eval.misc import AvgMeter
from eval.misc import cal_fmeasure, cal_Jaccard, cal_BER, cal_precision_recall, cal_mae


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=20000)
parser.add_argument('--pred_path', type=str, default="./results/lstn_LSTNB/41")
parser.add_argument('--ema', action='store_true')
parser.set_defaults(ema=False)
args = parser.parse_args()

gt_path = '/data/wangyh/data4/Datasets/shadow/video_new/visha2/test/labels'

args.save_path = args.pred_path
pred_path = os.path.join(args.pred_path, "lstn_LSTNB_ckpt_{}".format(str(args.epoch)))
if args.ema:
    log_path = os.path.join(args.save_path, 'results_ema.txt')
else:
    log_path = os.path.join(args.save_path, 'results.txt')
open(log_path, 'a+').write(str(datetime.datetime.now()) + '\n')


# =====================================================================================================
precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
mae_record = AvgMeter()
Jaccard_record = AvgMeter()
BER_record = AvgMeter()
shadow_BER_record = AvgMeter()
non_shadow_BER_record = AvgMeter()

start_time = time.time()
video_list = sorted(os.listdir(gt_path))
for vidx, video in enumerate(video_list):
    gt_list = sorted(os.listdir(os.path.join(gt_path, video)))
    # pred_list = sorted(os.listdir(os.path.join(pred_path, video)))
    for frame in gt_list:
        pred = np.array(Image.open(os.path.join(pred_path, video, frame)))
        gt = np.array(Image.open(os.path.join(gt_path, video, frame)))
        mae = cal_mae(pred, gt)  ##### MAE
        Jaccard = cal_Jaccard(pred, gt)  ##### Jaccard
        BER, shadow_BER, non_shadow_BER = cal_BER(pred, gt)  ##### BER

        precision, recall = cal_precision_recall(pred, gt)
        for pidx, pdata in enumerate(zip(precision, recall)):
            p, r = pdata
            recall_record[pidx].update(r)
            precision_record[pidx].update(p)

        mae_record.update(mae)
        BER_record.update(BER)
        Jaccard_record.update(Jaccard)
        shadow_BER_record.update(shadow_BER)
        non_shadow_BER_record.update(non_shadow_BER)
    print("Epoch:{}, {}/{}, Video:{} end".format(args.epoch, vidx, len(video_list), video))
# =====================================================================================================


fmeasure = cal_fmeasure([precord.avg for precord in precision_record], [rrecord.avg for rrecord in recall_record])
log = 'Epoch:{}, MAE:{}, F-beta:{}, Jaccard:{}, BER:{}, SBER:{}, non-SBER:{}'.format(\
                                                        args.epoch, mae_record.avg, fmeasure, Jaccard_record.avg, \
                                                        BER_record.avg, shadow_BER_record.avg, non_shadow_BER_record.avg)
open(log_path, 'a+').write(log + '\n' + str(datetime.datetime.now()) + '\n')
end_time = time.time()
open(log_path, 'a+').write("time consume: " + str(end_time-start_time) + "\n")
print(log)


