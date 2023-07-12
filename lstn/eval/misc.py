import os
import numpy as np
from medpy import metric

class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cal_mae(prediction, gt):
    # input should be np array with data type uint8
    assert prediction.dtype == np.uint8
    assert gt.dtype == np.uint8
    assert prediction.shape == gt.shape

    prediction = prediction / 255.
    gt = gt / 255.
    mae = np.mean(np.abs(prediction-gt))
    return mae

def cal_Jaccard(prediction, gt):
    # input should be np array with data type uint8
    assert prediction.dtype == np.uint8
    assert gt.dtype == np.uint8
    assert prediction.shape == gt.shape

    prediction = prediction / 255.
    gt = gt / 255.
    pred = (prediction > 0.5)
    gt = (gt > 0.5)
    Jaccard = metric.binary.jc(pred, gt)
    return Jaccard


def cal_BER(prediction, label, thr=127.5):
    prediction = (prediction > thr)
    label = (label > thr)
    prediction_tmp = prediction.astype(np.float)
    label_tmp = label.astype(np.float)
    TP = np.sum(prediction_tmp * label_tmp)
    TN = np.sum((1 - prediction_tmp) * (1 - label_tmp))
    Np = np.sum(label_tmp)
    Nn = np.sum((1-label_tmp))
    BER = 0.5 * (2 - TP / Np - TN / Nn) * 100
    shadow_BER = (1 - TP / Np) * 100
    non_shadow_BER = (1 - TN / Nn) * 100
    return BER, shadow_BER, non_shadow_BER


def cal_precision_recall(prediction, gt):
    assert prediction.dtype == np.uint8
    assert gt.dtype == np.uint8
    assert prediction.shape == gt.shape

    gt = gt / 255.
    prediction = prediction / 255.
    hard_gt = np.zeros(prediction.shape)
    hard_gt[gt>0.5] = 1
    t = np.sum(hard_gt)

    eps = 1e-4
    precision, recall = [], []
    # calculating precision and recall at 255 different binarizing thresholds
    for threshold in range(256):
        hard_prediction = np.zeros(prediction.shape)
        threshold = threshold / 255.
        hard_prediction[prediction > threshold] = 1
        tp = np.sum(hard_prediction * hard_gt)
        p = np.sum(hard_prediction)
        precision.append((tp + eps) / (p + eps))
        recall.append((tp + eps) / (t + eps))
    return precision, recall


def cal_fmeasure(precision, recall):
    assert len(precision) == 256
    assert len(recall) == 256
    beta_square = 0.3
    def fm(p, r):
        return ((1+beta_square)*p*r) / ((beta_square*p+r))
    max_fmeasure = max(list(map(fm, precision, recall)))
    return max_fmeasure
