import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from proposal_utils import get_deltas

#Refer to detectron2
def pairwise_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]
     
    width_height.clamp_(min=0)  # [N,M,2]
    inter = width_height.prod(dim=2)  # [N,M]
    del width_height

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def subsample_labels(labels, num_samples, positive_fraction):

    positive = torch.nonzero((labels != -1) & (labels != 0))
    negative = torch.nonzero(labels == 0)

    num_pos = int(num_samples * positive_fraction)
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel())[:num_pos]
    perm2 = torch.randperm(negative.numel())[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    
    # return pos_idx, neg_idx

    labels.fill_(-1)
    labels[pos_idx]=1
    labels[neg_idx]=0
    return labels
    
    

def get_training_target(gt_boxes, anchors):
    iou_matrix = pairwise_iou(gt_boxes, anchors)

    iou_vals, gt_idxs = iou_matrix.max(dim=0)

    gt_fgbg_labels = torch.ones(gt_idxs.size(), dtype=torch.int, device=torch.device('cuda:0'))
    assert (iou_vals<0).sum() == 0 and (iou_vals>1).sum() == 0
    gt_fgbg_labels[(iou_vals < 0.3)] = 0
    gt_fgbg_labels[(iou_vals >=0.3) & (iou_vals < 0.7)] = -1
    gt_fgbg_labels[(iou_vals >=0.7)] = 1
    
    match_gt_boxes = gt_boxes[gt_idxs]
    gt_anchor_deltas = get_deltas(anchors, match_gt_boxes)
    
    return gt_fgbg_labels, gt_anchor_deltas
