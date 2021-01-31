import numpy as np
import torch
import math

#Refer to detectron2
def get_deltas(pre_boxes, gt_boxes, weights = [1.0, 1.0, 1.0, 1.0]):
    wx, wy, ww, wh = weights
    pre_widths = pre_boxes[:, 2] - pre_boxes[:, 0]
    pre_heights = pre_boxes[:, 3] - pre_boxes[:, 1]
    pre_ctr_x = pre_boxes[:, 0] + 0.5 * pre_widths
    pre_ctr_y = pre_boxes[:, 1] + 0.5 * pre_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    
    dx = wx * (gt_ctr_x - pre_ctr_x) / pre_widths
    dy = wy * (gt_ctr_y - pre_ctr_y) / pre_heights
    dw = ww * torch.log(gt_widths / pre_widths)
    dh = wh * torch.log(gt_heights / pre_heights)

    deltas = torch.stack((dx, dy, dw, dh), dim=1)
    return deltas

def apply_delta_to_proposals(boxes, deltas, weights = [1.0, 1.0, 1.0, 1.0], ):

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    # Prevent sending too large values into torch.exp()
    scale_clamp = math.log(1000.0 / 16)
    dw = torch.clamp(dw, max=scale_clamp)
    dh = torch.clamp(dh, max=scale_clamp)

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    pred_boxes = torch.zeros_like(deltas)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2
    return pred_boxes


#Refer to https://blog.csdn.net/hongxingabc/article/details/78996407
def nms_np(dets, scores, thresh):
 
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    areas = (y2-y1+1) * (x2-x1+1)

    keep = []
    order = scores.argsort()[::-1]
    while order.size >0:
        i = order[0]
        keep.append(i)
 
 
        x11 = np.maximum(x1[i], x1[order[1:]]) 
        y11 = np.maximum(y1[i], y1[order[1:]])
        x22 = np.minimum(x2[i], x2[order[1:]])
        y22 = np.minimum(y2[i], y2[order[1:]])
        
 
        w = np.maximum(0, x22-x11+1)
        h = np.maximum(0, y22-y11+1)
       
        overlaps = w*h
        ious = overlaps / (areas[i]+areas[order[1:]] - overlaps)
 
        idx = np.where(ious<=thresh)[0]
        order = order[idx+1]
    return keep

def nms_torch(bboxes, scores, threshold):
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1)*(y2-y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()
            keep.append(i)

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0) 

        iou = inter / (areas[i]+areas[order[1:]]-inter) 
        idx = (iou <= threshold).nonzero().squeeze()
        if idx.numel() == 0:
            break
        order = order[idx+1] 
    return torch.LongTensor(keep)


