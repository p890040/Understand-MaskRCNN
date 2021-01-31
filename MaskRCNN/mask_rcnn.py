import cv2
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops.boxes import batched_nms, nms
from torchvision.ops.roi_align import RoIAlign

from backbone import TinyBackbone
from rpn.rpn import TinyRPN
from rpn.proposal_utils import apply_delta_to_proposals, get_deltas
from target_process_2nd import get_training_target


class TinyMaskRcnn(nn.Module):
    
    def __init__(self, anchors, is_train, num_class):
        '''
        It is tiny FasterRcnn. Only for feature_map shape [1024,50,80]
        '''
        super().__init__()
        self.is_train = is_train
        self.num_class = num_class
        
        self.backbone = TinyBackbone()
        
        self.rpn = TinyRPN(anchors, self.is_train)
        
        self.roi_align = RoIAlign(output_size=(14,14), spatial_scale=0.0625, sampling_ratio=0)
        
        self.fastrcnn_fc_layer = nn.Linear(1024*14*14, 1024)
        self.class_head  = nn.Linear(1024, num_class+1)# Add bg
        self.box_head  = nn.Linear(1024, num_class*4)
        self.maks_head = nn.Sequential(nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2, padding=0),
                                       nn.Conv2d(256, num_class, kernel_size=1, stride=1))
        
        
    def forward(self, x, GTs=None):
        features = self.backbone(x)
        
        if(self.is_train):
            gt_boxes, gt_classes, gt_masks = GTs
            proposals, scores, rpn_losses = self.rpn(features, GTs=[gt_boxes, gt_classes]) 
            # Add gt_boxes to proposals
            proposals = torch.cat([gt_boxes, proposals], axis=0)
            gt_boxes, gt_classes, gt_masks, proposals = get_training_target(gt_boxes, gt_classes, gt_masks, proposals)
        else:
            proposals, scores = self.rpn(features)
            
        _proposals = torch.cat([torch.zeros((proposals.shape[0],1)).cuda(), proposals], axis=1)
        feature_map = self.roi_align(features, _proposals)
        box_feauture = feature_map.reshape(feature_map.shape[0], -1)
        box_feauture = self.fastrcnn_fc_layer(box_feauture)
        
        pred_class_scores = self.class_head(box_feauture)
        pred_bbox_deltas = self.box_head(box_feauture)
        
        if(self.is_train):
            losses, fg_inds = self.losses(pred_bbox_deltas, pred_class_scores, proposals, GTs=[gt_boxes, gt_classes.type(torch.cuda.LongTensor), gt_masks])
            
            proposals= proposals[fg_inds]
            mask_feature = feature_map[fg_inds]
            gt_classes = gt_classes[fg_inds]
            gt_masks = gt_masks[fg_inds]
            pred_masks = self.maks_head(mask_feature)
            loss_mask = self.loss_mask(pred_masks, gt_classes, gt_masks)
            
            losses = {**rpn_losses, **losses, **loss_mask}

            return pred_bbox_deltas, pred_class_scores, losses
        else:
            with torch.no_grad():
                scores=F.softmax(pred_class_scores, 1)
                scores= scores[:,1:]
                
                classes = torch.arange(self.num_class)
                classes = classes.repeat(scores.shape[0],1)
                # Filter by threshold.
                _filter = scores > 0.5
                scores = scores[_filter]
                
                classes = torch.where(_filter)[1]
                
                num_pred = len(proposals)
                B = proposals.shape[1]
                boxes =  apply_delta_to_proposals(proposals.unsqueeze(1).expand(num_pred, self.num_class, B).reshape(-1, B) ,
                                                  pred_bbox_deltas.reshape(num_pred * self.num_class, B), 
                                                  weights = [10.0, 10.0, 5.0, 5.0])
                # box clip
                h, w = (800,1280)
                boxes[:,0].clamp_(min=0, max=w)
                boxes[:,1].clamp_(min=0, max=h)
                boxes[:,2].clamp_(min=0, max=w)
                boxes[:,3].clamp_(min=0, max=h)
                boxes = boxes.reshape(-1, self.num_class, 4)  # R x C x 4
                boxes = boxes[_filter]
 
                # NMS per class
                keep = batched_nms(boxes, scores, classes, 0.7)
                keep = keep[:200] #Max detection
                boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
                
                if(len(boxes)==0):
                    return boxes, scores,  torch.zeros((0,800,1280))
                
                # Mask ouput
                _boxes = torch.cat([torch.zeros((boxes.shape[0],1)).cuda(), boxes], axis=1)
                mask_feature = self.roi_align(features, _boxes)
                pred_masks = self.maks_head(mask_feature)
                num_masks = pred_masks.shape[0]
                indices = torch.arange(num_masks, device=torch.device('cuda'))
                masks = pred_masks[indices, classes].sigmoid()
                
                masks_full = self.mask_postprocess(boxes, masks)
                
                return boxes, scores, classes, masks_full


    def losses(self, pred_bbox_deltas, pred_class_scores, proposals, GTs):
        gt_boxes, gt_classes, gt_masks = GTs
        
        # Class loss
        loss_cls = F.cross_entropy(pred_class_scores, gt_classes, reduction="mean")
        
        # Box loss
        gt_boxes_deltas = get_deltas(proposals, gt_boxes, weights = [10.0, 10.0, 5.0, 5.0])
        fg_inds = torch.where(gt_classes>0)[0]
        fg_gt_classes = gt_classes[fg_inds]-1 #Chage start from 1 to start from 0. No bg.
        gt_class_cols = 4 * fg_gt_classes[:, None] + torch.arange(4, device=torch.device('cuda'))
        loss_box  = F.smooth_l1_loss(
            pred_bbox_deltas[fg_inds[:, None], gt_class_cols], gt_boxes_deltas[fg_inds], reduction="sum"
        )
        
        loss_box = loss_box / gt_classes.numel()
        print()

        losses = {'loss_cls_2nd':loss_cls, 'loss_box_2nd':loss_box}
        return losses, fg_inds
    
    def loss_mask(self, pred_masks, gt_classes, gt_masks):
        # Select most likely that class one.
        indices = torch.arange(pred_masks.shape[0])
        pred_masks = pred_masks[indices, (gt_classes-1).type(torch.cuda.LongTensor)]
        loss_mask = {'loss_mask':F.binary_cross_entropy_with_logits(pred_masks, gt_masks.type(torch.cuda.FloatTensor), reduction="mean")}
        return loss_mask
        
    
    def mask_postprocess(self, boxes, masks, image_shape=(800,1280)):
        img_h, img_w = image_shape
        boxes_h = (boxes[:,3] - boxes[:,1]).type(torch.cuda.LongTensor)
        boxes_w = (boxes[:,2] - boxes[:,0]).type(torch.cuda.LongTensor)
        boxes = boxes.type(torch.cuda.LongTensor)
        zero_masks = torch.zeros((boxes.shape[0], img_h, img_w), dtype=bool)
        for i in range(boxes.shape[0]):
            mask = F.interpolate(masks[i][None,None,...], size=(boxes_h[i], boxes_w[i]))[0][0]
            mask = mask>0.5
            zero_masks[i, boxes[i][1]:boxes[i][1]+boxes_h[i], boxes[i][0]:boxes[i][0]+boxes_w[i]] = mask
        return zero_masks