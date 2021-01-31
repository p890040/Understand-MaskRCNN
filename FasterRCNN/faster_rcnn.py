import cv2
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops.boxes import batched_nms, nms

from backbone import TinyBackbone
from rpn.rpn import TinyRPN
from rpn.proposal_utils import apply_delta_to_proposals, get_deltas
from target_process_2nd import get_training_target

from torchvision.ops.roi_align import RoIAlign


class TinyFasterRcnn(nn.Module):
    
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
        
        
    def forward(self, x, GTs=None):
        features = self.backbone(x)
        
        if(self.is_train):
            proposals, scores, rpn_losses = self.rpn(features, GTs)
            gt_boxes, gt_classes = GTs
            # Add gt_boxes to proposals
            proposals = torch.cat([gt_boxes, proposals], axis=0)
            # Sample and select fixed number of gt and proposals pairs.
            gt_boxes, gt_classes, proposals = get_training_target(gt_boxes, gt_classes, proposals)
        else:
            proposals, scores = self.rpn(features)
            
        _proposals = torch.cat([torch.zeros((proposals.shape[0],1)).cuda(), proposals], axis=1)
        feature_map = self.roi_align(features, _proposals)
        box_feauture = feature_map.reshape(feature_map.shape[0], -1)
        box_feauture = self.fastrcnn_fc_layer(box_feauture)
        
        pred_class_scores = self.class_head(box_feauture)
        pred_bbox_deltas = self.box_head(box_feauture)
        
        if(self.is_train):
            losses = self.losses(pred_bbox_deltas, pred_class_scores, proposals, GTs=[gt_boxes, gt_classes.type(torch.cuda.LongTensor)])
            losses = {**rpn_losses, **losses}
            return pred_bbox_deltas, pred_class_scores, losses
        else:
            with torch.no_grad():
                scores=F.softmax(pred_class_scores, 1)
                scores= scores[:,1:]
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
                return boxes, scores, classes


    def losses(self, pred_bbox_deltas, pred_class_scores, proposals, GTs):
        gt_boxes, gt_classes = GTs
        
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
        return losses