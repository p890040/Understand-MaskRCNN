import cv2
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops.boxes import batched_nms, nms

from .anchor import TinyAnchor
from .proposal_utils import get_deltas, apply_delta_to_proposals, nms_np, nms_torch
from .target_process import get_training_target, subsample_labels


class TinyRPN(nn.Module):
    
    def __init__(self, anchors, is_train):
        '''
        It is tiny rpn. Only for feature_map shape [1024,50,80]
        '''
        super().__init__()
        feature_map_channel = 1024
        self.conv = nn.Conv2d(feature_map_channel, feature_map_channel, kernel_size=3, stride=1, padding=1)
        self.object_scores = nn.Conv2d(feature_map_channel, 5*3, kernel_size=1, stride=1) # 5 anchor size, 3 anchor ratio.
        self.anchor_deltas = nn.Conv2d(feature_map_channel, 5*3*4, kernel_size=1, stride=1) # Size 4 means x, y, x_delta, y_delta
        self.anchors = anchors
        self.is_train = is_train
        
    def forward(self, feature_map, GTs=None):
        x = F.relu(self.conv(feature_map))
        pred_object_scores = self.object_scores(x)
        pred_anchor_deltas = self.anchor_deltas(x)#.reshape(1, 15, 50, 80, 4)
        
        if(self.is_train):
            losses = self.losses(pred_anchor_deltas, pred_object_scores, self.anchors, GTs)
            # return 0, 0, losses
        with torch.no_grad():
            scores = self.predict_scores(pred_object_scores)
            proposals = self.predict_proposals(self.anchors, pred_anchor_deltas)
            proposals, scores = self.find_top_rpn_proposals(proposals, scores)
            
        if(self.is_train):
            return proposals, scores, losses
        return proposals, scores 
    
    def predict_proposals(self, anchors, pred_anchor_deltas):
        pred_anchor_deltas = pred_anchor_deltas.reshape(1, 15, 4, 50, 80).permute(0, 3, 4, 1, 2).reshape(-1, 4)
        proposals = apply_delta_to_proposals(anchors, pred_anchor_deltas)
        return proposals

    def predict_scores(self, pred_object_scores):
        scores = pred_object_scores.permute(0, 2, 3, 1).reshape(-1)
        return scores
    
    def find_top_rpn_proposals(self, proposals, scores, image_shape=(800,1280), nms_trh=0.7, pre_nms_topk=12000, post_nms_topk=2000):
        scores, idx = scores.sort(descending=True)
        top_scores = scores[:pre_nms_topk]
        top_proposals = proposals[idx[:pre_nms_topk]]

        valid_mask = torch.isfinite(top_proposals).all(dim=1) & torch.isfinite(top_scores)
        if not valid_mask.all():
            top_proposals = top_proposals[valid_mask]
            top_scores = top_scores[valid_mask]

        # box clip
        h, w = image_shape
        top_proposals[:,0].clamp_(min=0, max=w)
        top_proposals[:,1].clamp_(min=0, max=h)
        top_proposals[:,2].clamp_(min=0, max=w)
        top_proposals[:,3].clamp_(min=0, max=h)
        
        # Remove h or w < 0
        widths = top_proposals[:, 2] - top_proposals[:, 0]
        heights = top_proposals[:, 3] - top_proposals[:, 1]
        keep = (widths > 0) & (heights > 0)
        if keep.sum().item() != len(top_proposals):
            top_proposals, top_scores = top_proposals[keep], top_scores[keep]
        
        # NMS post processing
# =============================================================================
#         keep = nms(top_proposals, top_scores, nms_trh).numpy()
#         keep2 = nms_np(top_proposals.numpy(), top_scores.numpy(), nms_trh)
#         keep3 = nms_torch(top_proposals, top_scores, nms_trh)
# =============================================================================
        # keep = nms_torch(top_proposals, top_scores, nms_trh)
        keep = nms(top_proposals, top_scores, nms_trh)
        keep = keep[:post_nms_topk]
        top_proposals, top_scores = top_proposals[keep], top_scores[keep]

        return top_proposals, top_scores

    def losses(self, pred_anchor_deltas, pred_object_scores, anchors, GTs):

        gt_boxes, gt_labels = GTs
        num_samples= 256
        gt_fgbg_labels, gt_anchor_deltas = get_training_target(gt_boxes, anchors)
        gt_fgbg_labels = subsample_labels(gt_fgbg_labels, num_samples, 0.5)
        
        # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N*Hi*Wi*A, )
        pred_object_scores = pred_object_scores.permute(0, 2, 3, 1).flatten()
        # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B)
        #          -> (N*Hi*Wi*A, B)
        B=4
        x = pred_anchor_deltas
        pred_anchor_deltas = pred_anchor_deltas.view(x.shape[0], -1, B, x.shape[-2], x.shape[-1]).permute(0, 3, 4, 1, 2).reshape(-1, B)
        
        # Losses
        # box loss
        # smooth_l1_beta=0.0
        pos_masks = gt_fgbg_labels == 1
        loss_box  = F.smooth_l1_loss(
            pred_anchor_deltas[pos_masks], gt_anchor_deltas[pos_masks], reduction="sum"
        )
        # fg bg loss
        valid_masks = gt_fgbg_labels >= 0
        loss_cls = F.binary_cross_entropy_with_logits(
            pred_object_scores[valid_masks],
            gt_fgbg_labels[valid_masks].to(torch.float32),
            reduction="sum",
        )
        
        normalizer = 1.0 / num_samples
        loss_cls = loss_cls * normalizer  # cls: classification loss
        loss_box = loss_box * normalizer  # loc: localization loss
        losses = {'loss_cls':loss_cls, 'loss_box':loss_box}
        return losses
        
        


if __name__ == "__main__":
    """"For test"""
    image = torch.randn((1, 3, 800, 1280), device= torch.device('cuda'))
    backbone = TinyBackbone().cuda()
    feature_map = backbone(image)
    
    rpn_model = TinyRPN(TinyAnchor().anchor, False).cuda()
    rpn_output = rpn_model(feature_map)
    