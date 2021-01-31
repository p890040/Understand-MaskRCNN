import numpy as np
import os
import torch


class TinyAnchor():
    
    def __init__(self, default_anchor_path='1280_800_default_anchors.npy'):
        self.anchor = torch.tensor(np.load(default_anchor_path), requires_grad=False, device=torch.device('cuda:0'))

if __name__ == "__main__":
    """"For test"""
    anchors = TinyAnchor('1280_800_default_anchors.npy').anchor