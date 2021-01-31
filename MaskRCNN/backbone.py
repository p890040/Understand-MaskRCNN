import cv2
import numpy as np
import os
import torch
from torch import nn

class TinyBackbone(nn.Module):
    
    def __init__(self):
        '''
        It is tiny backbone. Only for image shape [800,1280,3]. H,W,C
        Expect output shape [1024,50,80] C,H,W
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """
        Args:
            Input is a image.
        Output:
            Feature map.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


if __name__ == "__main__":
    """"For test"""
    image = torch.randn((1, 3, 800, 1280))
    backbone = TinyBackbone()
    feature_map = backbone(image)
    
