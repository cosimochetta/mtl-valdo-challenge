""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet3d_parts import *


class UNet3d(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32, 64)
        self.down1 = Down(64, 64, 128)
        self.down2 = Down(128, 128, 256)
        self.down3 = Down(256, 256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        seg_logits = self.outc(x)
        return classif_logits, seg_logits
    
    
