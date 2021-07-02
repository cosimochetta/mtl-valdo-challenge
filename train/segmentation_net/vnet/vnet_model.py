import torch.nn as nn

from .vnet_parts import *


class VNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.input_conv = InputLayer(n_channels, 16)
        
        self.down1 = DownConv(16, 32, depth=2)
        self.down2 = DownConv(32, 64, depth=3)
        self.down3 = DownConv(64, 128, depth=3)
        self.down4 = DownConv(128, 256, depth=3)

        self.up1 = UpConv(256, 256, depth=3)
        self.up2 = UpConv(256, 128, depth=3)
        self.up3 = UpConv(128, 64, depth=3)
        self.up4 = UpConv(64, 32, depth=2)
        
        self.output_conv = OutputLayer(32, 1)

    def forward(self, x):
        x1 = self.input_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.output_conv(x)
        return out
    
    
