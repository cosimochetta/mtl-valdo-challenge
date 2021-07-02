# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 21:02:06 2021

@author: minoc
"""

import torch
import torch.nn as nn
from torchsummary import summary

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        
    def forward(self, x):
        return self.conv(x)
        
class ConvBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 nconv, 
                 dilation,):
        super(ConvBlock, self).__init__()
       
        self.convs = nn.Sequential(ConvLayer(in_channels, out_channels, kernel_size=(3,3), dilation=dilation, padding=dilation))
        for _ in range(nconv-1):
            self.convs.add_module("conv2d", ConvLayer(out_channels, out_channels, kernel_size=(3,3), dilation=dilation, padding=dilation))
      
    def forward(self, x):
        return self.convs(x)
    
 
class FullyConv(nn.Module):
    
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dropout=0,
                 ):
        super(FullyConv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            )
        
    def forward(self, x):
        return self.conv(x)


class PNet(nn.Module):
    # Input 32*32
    
    def __init__(self):
        super(PNet, self).__init__()
         
        
        self.layer1 = ConvBlock(3, 64, 2, 1)
        self.layer2 = ConvBlock(64, 64, 2, 2)
        self.layer3 = ConvBlock(64, 64, 3, 4)
        self.layer4 = ConvBlock(64, 64, 3, 8)
        self.layer5 = ConvBlock(64, 64, 3, 16)
        
        self.dropout = nn.Dropout(0.5)
        self.fully_conv1 = FullyConv(320, 128, kernel_size=1, dropout=0.5)
        self.fully_conv2 = nn.Conv2d(128, 2, kernel_size=1)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.dropout(x)
        x = self.fully_conv1(x)
        x = self.fully_conv2(x)
        return x


def test_pnet():
    net = PNet()
    summary(net, (3, 32, 32))

if __name__ == "__main__":
    test_pnet()
    