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


class DilatedNetworkClassifier(nn.Module):
    # Input 32*32
    
    def __init__(self):
        super(DilatedNetworkClassifier, self).__init__()
         
        self.resizer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
            )
        
        self.layer1 = ConvBlock(64, 64, 2, 1)
        self.layer2 = ConvBlock(64, 64, 2, 2)
        self.layer3 = ConvBlock(64, 64, 3, 4)
        self.layer4 = ConvBlock(64, 64, 3, 8)

        self.fully_conv1 = FullyConv(256, 64, kernel_size=8, dropout=0.5)
        self.fully_conv2 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.resizer(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.fully_conv1(x)
        x = self.fully_conv2(x)
        return x

    def get_original_space_params(self):
        # Xi =  14 + 4*Xo
        return 14, 4

def test_dilated_network_classifier():
    net = DilatedNetworkClassifier()
    summary(net, (3, 512, 512))

if __name__ == "__main__":
    test_dilated_network_classifier()
    