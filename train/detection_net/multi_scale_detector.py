# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchsummary import summary

class ConvLayer2d(nn.Module):
    
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 batch_norm=True, 
                 stride=1, 
                 padding=0):
        super(ConvLayer2d, self).__init__()
       
  
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
    
    def forward(self, x):
        return self.conv(x)
    
 
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

class MultiScaleDetector(nn.Module):
    # Patch training 51x51
    # output original space: Xi = 4*Xo + 25
    
    def __init__(self, conv_size=[20, 40, 80, 110], fc_size=[110, 110], dropout=0.5):
        super(MultiScaleDetector, self).__init__()
        
        self.conv_size = conv_size
        self.fc_size = fc_size
        self.convolutions = nn.Sequential(
                ConvLayer2d(3, conv_size[0], 7, padding=3, stride=2),   # h=(x+1)/2
                nn.MaxPool2d(2),                                        # h=(x+1)/4
                ConvLayer2d(conv_size[0], conv_size[1], 3, padding=1),             # h=(x+1)/4 - 4
                ConvLayer2d(conv_size[1], conv_size[2], 3, padding=1),             # h=(x+1)/4 - 6
                ConvLayer2d(conv_size[2], conv_size[3], 3, padding=1),             # h=(x+1)/4 - 8
            )
                
        # Fully convolutional layers
        self.fully_conv1 = FullyConv(conv_size[3], fc_size[0], kernel_size=8, stride=1, dropout=dropout)  # o=(h-5)/5+1
        self.fully_conv2 = FullyConv(fc_size[0], fc_size[1], kernel_size=1, dropout=dropout)    # o=(h-5)/5+1
        self.fully_conv3 = nn.Conv2d(fc_size[1], 1, kernel_size=1)                              # o=(h-5)/5+1
        
    def forward(self, x):
        x = self.convolutions(x)
        x = self.fully_conv1(x)
        x = self.fully_conv2(x)
        x = self.fully_conv3(x)
        return x
    
    def get_original_space_params(self):
        # Xi = 25 + 4*Xo
        return 14, 4

def test_multi_scale_detector():
    net = MultiScaleDetector()
    summary(net, (3, 512, 512))

if __name__=='__main__':
    test_multi_scale_detector()