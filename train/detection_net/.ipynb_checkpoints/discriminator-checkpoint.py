# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


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
    
    def set_weight(self, weight):
        self.conv[0].weight.copy_(weight)
 
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

class FullyConnected(nn.Module):
    
    def __init__(self, in_channels, out_channels, dropout=0):
        super(FullyConnected, self).__init__()

        self.fully_connected = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True),
            )
          
    def forward(self, x):
        return self.fully_connected(x)

    def weight(self):
        return self.fully_connected[0].weight

class CandidateDetector(nn.Module):
    # Input 387, output 43, reduction factor 9
    
    
    def __init__(self, conv_size=[20, 40, 80, 110], fc_size=[300, 300, 200], dropout=0.5):
        super(CandidateDetector, self).__init__()
        
        self.conv_size = conv_size
        self.fc_size = fc_size
        self.convolutions = nn.Sequential(
                ConvLayer2d(3, conv_size[0], 7, padding=3, stride=2),   # h=(x+1)/2
                nn.MaxPool2d(2),                                        # h=(x+1)/4
                ConvLayer2d(conv_size[0], conv_size[1], 5),             # h=(x+1)/4 - 4
                ConvLayer2d(conv_size[1], conv_size[2], 3),             # h=(x+1)/4 - 6
                ConvLayer2d(conv_size[2], conv_size[3], 3),             # h=(x+1)/4 - 8
            )
                
        # Fully convolutional layers
        self.fully_conv1 = FullyConv(conv_size[3], fc_size[0], kernel_size=5, stride=1, dropout=dropout)  # o=(h-5)/5+1
        self.fully_conv2 = FullyConv(fc_size[0], fc_size[1], kernel_size=1, dropout=dropout)    # o=(h-5)/5+1
        self.fully_conv3 = FullyConv(fc_size[1], fc_size[2], kernel_size=1, dropout=dropout)    # o=(h-5)/5+1
        self.fully_conv4 = nn.Conv2d(fc_size[2], 1, kernel_size=1)                              # o=(h-5)/5+1
        
    def forward(self, x):
        x = self.convolutions(x)
                
        #x = x.view(x.size(0), 25*self.conv_size[3], -1, -1)
        #print(x.shape)
        #x = self.fc1(x)
        #x = self.fc2(x)
        #x = self.fc3(x)
        #print(x.shape)
        x = self.fully_conv1(x)
        x = self.fully_conv2(x)
        x = self.fully_conv3(x)
        x = self.fully_conv4(x)
        #print(x.shape)
        return x

def Test_CandidateDetector():
    net = CandidateDetector()
    x1 = torch.ones([1,3,51,51])
    o1 = net(x1)
    print(f"out shape {o1.shape}")
    
    x2 = torch.ones([1,3,512,512])
    o2 = net(x2)
    print(f"out shape {o2.shape}")


    for x in range(51,512):
        
        h = ((x-1)/2+1)/2 - 8
        o = (h-5)/1+1
        print(f"{x} / {o}: \t{x/o}")
    
    
    
if __name__=='__main__':
    Test_CandidateDetector()

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        
        self.fc1_input_size = self.get_fc1_input_size(input_size)
        self.c1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 2), stride=1, padding=0)
        self.c2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 2), stride=1, padding=0)
        self.c3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 1), stride=1, padding=0)
        self.c4 = nn.Conv3d(128, 128, kernel_size=(3, 3, 1), stride=1, padding=0)
        self.c5 = nn.Conv3d(256, 256, kernel_size=(3, 3, 1), stride=1, padding=0)
        self.c6 = nn.Conv3d(256, 256, kernel_size=(3, 3, 1), stride=1, padding=0)

        self.fc1 = nn.Linear(self.fc1_input_size * 64, 500)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 1)
        
        self.max_pool = nn.MaxPool3d((2,2,1))
        self.relu = nn.ReLU(inplace=True)
        #self.drop3d = nn.Dropout3d(0.3)
        self.drop = nn.Dropout(0.5)

    def get_fc1_input_size(size):
        x, y, z = size
        x = ((x-6) / 2 - 8)
        y = ((y-6) / 2 - 8)
        z = ((x-4) / 2 - 2)
        return x*y*z
        
    def forward(self, x):
        x = self.relu(self.c1(x))
        x = self.relu(self.c2(x))
        x = self.max_pool(x)
        x = self.relu(self.c3(x))
        x = self.relu(self.c4(x))
        x = self.relu(self.c5(x))
        x = self.relu(self.c6(x))
        
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.drop(self.fc1(x)))
        x = self.relu(self.drop(self.fc2(x)))
        x = self.relu(self.drop(self.fc3(x)))
        return x
