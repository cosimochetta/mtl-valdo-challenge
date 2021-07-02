# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 06:38:11 2021

@author: minoc
"""

from .unet import UNet
from .pnet import PNet

def get_segmentation_net(name='resnet'):
    if name not in ['unet', 'pnet']:
        raise Exception("Model does not exist")
    
    if name == 'unet':
        return UNet(3, 2, bilinear=False)
    elif name == 'pnet':
        return PNet()
    
