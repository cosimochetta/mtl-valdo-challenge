# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 20:21:04 2021

@author: minoc
"""

from .dilated_network_classifier import DilatedNetworkClassifier
from .multi_scale_detector import MultiScaleDetector
from .resnet import resnet10clf

def get_detector(name='resnet'):
    if name not in ['resnet', 'multi_scale', 'dilated_network']:
        raise Exception("Model does not exist")
    
    if name == 'resnet':
        return resnet10clf(3, 1)
    elif name == 'multi_scale':
        return MultiScaleDetector()
    elif name == 'dilated_network':
        return DilatedNetworkClassifier()
    