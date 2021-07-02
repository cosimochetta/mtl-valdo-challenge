""" Full assembly of the parts to form the complete network """

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from .unet_parts import *
from .MR_layer import MR


def get_blocks(model, inst_type):
    if isinstance(model, inst_type):
        blocks = [model]
    else:
        blocks = []
        for child in model.children():
            blocks += get_blocks(child, inst_type)
    return blocks


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, sigma, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_tasks = n_classes + 1
        self.sigma = sigma
        self.active_task = 0
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.n_tasks, self.sigma, bilinear)
        self.up2 = Up(512, 256 // factor, self.n_tasks, self.sigma, bilinear)
        self.up3 = Up(256, 128 // factor, self.n_tasks, self.sigma, bilinear)
        self.up4 = Up(128, 64, self.n_tasks, self.sigma, bilinear)
        self.outcs = nn.ModuleList([OutConv(64, 1), OutConv(64, 1), OutConv(64, 2)])

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        seg_logits = self.outcs[self.active_task](x)
        return seg_logits
    
    def get_convs(self):
        blocks = get_blocks(self, nn.Conv2d)[:-2]
        return blocks

    def get_weights(self):
        return [elt.weight for elt in self.get_convs()]

    def get_weight(self, depth):
        return self.get_weights()[depth]

    def get_BNs(self):
        return get_blocks(self, nn.BatchNorm2d)

    def get_MRs(self):
        return get_blocks(self, MR)

    def get_conv(self, depth):
        return self.get_convs()[depth]

    def get_BN(self, depth):
        return self.get_BNs()[depth]

    def get_MR(self, depth):
        return self.get_MRs()[depth-10]

    def get_routing_masks(self):
        return [elt.get_unit_mapping() for elt in self.get_MRs()]
    
    def get_routing_mask(self, depth):
        return self.get_routing_masks()[depth-10]
    
    def change_task(self, task):
        def aux(m):
            if hasattr(m, 'active_task'):
                m.set_active_task(task)
        self.apply(aux)

    def set_active_task(self, active_task):
        self.active_task = active_task

    def set_routing_mask(self, depth, new_mask, device, reset=False):
        MR_layer = self.get_MR(depth)
        MR_layer.assign_mapping(new_mask, device, reset=reset)

    def update_routing_masks(self, size, ratio, nb_updates, device):
        weights = [elt.detach().cpu().numpy() for elt in self.get_weights()]
        # Select swaps to apply for each layer
        
        for depth in range(10,len(weights)):
            #if nb_updates < ratio*weights[depth].shape[0]
            routing_mask, tested_units = self.get_routing_mask(depth) # Get current mask and history mask
            nb_free = np.sum(1-routing_mask, axis=1)[0]
            # If update ratio reached, pass
            if nb_updates >= round(ratio*nb_free):
                continue
            # If not any possible update left, pass
            if not all([np.sum(tested_units[i,:]) < tested_units.shape[1] for i in range(routing_mask.shape[0])]):
                continue
            update_size = min(np.sum(1-tested_units[0,:]), size)
    
            # Get replacement candidates
            to_activate = np.array([random.sample(np.where(1-tested_units[i,:])[0].tolist(), k=update_size) for i in range(routing_mask.shape[0])])
            # Get candidates to discard
            to_discard = np.array([random.sample(np.where(routing_mask[i,:])[0].tolist(), k=update_size) for i in range(routing_mask.shape[0])])
            
            # Create the new routing mask, and update the model
            new_TR = np.array(routing_mask)
            for task in range(len(to_activate)):
                for i in range(update_size):
                    new_TR[task, to_activate[task][i]] = 1
                    new_TR[task, to_discard[task][i]] = 0
            self.set_routing_mask(depth, new_TR, device) 
