# -*- coding: utf-8 -*-

from scipy.ndimage import label
import numpy as np
from glob import glob
import nibabel as nib
import os
import matplotlib.pyplot as plt
data_path = "D:/Datasets/valdo/Task3/Train"

def create_bounding_box(mask):
    labeled_array, num_objects = label(mask)
    boxes = []
    sizes = []
    for i in range(1, num_objects+1):
        pos = np.where(labeled_array == i)
        xmin = np.min(pos[0])
        xmax = np.max(pos[0])
        ymin = np.min(pos[1])
        ymax = np.max(pos[1])
        zmin = np.min(pos[2])
        zmax = np.max(pos[2])
        boxes.append([xmin, ymin, zmin, xmax, ymax, zmax])
        sizes.append({'x_size': xmax-xmin, 
                      'y_size': ymax-ymin, 
                      'z_size': zmax-zmin,
                      'volume': (xmax-xmin)*(ymax-ymin)*(zmax-zmin)})
    return labeled_array, num_objects, boxes, sizes


tot_objects = 0
tot_boxes = []
tot_sizes = []
for path in glob(data_path + "/*/*Rater*.nii.gz"):
    print(f"detection: {os.path.basename(path)} ")
    mask = nib.load(path).get_fdata()
    mask = np.floor(mask+0.5)
    
    if np.max(mask > 0):
        _, num_objects, boxes, sizes = create_bounding_box(mask)
        print(f"Number of objects: {num_objects}")
        print(f"boxes: {boxes}")   
        print(f"sizes: {sizes}")
        tot_objects += num_objects
        tot_boxes += boxes
        tot_sizes += sizes
    else:
        print("Empty mask")
    

# %%

x_sizes = [size['x_size'] for size in tot_sizes]
y_sizes = [size['y_size'] for size in tot_sizes]
z_sizes = [size['z_size'] for size in tot_sizes]
volumes = [size['volume'] for size in tot_sizes]
fig1, ax1 = plt.subplots()
ax1.hist(x_sizes, bins=20, cumulative=False)
fig1, ax1 = plt.subplots()
ax1.hist(y_sizes, bins=20, cumulative=False)
fig1, ax1 = plt.subplots()
ax1.hist(z_sizes, bins=20, cumulative=False)
