import os
import argparse
import sys
import numpy as np

from scipy import stats
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils import (
    create_task3_label,
    bounding_box_slices,
    get_file_list_task2,
    get_file_list_task3,
    get_nifty_list,
    load_nifty_from_file,
)

parser = argparse.ArgumentParser(description='Plot histogram of point position along X, Y,  axis')
parser.add_argument('--dataroot', default="D:/Datasets/valdo_norm", type=str, help='dataset root')
parser.add_argument('--task', default=3 , type=int, choices=[2, 3], help='choose task')
opt = parser.parse_args()

def to_percentile(data_array, initial_range, final_range=100):
    
    out_data = np.zeros([len(data_array), final_range])
    i = 0
    for data, in_range in zip(data_array, initial_range):
        step = in_range / final_range
        
        for j, val in enumerate(np.arange(step, in_range+step, step)):
            out_data[i][j] = (np.logical_and(data <= val, data>val-step)).sum()
        i += 1
    return out_data
        

def load_label_cmb(path):
    _, _, _, cmb_paths = get_file_list_task2(path)
    mask_paths = get_nifty_list(path, "_mask")
    labels = []
    for cmb_path, mask_path in zip(cmb_paths, mask_paths):
        cmb = load_nifty_from_file(cmb_path)
        labels.append(cmb)
    return labels
    
def load_label_lacunes(path):
    _, _, _, r1_paths, r2_paths = get_file_list_task3(path)
    mask_paths = get_nifty_list(path, "_mask")
    labels = []
    for r1_path, r2_path, mask_path in zip(r1_paths, r2_paths, mask_paths):
        mask = load_nifty_from_file(mask_path)
        r1 = load_nifty_from_file(r1_path)       
        r2 = load_nifty_from_file(r2_path)
        label = create_task3_label(r1, r2)
        bb = bounding_box_slices(mask)
        labels.append(label[bb])
        
    return labels
    
        
def plot_histogram(axis, data, num_positions=100, label=None, alpha=0.05, color=None):
    #data[data == None] = 0
    values = data.ravel()
    
    kernel = stats.gaussian_kde(values)
    positions = np.linspace(values.min(), values.max(), num=num_positions)
    histogram = kernel(positions)
    kwargs = dict(linewidth=1, color='black' if color is None else color, alpha=alpha)
    if label is not None:
        kwargs['label'] = label
    axis.plot(positions, histogram, **kwargs)
# %%

labels = []
print("Loading data...", end="")
if opt.task == 3:
    labels = load_label_lacunes(opt.dataroot + "/Task3")
else:
    labels = load_label_cmb(opt.dataroot + "/Task2")
print("Done")
# %%
print("Computing distribution...")
hist_range = 100
X_distribution = np.zeros([hist_range])
Y_distribution = np.zeros([hist_range])
Z_distribution = np.zeros([hist_range])

for i, label in enumerate(labels):
    print(i)
    positions = np.where(label == 1)
    
    point_count = to_percentile(positions, label.shape, hist_range)
    X_distribution += point_count[0]
    Y_distribution += point_count[1]
    Z_distribution += point_count[2]
    
# %%
    
plt.bar(range(labels[0].shape[0]), X_distribution)
plt.bar(range(labels[0].shape[1]), Y_distribution)
plt.bar(range(labels[0].shape[2]), Z_distribution)
