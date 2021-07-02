import numpy as np

from scipy.ndimage import label
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR, MultiplicativeLR
from monai.optimizers import Novograd
from glob import glob
  
def rescale_data(data, mask, min_value=None, max_value=None):
    """
    Rescale data in range [min_value, max_value]
    Ignore data values where mask == 0.

    Parameters
    ----------
    data : array
    mask : array of shape equal to data
    min_value : lowest value for rescale, if None compute it from the data
    max_value : highest value for rescale, if None compute it from the data
    
    Returns
    -------
    data : rescaled array

    """
    
    # Invert mask for masked array
    mask = mask == 0
    ma = np.ma.array(data, mask=mask)
    
    if not min_value:
        min_value = ma.min()
    if not max_value:
        max_value = ma.max()
    
    ma = (ma - min_value) / (max_value - min_value)
    data = ma.filled(0)
    return data

def create_bounding_box(mask):
    labeled_array, num_objects = label(mask)
    boxes = []
    centers = []
    for i in range(1, num_objects+1):
        pos = np.where(labeled_array == i)
        xmin = np.min(pos[-1])
        xmax = np.max(pos[-1])
        ymin = np.min(pos[-2])
        ymax = np.max(pos[-2])
        zmin = np.min(pos[-1])
        zmax = np.max(pos[-1])
        boxes.append([xmin, ymin, zmin, xmax, ymax, zmax])
        centers.append([(xmax+xmin) // 2, (ymax+ymin) // 2, (zmax+zmin) // 2])
    return labeled_array, num_objects, boxes, centers

####################
# TRAINING GETTERS #
####################
  
def get_optimizer(name, model, lr, kwargs):
    optimizer_dict = {
        'Adam': Adam,
        'Novograd': Novograd,
        'SGD': SGD
    }
    
    assert name in optimizer_dict.keys()
    return optimizer_dict[name](model, lr=lr, **kwargs)

def get_scheduler(name, optimizer, kwargs):
    class no_scheduler:
        def step(self):
            return
    scheduler = no_scheduler()
    if name == 'LambdaLR':
        lr_lambda = lambda epoch: epoch ** kwargs['lr_init']
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    if name == 'MultiplicativeLR':
        lr_lambda = lambda epoch: epoch ** kwargs['lr_init']
        scheduler = MultiplicativeLR(optimizer, lr_lambda=lr_lambda)
        
    return scheduler
   
def get_file_list_task3_patches(root_dir):
    file_list = glob(root_dir + "/**/*", recursive=True)
    file_list = [file for file in file_list if '.npy' in file]

    fl_T1 = sorted([file for file in file_list if '_t1' in file])
    fl_T2 = sorted([file for file in file_list if '_t2' in file])
    fl_FLAIR = sorted([file for file in file_list if '_flair' in file])
    fl_R1 = sorted([file for file in file_list if '_r1' in file])
    fl_R2 = sorted([file for file in file_list if '_r2' in file])
    fl_LABEL = sorted([file for file in file_list if '_label' in file])
    assert len(fl_T1) == len(fl_T2) == len(fl_FLAIR) == len(fl_R1) == len(fl_R2) == len(fl_LABEL)
    return fl_T1, fl_T2, fl_FLAIR, fl_R1, fl_R2, fl_LABEL

