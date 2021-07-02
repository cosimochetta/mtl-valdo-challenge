# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 18:31:53 2021

@author: minoc
"""

import torch

import torch.nn.functional as F
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import BlendMode, PytorchPadMode, fall_back_tuple
'''
def shift_and_stitch_inference(
    inputs: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor],
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Note:
        - input must be channel-first and have a batch dim, supports N-D sliding window.
    """
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]
    roi_size = image_size_

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device
     
        
    reduction_factor = _get_reduction_factor(inputs[0], predictor)
    
    
    
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=PytorchPadMode(padding_mode).value, value=cval)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    #with torch.no_grad:
        

    # Perform predictions
    output_image, count_map = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    _initialized = False
    for slice_g in range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        seg_prob = predictor(window_data, *args, **kwargs).to(device)  # batched patch segmentation

        if not _initialized:  # init. buffer at the first iteration
            output_classes = seg_prob.shape[1]
            output_shape = [batch_size, output_classes] + list(image_size)
            # allocate memory to store the full output and the count for overlapping parts
            output_image = torch.zeros(output_shape, dtype=torch.float32, device=device)
            count_map = torch.zeros(output_shape, dtype=torch.float32, device=device)
            _initialized = True

        # store the result in the proper location of the full output. Apply weights from importance map.
        for idx, original_idx in zip(slice_range, unravel_slice):
            output_image[original_idx] += importance_map * seg_prob[idx - slice_g]
            count_map[original_idx] += importance_map

    # account for any overlapping sections
    output_image = output_image / count_map

    final_slicing: List[slice] = []
    for sp in range(num_spatial_dims):
        slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
        final_slicing.insert(0, slice_dim)
    while len(final_slicing) < len(output_image.shape):
        final_slicing.insert(0, slice(None))
    return output_image[final_slicing]



def _get_reduction_factor(inputs, predictor):
    with torch.no_grad():
        outputs = predictor(inputs)
        output_shape = outputs.shape[-1]
        
    return inputs.shape[-1] / output_shape 
    

def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)




'''




############################################
############################################
############################################
############################################
############################################
############################################


def generate_shifted_patches(data: torch.Tensor, shift_size: int=9):
    """
    Parameters
    ----------
    data : Torch tensor
        Torch tensor of shape N[387,387], with N being the number of channels
    shift_size : int
        Number of shift S in x and y direction
    
    Returns
    -------
    patches: Torch tensor
        Torch tensor of shape SN[396,396] containing the shifted patches of the 
        input image
    """
    
    #assert data.shape[-2:] = (351, 351)
    
    patches = torch.zeros([shift_size**2, data.shape[0], data.shape[1]+shift_size, data.shape[2]+shift_size])    
    xlen, ylen = data.shape[-2:]
    
    
    xb = (shift_size+1) // 2    # initial padding to center the image on x axis
    yb = (shift_size+1) // 2    # initial padding to center the image on y axis
    
    i = 0
    for x in range(shift_size):
        for y in range(shift_size):
            patches[i,:,shift_size-x:shift_size-x+xlen,shift_size-y:shift_size-y+ylen] = data
            #patches[i] = F.pad(input=data, pad=(, 1, 1, 1), mode='constant', value=0)
            i += 1
    print(patches)
    return patches
    
    
def stich_patches(patches, original_shape, shift_size):
    
    output = torch.zeros([patches.shape[1], original_shape[0], original_shape[1]])    
    s = shift_size // 2 # starting offset
    i = 0
    for x in range(shift_size):
        for y in range(shift_size):
            sizes = output[:,x::shift_size,y::shift_size].shape[-2:]
                
            if x != 0 and y != 0:
                patch = patches[i,:,:-x,:-y]
            elif y != 0:
                patch = patches[i,:,:,:-y]
            elif x != 0:
                patch = patches[i,:,:-x,:]
            else:
                patch = patches[i,:,:]
                
            if patch.shape[-2:] != sizes:
                patch = patch[:,:sizes[0], :sizes[1]]
            output[:,x::shift_size,y::shift_size] = patch
            i += 1
    return output
    
    
    

def shift_and_snitch(data, model, shift_size=9, batch_size=128):
    
    # method explanation: https://www.programmersought.com/article/84864091894/
    
    patched_output = torch.zeros([data.shape[0],1,*data.shape[-2:]])
    #padder = SpatialPad(model.shape)
    
    patches = generate_shifted_patches(data, shift_size) 
    
    patches_output = None
    
    with torch.no_grad():
        iterations = len(patches) // batch_size+1
        for i in range(iterations):
            pred = model(patches[i*batch_size:(i+1)*batch_size])
            patches_output = torch.vstack([patches_output, pred]) if patches_output != None else pred
            
    print(patches_output.shape)
    output = stich_patches(patches_output, data.shape[-2:], shift_size)
    return output
        
   

def TESTMAXPOOL_shift_and_snitch(n=2):
    
    model = torch.nn.MaxPool2d(n)
    
    x = torch.tensor(list(range(1,n**2+1))).reshape([1,n,n])
    print(x)
    output = shift_and_snitch(x, model, n, 256)
    print(output)
    return output         
            
def TEST_shift_and_snitch(model, red, s):
    
    x2 = torch.tensor(list(range(1,s**2+1))).reshape([1,s,s])
    xx = torch.vstack([x2,x2,x2])
    print(xx[0])
    output = shift_and_snitch(xx, model, red,256)
    print(output[0])
    return output