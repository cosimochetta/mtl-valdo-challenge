from typing import List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from monai.inferers import sliding_window_inference

from monai.data.utils import (
    compute_importance_map, 
    get_valid_patch_size,
)    

from monai.utils import fall_back_tuple, PytorchPadMode, BlendMode

def sliding_window_3dimg_2dmodel(data, label, model, roi_size=(96, 96), overlap=0.25, mode=BlendMode.CONSTANT, sigma_scale=0.125):
    '''
    ------
    INPUTS
    ------
    img: Tensor of shape BMHWD, M=input channels
    label: Tensor of shape BM'HWD, M'=output channels, used to infer the output shape
    model: model used for inference, takes as input images of shape [M, roi_size[0], roi_size[1]]
    roi_size: size of the patches
    overlap: percentage of overlab
    mode : how to blend data in case of overlap.
        The default is BlendMode.CONSTANT.
    sigma_scale : variance used in case of BendMode.GAUSSIAN
        The default is 0.125.
    
    ------
    OUTPUTS
    ------
    prediction: Tensor of shape BM'HWD with predicted label
    
    '''
    
    prediction = torch.empty(label.shape)
    with torch.no_grad():
        for b in range(data.shape[0]):
            for i in range(data.shape[-1]):
                data_slice = data[b,:,:,:,i].unsqueeze(0)
                pred_slice = sliding_window_inference(data_slice, 
                                                        roi_size, 
                                                        128, 
                                                        model,
                                                        mode=mode,
                                                        overlap=overlap,
                                                        sigma_scale=sigma_scale
                                                        )
                prediction[b,:,:,:,i] = pred_slice[0]
    return prediction
    
def center_window_inference(inputs,
                            predictor, 
                            centers,    
                            sw_batch_size=128, 
                            roi_size=(48,48), 
                            sw_device='cpu', 
                            device='cpu', 
                            mode: Union[BlendMode, str] = BlendMode.CONSTANT,
                            sigma_scale: Union[Sequence[float], float] = 0.125,
                            padding_mode = PytorchPadMode.CONSTANT,
                            cval = 0.0,
                            *args, **kwargs):
    """

    Parameters
    ----------
    inputs : torch array of shape BNWH
    predictor : callable model that
    centers : list containing centers of patches that will be passed to the predictor
    sw_batch_size : batch size of the inference
    roi_size : size of the patch. The default is (48,48).
    sw_device : device of the inference. The default is 'cpu'.
    device : device of the original data. The default is 'cpu'.
    mode : Union[BlendMode, str], how to blend data in case of overlap.
        The default is BlendMode.CONSTANT.
    sigma_scale : Union[Sequence[float], float], variance used in case of BendMode.GAUSSIAN
        The default is 0.125.
    padding_mode : The default is PytorchPadMode.CONSTANT.
    cval : Value of the padding. The default is 0.0.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    torch array of shape BN'WH where N' is the number of classes outputted by 
    the predictor.

    """
    num_spatial_dims = len(inputs.shape) - 2

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    
    inputs = F.pad(inputs, pad=pad_size, mode=PytorchPadMode(padding_mode).value, value=cval)

    # Store all slices in list
    slices = point_patch_slices(image_size, roi_size, centers, pad_size)
    total_slices = len(slices)
    
    
    #num_win = len(slices)  # number of windows per image
    #total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    importance_map = compute_importance_map(
        get_valid_patch_size(image_size, roi_size), mode=mode, sigma_scale=sigma_scale, device=device
    )

    # Perform predictions
    output_image, count_map = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    _initialized = False
    for slice_g in range(0, total_slices, sw_batch_size):
        print(f"{slice_g} / {total_slices}", end="\r")
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [slices[idx] for idx in slice_range]
        
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
    
def point_patch_slices(
    image_size: Sequence[int],
    patch_size: Sequence[int],
    centers: Sequence[int],
    pad_size: Sequence[int],
) -> List[Tuple[slice, ...]]:
    """
    Enumerate all slices defining ND patches of size `patch_size` from an `image_size` input image 
    centered on points

    Args:
        image_size: dimensions of image to iterate over (H, W, [D])
        patch_size: size of patches to generate slices (H, W, [D])
        centers: center points of the patches, number of points * (N, H, W, [D])
        pad_size: padding to add at the points (H, W, [D])
        
    Returns:
        a list of slice objects defining each patch, [[slice(N), :, slice(H), slice(W), [slice(D)]]]
    """
    num_spatial_dims = len(centers[0])
    patch_size = get_valid_patch_size(image_size, patch_size)
    
    slices = []
    
    for center in centers:
        patch_slice = [slice(center[0], center[0]+1), slice(None)]
        patch_slice.extend((slice(s-pad_size[d], s - pad_size[d] + patch_size[d]) for d, s in enumerate(center[1:])))
        slices.append(patch_slice)
        
    return slices
    