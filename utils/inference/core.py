from typing import List, Tuple, Callable, Any

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from .faceshifter_run import faceshifter_batch
from .image_processing import crop_face, normalize_and_torch
from .video_processing import read_video, crop_frames_and_get_transforms, resize_frames # ,crop_frames_and_get_transforms_multi

# used
def transform_target_to_torch(resized_frs: np.ndarray, half=True) -> torch.tensor:
    """
    Transform target, so it could be used by model
    """
    target_batch_rs = torch.tensor(resized_frs.copy()).cuda()
    target_batch_rs = target_batch_rs[:, :, :, [2,1,0]]/255.
        
    if half:
        target_batch_rs = target_batch_rs.half()
        
    target_batch_rs = (target_batch_rs - 0.5)/0.5 # normalize
    target_batch_rs = target_batch_rs.permute(0, 3, 1, 2)
    
    return target_batch_rs

# used
def model_inference(full_frames: List[np.ndarray],
                    source: np.ndarray,
                    listNetArc: Callable,
                    G: Callable,
                    app: Callable, 
                    crop_size=224,
                    BS=60,
                    half=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Using original frames get faceswaped frames and transofrmations
    """
    # Get the cropped faces from original frames and transformations to get those crops
    crop_frames, tfm_array = crop_frames_and_get_transforms(full_frames, app, crop_size)
    
    # Normalize image and transform to torch
    source = normalize_and_torch(source)
    
    # Get Arcface embeddings of source image
    if len(listNetArc) == 1:
        netArc = listNetArc[0]
        source_embeds = netArc(F.interpolate(source, scale_factor=0.5, mode='bilinear', align_corners=True))
    else:
        netArc, netArc2 = listNetArc[0], listNetArc[1]
        source_embeds1 = netArc(F.interpolate(source, scale_factor=0.5, mode='bilinear', align_corners=True))
        source_embeds2 = netArc2(F.interpolate(source, scale_factor=0.5, mode='bilinear', align_corners=True))
        source_embeds = (source_embeds1+source_embeds2)*0.5
    
    
    # Resize croped frames and get vector which shows on which frames there were faces
    resized_frs, present = resize_frames(crop_frames)
    resized_frs = np.array(resized_frs)
    
    # transform embeds of Xs and target frames to use by model
    try:
        target_batch_rs = transform_target_to_torch(resized_frs, half=half)
    except:
        return None, None, None, None
    
    if half:
        source_embeds = source_embeds.half()
        
    # run model
    size = target_batch_rs.shape[0]
    model_output = []

    for i in tqdm(range(0, size, BS)):
        Y_st = faceshifter_batch(source_embeds, target_batch_rs[i:i+BS], G)
        model_output.append(Y_st)
    torch.cuda.empty_cache()
    model_output = np.concatenate(model_output)
    
    # create list of final frames with transformed faces
    final_frames = []
    idx_fs = 0

    for pres in tqdm(present):
        if pres == 1:
            final_frames.append(model_output[idx_fs])
            idx_fs += 1
        else:
            final_frames.append([])
    
    return final_frames, crop_frames, full_frames, tfm_array
    
    
# def model_inference_multi(full_frames: List[np.ndarray],
#                     source: List,
#                     target: List, 
#                     netArc: Callable,
#                     G: Callable,
#                     app: Callable, 
#                     crop_size=224,
#                     BS=1,
#                     half=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Using original frames get faceswaped frames and transofrmations
#     """
#     # Get Arcface embeddings of target image
#     target_embeds = []
#     for target_curr in target:
#         target_curr = normalize_and_torch(target_curr)
#         target_embeds.append(netArc(F.interpolate(target_curr, scale_factor=0.5, mode='bilinear', align_corners=True)))

#     # Get the cropped faces from original frames and transformations to get those crops
#     crop_frames_list, tfm_array_list = crop_frames_and_get_transforms_multi(full_frames, target_embeds, app, netArc, crop_size)
    
#     # Normalize source images and transform to torch and get Arcface embeddings
#     source_embeds = []
#     for source_curr in source:
#         source_curr = normalize_and_torch(source_curr)
#         source_embeds.append(netArc(F.interpolate(source_curr, scale_factor=0.5, mode='bilinear', align_corners=True)))
    
#     final_frames_list = []
#     for idx, (crop_frames, tfm_array, source_embeds) in enumerate(zip(crop_frames_list, tfm_array_list, source_embeds)):
#         # Resize croped frames and get vector which shows on which frames there were faces
#         resized_frs, present = resize_frames(crop_frames)
#         resized_frs = np.array(resized_frs)

#         # transform embeds of Xs and target frames to use by model
#         target_batch_rs = transform_target_to_torch(resized_frs, half=half)

#         if half:
#             source_embeds = source_embeds.half()

#         # run model
#         size = target_batch_rs.shape[0]
#         model_output = []

#         for i in tqdm(range(0, size, BS)):
#             Y_st = faceshifter_batch(source_embeds, target_batch_rs[i:i+BS], G)
#             model_output.append(Y_st)
#         torch.cuda.empty_cache()
#         model_output = np.concatenate(model_output)

#         # create list of final frames with transformed faces
#         final_frames = []
#         idx_fs = 0

#         for pres in tqdm(present):
#             if pres == 1:
#                 final_frames.append(model_output[idx_fs])
#                 idx_fs += 1
#             else:
#                 final_frames.append([])
#         final_frames_list.append(final_frames)
    
#     return final_frames_list, crop_frames_list, full_frames, tfm_array_list   