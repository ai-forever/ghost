import cv2
import numpy as np
import os
from typing import List, Tuple, Callable, Any
from tqdm import tqdm
# from IPython.display import clear_output
import traceback

from .masks import face_mask, face_mask_static
from .image_processing import normalize_and_torch

import torch
import torch.nn.functional as F
import kornia

def video_is_less_than(path_to_video: str, seconds=10.) -> bool:
    
    cap = cv2.VideoCapture(path_to_video)
    fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    return frames < fps*seconds


def add_audio_from_another_video(video_with_sound: str, 
                                 video_without_sound: str, 
                                 audio_name: str, 
                                 fast_cpu=True,
                                 gpu=False) -> None:
    
    if not os.path.exists('./examples/audio/'):
        os.makedirs('./examples/audio/')
    fast_cmd = "-c:v libx264 -preset ultrafast -crf 18" if fast_cpu else ""
    gpu_cmd = "-c:v h264_nvenc" if gpu else ""
    os.system(f"ffmpeg -i {video_with_sound} -vn -vcodec h264_nvenc ./examples/audio/{audio_name}.m4a")
    os.system(f"ffmpeg -i {video_without_sound} -i ./examples/audio/{audio_name}.m4a {fast_cmd} {gpu_cmd}{video_without_sound[:-4]}_audio.mp4 -y")
    os.system(f"rm -rf ./examples/audio/{audio_name}.m4a")
    os.system(f"mv {video_without_sound[:-4]}_audio.mp4 {video_without_sound}")
    
        
def read_video(path_to_video: str) -> Tuple[List[np.ndarray], float]:
    """
    Read video by frames using its path
    """
    
    # load video 
    cap = cv2.VideoCapture(path_to_video)
    
    width_original, height_original = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #
    fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    full_frames = []
    i = 0 # current frame

    while(cap.isOpened()):
        if i == frames:
            break

        ret, frame = cap.read()

        i += 1
        if ret==True:
            full_frames.append(frame)
            p = i * 100 / frames
            # clear_output(wait=True)
            # print(int(p), '%', 'current frame=', i)
        else:
            break
    
    cap.release()
    
    return full_frames, fps


def crop_frames_and_get_transforms(full_frames: List[np.ndarray], 
                app: Callable, 
                crop_size: int) -> Tuple[List[Any], List[Any]]:
    """
    Crop faces from frames and get respective tranforms
    """

    crop_frames = []
    tfm_array = []

    for frame in tqdm(full_frames):
        try:
            frame_align_crop, tfm = app.get(frame, crop_size)
            crop_frames.append(frame_align_crop[0])
            tfm_array.append(tfm[0])
        except TypeError:
            crop_frames.append([])
            tfm_array.append([])
   
    return crop_frames, tfm_array


def crop_frames_and_get_transforms_multi(full_frames: List[np.ndarray],
                target_embeds: List,
                app: Callable, 
                netArc: Callable,
                crop_size: int,
                similarity_th=0.6) -> Tuple[List[Any], List[Any]]:
    """
    Crop faces from frames and get respective tranforms
    """
    
    crop_frames = [ [] for _ in range(len(target_embeds)) ]
    tfm_array = [ [] for _ in range(len(target_embeds)) ]

    for frame in tqdm(full_frames):
        try:
            faces, tfms = app.get(frame, crop_size)
            face_embeds = []
            for i, face in (enumerate(faces)):
                face_norm = normalize_and_torch(face)
                face_embeds.append(netArc(F.interpolate(face_norm, scale_factor=0.5, mode='bilinear', align_corners=True)))
            similarity = []
            for target_embeds_curr in target_embeds:
                similarity.append([])
                for face_embeds_curr in face_embeds:
                    similarity[-1].append((1 - torch.cosine_similarity(target_embeds_curr, face_embeds_curr, dim=1)).mean().detach().cpu().numpy())
            for idx, similarity_curr in enumerate(similarity):
                best_idx = np.argmin(similarity_curr)
                if similarity_curr[best_idx] < similarity_th:
                    crop_frames[idx].append(faces[best_idx])
                    tfm_array[idx].append(tfms[best_idx])
                else:
                    crop_frames[idx].append([])
                    tfm_array[idx].append([])
                              
        except TypeError:
            for q in range (len(target_embeds)):                
                crop_frames[q].append([])
                tfm_array[q].append([])
   
    return crop_frames, tfm_array


def resize_frames(crop_frames: List[np.ndarray], new_size=(256, 256)) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Resize frames to new size
    """
    
    resized_frs = []
    present = np.ones(len(crop_frames))

    for i, crop_fr in tqdm(enumerate(crop_frames)):
        try:
            resized_frs.append(cv2.resize(crop_fr, new_size))
        except:
            present[i] = 0
            
    return resized_frs, present


def get_final_video_frame(final_frames: List[np.ndarray],
                          crop_frames: List[np.ndarray],
                          full_frames: List[np.ndarray],
                          tfm_array: List[np.ndarray],
                          OUT_VIDEO_NAME: str,
                          fps: float, 
                          handler) -> None:
    """
    Create final video from frames
    """

    out = cv2.VideoWriter(f"{OUT_VIDEO_NAME}", cv2.VideoWriter_fourcc(*'MP4V'), fps, (full_frames[0].shape[1], full_frames[0].shape[0]))

    params = None 
    size = (full_frames[0].shape[0], full_frames[0].shape[1])
    
    for i in tqdm(range(len(final_frames))):
        if i == len(full_frames):
            break  
        try:
            if len(crop_frames[i]) == 0:
                out.write(full_frames[i])
                params = None
                continue
                
            swap = cv2.resize(final_frames[i], (224, 224))
            landmarks = handler.get_without_detection_without_transform(swap)
            if params == None:     
                landmarks_tgt = handler.get_without_detection_without_transform(crop_frames[i])
            
            if params == None:
                mask, params = face_mask_static(swap, landmarks, landmarks_tgt, params)
            else:
                mask = face_mask_static(swap, landmarks, landmarks_tgt, params)  
            
            swap = torch.from_numpy(swap).cuda().permute(2,0,1).unsqueeze(0).type(torch.float32)
            mask = torch.from_numpy(mask).cuda().unsqueeze(0).unsqueeze(0).type(torch.float32)
            frame = torch.from_numpy(full_frames[i]).cuda().permute(2,0,1).unsqueeze(0)
            mat = torch.from_numpy(tfm_array[i]).cuda().unsqueeze(0).type(torch.float32)
            
            mat_rev = kornia.invert_affine_transform(mat)
            swap_t = kornia.warp_affine(swap, mat_rev, size)
            mask_t = kornia.warp_affine(mask, mat_rev, size)
            final = (mask_t*swap_t + (1-mask_t)*frame).type(torch.uint8).squeeze().permute(1,2,0).cpu().detach().numpy()
            out.write(final)
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            print(e)
            print(i)
            out.write(full_frames[i])

    out.release()
    
    
def get_final_video(final_frames: List[np.ndarray],
                    crop_frames: List[np.ndarray],
                    full_frames: List[np.ndarray],
                    tfm_array: List[np.ndarray],
                    OUT_VIDEO_NAME: str,
                    fps: float, 
                    handler) -> None:
    """
    Create final video from frames
    """

    out = cv2.VideoWriter(f"./videos/result/{OUT_VIDEO_NAME}", cv2.VideoWriter_fourcc(*'MP4V'), fps, (full_frames[0].shape[1], full_frames[0].shape[0]))
    landmarks = np.ones((106, 2))

    params = None 
    for i in tqdm(range(len(final_frames))):
        if i == len(full_frames):
            break             
        try:
            try:
                swap = cv2.resize(final_frames[i], (224, 224))
                landmarks_prev = landmarks.copy()
                landmarks = handler.get_without_detection(swap)
                if params == None:
                    landmarks_tgt = handler.get_without_detection(crop_frames[i])
            except:
                landmarks = landmarks_prev
                landmarks_tgt = landmarks_prev
            
            if len(crop_frames[i]) == 0:
                out.write(full_frames[i])
                params = None
                continue
                
            if params == None:
                mask, params = face_mask_static(swap, landmarks, landmarks_tgt, params)
            else:
                mask = face_mask_static(swap, landmarks, landmarks_tgt, params)  
                
            #mask = face_mask(swap, landmarks, landmarks_tgt)
            
            mat_rev = cv2.invertAffineTransform(tfm_array[i])

            # frame = blend(cv2.resize(crop_frames[i], (256, 256)), final_frames[i])
            # frame = cv2.resize(final_frames[i], (224, 224))

            swap_t = cv2.warpAffine(swap, mat_rev, (full_frames[0].shape[1], full_frames[0].shape[0]), borderMode=cv2.BORDER_REPLICATE)
            mask_t = cv2.warpAffine(mask, mat_rev, (full_frames[0].shape[1], full_frames[0].shape[0]))
            mask_t = np.expand_dims(mask_t, 2)
            
            final = mask_t*swap_t + (1-mask_t)*full_frames[i]
            final = np.array(final, dtype='uint8')
            # final[0:100, 0:100] = cv2.resize(source_256, (100, 100))
            out.write(final)
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            print(e)
            print(i)
            out.write(full_frames[i])
    if params == [15, 15, 15]:
        print('Щеки не переносятся')
    else:
        print('Переносим щеки')
    out.release()
    

def get_final_video_multi(final_frames: List[np.ndarray],
                    crop_frames: List[np.ndarray],
                    full_frames: List[np.ndarray],
                    tfm_array: List[np.ndarray],
                    OUT_VIDEO_NAME: str,
                    fps: float, 
                    handler) -> None:
    """
    Create final video from frames
    """

    out = cv2.VideoWriter(f"./videos/result/{OUT_VIDEO_NAME}", cv2.VideoWriter_fourcc(*'MP4V'), fps, (full_frames[0].shape[1], full_frames[0].shape[0]))
    landmarks = np.ones((106, 2))


    for i in tqdm(range(len(full_frames))):
        if i == len(full_frames):
            break
        for j in range(len(crop_frames)):
            try:
                try:
                    landmarks_prev = landmarks.copy()
                    landmarks = handler.get_without_detection(crop_frames[j][i])
                except:
                    landmarks = landmarks_prev

                if len(crop_frames[j][i]) == 0:
                    continue

                mask = face_mask(crop_frames[j][i], landmarks)

                mat_rev = cv2.invertAffineTransform(tfm_array[j][i])

                # frame = blend(cv2.resize(crop_frames[i], (256, 256)), final_frames[i])
                frame = cv2.resize(final_frames[j][i], (224, 224))

                swap_t = cv2.warpAffine(frame, mat_rev, (full_frames[0].shape[1], full_frames[0].shape[0]), borderMode=cv2.BORDER_REPLICATE)
                mask_t = cv2.warpAffine(mask, mat_rev, (full_frames[0].shape[1], full_frames[0].shape[0]))
                mask_t = np.expand_dims(mask_t, 2)

                final = mask_t*swap_t + (1-mask_t)*full_frames[i]
                full_frames[i] = np.array(final, dtype='uint8')

            except Exception as e:
                traceback.print_tb(e.__traceback__)
                print(e)
                print(i)
                
        out.write(full_frames[i])

    out.release()