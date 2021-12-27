import cv2
import numpy as np
import os
from PIL import Image
from typing import List, Tuple, Callable, Any
from tqdm import tqdm
import traceback

from .masks import face_mask_static
from .image_processing import normalize_and_torch, normalize_and_torch_batch, crop_face

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import kornia


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
        else:
            break
    
    cap.release()
    
    return full_frames, fps


def get_target(full_frames: List[np.ndarray], 
               app: Callable, 
               crop_size: int):
    i = 0
    target = None
    while target is None:
        if i < len(full_frames):
            try:
                target = [crop_face(full_frames[i], app, crop_size)[0]]
            except TypeError:
                i += 1
        else:
            print("Video doesn't contain face!")
            break
    return target


def crop_frames_and_get_transforms(full_frames: List[np.ndarray],
                target_embeds: List,
                app: Callable, 
                netArc: Callable,
                crop_size: int,
                set_target: bool,
                similarity_th: float) -> Tuple[List[Any], List[Any]]:
    """
    Crop faces from frames and get respective tranforms
    """
    
    crop_frames = [ [] for _ in range(target_embeds.shape[0]) ]
    tfm_array = [ [] for _ in range(target_embeds.shape[0]) ]
    
    target_embeds = F.normalize(target_embeds)
    for frame in tqdm(full_frames):
        try:
            faces, tfms = app.get(frame, crop_size)
            if len(faces) > 1 or set_target:
                face_norm = normalize_and_torch_batch(np.array(faces))
                face_norm = F.interpolate(face_norm, scale_factor=0.5, mode='bilinear', align_corners=True)
                face_embeds = netArc(face_norm)
                face_embeds = F.normalize(face_embeds)

                similarity = face_embeds@target_embeds.T
                best_idxs = similarity.argmax(0).detach().cpu().numpy()
                for idx, best_idx in enumerate(best_idxs):
                    if similarity[best_idx][idx] > similarity_th:
                        crop_frames[idx].append(faces[best_idx])
                        tfm_array[idx].append(tfms[best_idx])
                    else:
                        crop_frames[idx].append([])
                        tfm_array[idx].append([])
            else:
                crop_frames[0].append(faces[0])
                tfm_array[0].append(tfms[0])
        except TypeError:
            for q in range (len(target_embeds)):                
                crop_frames[q].append([])
                tfm_array[q].append([])
    torch.cuda.empty_cache()
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

    out = cv2.VideoWriter(f"{OUT_VIDEO_NAME}", cv2.VideoWriter_fourcc(*'MP4V'), fps, (full_frames[0].shape[1], full_frames[0].shape[0]))
    size = (full_frames[0].shape[0], full_frames[0].shape[1])
    params = [None for i in range(len(crop_frames))]
    result_frames = full_frames.copy()
    
    for i in tqdm(range(len(full_frames))):
        if i == len(full_frames):
            break
        for j in range(len(crop_frames)):
            try:
                swap = cv2.resize(final_frames[j][i], (224, 224))
                
                if len(crop_frames[j][i]) == 0:
                    params[j] = None
                    continue
                    
                landmarks = handler.get_without_detection_without_transform(swap)
                if params[j] == None:     
                    landmarks_tgt = handler.get_without_detection_without_transform(crop_frames[j][i])
                    mask, params[j] = face_mask_static(swap, landmarks, landmarks_tgt, params[j])
                else:
                    mask = face_mask_static(swap, landmarks, landmarks_tgt, params[j])    
                        
                swap = torch.from_numpy(swap).cuda().permute(2,0,1).unsqueeze(0).type(torch.float32)
                mask = torch.from_numpy(mask).cuda().unsqueeze(0).unsqueeze(0).type(torch.float32)
                full_frame = torch.from_numpy(result_frames[i]).cuda().permute(2,0,1).unsqueeze(0)
                mat = torch.from_numpy(tfm_array[j][i]).cuda().unsqueeze(0).type(torch.float32)
                
                mat_rev = kornia.invert_affine_transform(mat)
                swap_t = kornia.warp_affine(swap, mat_rev, size)
                mask_t = kornia.warp_affine(mask, mat_rev, size)
                final = (mask_t*swap_t + (1-mask_t)*full_frame).type(torch.uint8).squeeze().permute(1,2,0).cpu().detach().numpy()
                
                result_frames[i] = final
                torch.cuda.empty_cache()

            except Exception as e:
                pass
                
        out.write(result_frames[i])

    out.release()
    

class Frames(Dataset):
    def __init__(self, frames_list):
        self.frames_list = frames_list
        
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):        
        frame = Image.fromarray(self.frames_list[idx][:,:,::-1])
            
        return self.transforms(frame)

    def __len__(self):
        return len(self.frames_list)


def face_enhancement(final_frames: List[np.ndarray], model) -> List[np.ndarray]:
    enhanced_frames_all = []
    for i in range(len(final_frames)):
        enhanced_frames = final_frames[i].copy()
        face_idx = [i for i, x in enumerate(final_frames[i]) if not isinstance(x, list)]
        face_frames = [x for i, x in enumerate(final_frames[i]) if not isinstance(x, list)]
        ff_i = 0

        dataset = Frames(face_frames)
        dataloader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=1, drop_last=False)

        for iteration, data in tqdm(enumerate(dataloader)):
            frames = data
            data = {'image': frames, 'label': frames}
            generated = model(data, mode='inference2')
            generated = torch.clamp(generated*255, 0, 255)
            generated = (generated).type(torch.uint8).permute(0,2,3,1).cpu().detach().numpy()
            for generated_frame in generated:
                enhanced_frames[face_idx[ff_i]] = generated_frame[:,:,::-1]
                ff_i+=1
        enhanced_frames_all.append(enhanced_frames)
        
    return enhanced_frames_all
