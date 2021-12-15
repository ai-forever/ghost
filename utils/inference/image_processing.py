import base64
from io import BytesIO
from typing import Callable

import numpy as np
import torch
import cv2
from .masks import face_mask_static # , face_mask


# def encode_img(img):
#     """
#     Encode np.array image to bs64 format
#     """
#     jpg_img = cv2.imencode('.jpg', img)
#     img_b64 = base64.b64encode(jpg_img[1]).decode('utf-8')
#     return img_b64


# def decode_img(img_b64):
#     """
#     Decode bs64 image to np.ndarray format
#     """
#     bin_img = base64.b64decode(img_b64)
#     buff = BytesIO(bin_img)
#     img = cv2.imdecode(np.frombuffer(buff.getbuffer(), np.uint8), -1)
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     return img

# used
def crop_face(image_full: np.ndarray, app: Callable, crop_size: int) -> np.ndarray:
    """
    Crop face from image and resize
    """
    image, _ = app.get(image_full, crop_size)
    return image

# used
def normalize_and_torch(image: np.ndarray) -> torch.tensor:
    """
    Normalize image and transform to torch
    """
    image = torch.tensor(image.copy(), dtype=torch.float32).cuda()
    if image.max() > 1.:
        image = image/255.
    
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = (image - 0.5) / 0.5

    return image


# def get_final_image(final_frame: np.ndarray,
#                     crop_frame: np.ndarray,
#                     full_frame: np.ndarray,
#                     tfm_array: np.ndarray,
#                     handler) -> None:
#     """
#     Create final video from frames
#     """
#     params = None
#     landmarks = handler.get_without_detection_without_transform(final_frame)     
#     landmarks_tgt = handler.get_without_detection_without_transform(crop_frame)
                
#     mask, _ = face_mask_static(crop_frame, landmarks, landmarks_tgt, params)
#     mat_rev = cv2.invertAffineTransform(tfm_array)

#     frame = cv2.resize(final_frame, (224, 224))
#     swap_t = cv2.warpAffine(frame, mat_rev, (full_frame.shape[1], full_frame.shape[0]), borderMode=cv2.BORDER_REPLICATE)
#     mask_t = cv2.warpAffine(mask, mat_rev, (full_frame.shape[1], full_frame.shape[0]))
#     mask_t = np.expand_dims(mask_t, 2)

#     final = mask_t*swap_t + (1-mask_t)*full_frame
#     final = np.array(final, dtype='uint8')
#     return final