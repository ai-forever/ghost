import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from AdaptiveWingLoss.utils.utils import get_preds_fromhm
from .image_processing import torch2image


transforms_base = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


def detect_landmarks(inputs, model_ft):
    mean = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2).to(inputs.device)
    std = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2).to(inputs.device)
    inputs = (std * inputs) + mean

    outputs, boundary_channels = model_ft(inputs)    
    pred_heatmap = outputs[-1][:, :-1, :, :].cpu() 
    pred_landmarks, _ = get_preds_fromhm(pred_heatmap)
    landmarks = pred_landmarks*4.0
    eyes = torch.cat((landmarks[:,96,:], landmarks[:,97,:]), 1)
    return eyes, pred_heatmap[:,96,:,:], pred_heatmap[:,97,:,:]


def paint_eyes(images, eyes):
    list_eyes = []
    for i in range(len(images)):
        mask = torch2image(images[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) 
        
        cv2.circle(mask, (int(eyes[i][0]),int(eyes[i][1])), radius=3, color=(0,255,255), thickness=-1)
        cv2.circle(mask, (int(eyes[i][2]),int(eyes[i][3])), radius=3, color=(0,255,255), thickness=-1)
        
        mask = mask[:, :, ::-1]
        mask = transforms_base(Image.fromarray(mask))
        list_eyes.append(mask)
    tensor_eyes = torch.stack(list_eyes)
    return tensor_eyes