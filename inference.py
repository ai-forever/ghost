import sys
import argparse

import cv2
import torch
import time
import os
import cv2

from utils.inference.image_processing import crop_face, get_final_image
from utils.inference.video_processing import read_video, get_final_video_frame, add_audio_from_another_video, face_enhancement
from utils.inference.core import model_inference

from network.AEI_Net import AEI_Net
from coordinate_reg.image_infer import Handler
from insightface_func.face_detect_crop_single import Face_detect_crop
from arcface_model.iresnet import iresnet100
from models.pix2pix_model import Pix2PixModel
from models.config_sr import TestOptions

def main(args):
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))

    # основная модель для генерации
    G = AEI_Net(c_id=512)
    G.eval()
    G.load_state_dict(torch.load(args.G_path, map_location=torch.device('cpu')))
    G = G.cuda()
    G = G.half()

    # модель arcface для того, чтобы достать эмбеддинг лица
    netArc = iresnet100(fp16=False)
    netArc.load_state_dict(torch.load('arcface_model/backbone.pth'))
    netArc=netArc.cuda()
    netArc.eval()

    # модель, которая позволяет находить точки лица
    handler = Handler('./coordinate_reg/model/2d106det', 0, ctx_id=0, det_size=640)

    # модель, увеличивающая четкость лица
    if args.use_sr:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        torch.backends.cudnn.benchmark = True
        opt = TestOptions()
        #opt.which_epoch ='10_7'
        model = Pix2PixModel(opt)
        model.netG.train()
    
    source_full = cv2.imread(args.source_path)
    OUT_VIDEO_NAME = "examples/results/result.mp4"
    crop_size = 224 # don't change this
    
    # check, if we can detect face on the source image
    try:
        source = crop_face(source_full, app, crop_size)[0]
        source = source[:, :, ::-1]
        print("Everything is ok!")
    except TypeError:
        print("Bad source image. Choose another one.")
        
    if args.image_to_image:
        target_full = cv2.imread(args.target_path)
        full_frames = [target_full]
    else:
        full_frames, fps = read_video(args.target_path)
    
    final_frames, crop_frames, full_frames, tfm_array = model_inference(full_frames,
                                                                    source,
                                                                    [netArc],
                                                                    G,
                                                                    app, 
                                                                    crop_size=crop_size,
                                                                    BS=args.batch_size)
    if args.use_sr:
        final_frames = face_enhancement(final_frames, model)
        
    if args.image_to_image:
        result = get_final_image(final_frames[0], crop_frames[0], full_frames[0], tfm_array[0], handler)
        cv2.imwrite('examples/results/result.png', result)
        print('Swapped Image saved with path examples/results/result.png')
        
    else:
        get_final_video_frame(final_frames,
                        crop_frames,
                        full_frames,
                        tfm_array,
                        OUT_VIDEO_NAME,
                        fps, 
                        handler)

        add_audio_from_another_video(args.target_path, OUT_VIDEO_NAME, "audio")

        print(f"Video saved with path {OUT_VIDEO_NAME}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # dataset params
    parser.add_argument('--G_path', default='weights/G_0_035000_init_arch_arcface2.pth', type=str, help='Path to weights for G')
    parser.add_argument('--batch_size', default=40, type=int)
    parser.add_argument('--use_sr', default=False, type=bool)
    parser.add_argument('--source_path', default='examples/images/elon_musk.jpg', type=str)
    parser.add_argument('--target_path', default='examples/images/beckham.jpg', type=str)
    parser.add_argument('--image_to_image', default=False, type=bool, help='True for image to image swap, False for swap on video')
    

    args = parser.parse_args()
    
    main(args)
