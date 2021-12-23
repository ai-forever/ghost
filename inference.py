import sys
import argparse
import cv2
import torch
import time
import os

from utils.inference.image_processing import crop_face, get_final_image
from utils.inference.video_processing import read_video, get_target, get_final_video, add_audio_from_another_video, face_enhancement
from utils.inference.core import model_inference

from network.AEI_Net import AEI_Net
from coordinate_reg.image_infer import Handler
from insightface_func.face_detect_crop_multi import Face_detect_crop
from arcface_model.iresnet import iresnet100
from models.pix2pix_model import Pix2PixModel
from models.config_sr import TestOptions


def init_models(args):
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
    else:
        model = None
    
    return app, G, netArc, handler, model
    
    
def main(args):
    app, G, netArc, handler, model = init_models(args)
    
    source_full = cv2.imread(args.source_path)
    source_images = [source_full]
    source = []
    try:
        for source_image in source_images:     
            source.append(crop_face(source_image, app, args.crop_size)[0][:, :, ::-1])
        print("Everything is ok!")
    except TypeError:
        print("Bad source images")
        
    if args.image_to_image:
        target_full = cv2.imread(args.target_path)
        full_frames = [target_full]
    else:
        full_frames, fps = read_video(args.target_video)
    target = get_target(full_frames, app, args.crop_size)
    
    final_frames_list, crop_frames_list, full_frames, tfm_array_list = model_inference(full_frames,
                                                                                   source,
                                                                                   target,
                                                                                   netArc,
                                                                                   G,
                                                                                   app, 
                                                                                   crop_size=args.crop_size,
                                                                                   BS=args.batch_size)
    
    if args.use_sr:
        final_frames_list = face_enhancement(final_frames_list, model)
    
    if args.image_to_image:
        result = get_final_image(final_frames_list[0], crop_frames_list[0], full_frames[0], tfm_array_list[0], handler)
        cv2.imwrite(args.out_image_name, result)
        print(f'Swapped Image saved with path {args.out_image_name}')
        
    else:
        get_final_video(final_frames_list,
                        crop_frames_list,
                        full_frames,
                        tfm_array_list,
                        args.out_video_name,
                        fps, 
                        handler)

        add_audio_from_another_video(args.target_video, args.out_video_name, "audio")

        print(f"Video saved with path {args.out_video_name}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # dataset params
    parser.add_argument('--G_path', default='weights/G_0_035000_init_arch_arcface2.pth', type=str, help='Path to weights for G')
    parser.add_argument('--batch_size', default=40, type=int)
    parser.add_argument('--crop_size', default=224, type=int, help="Don't change this")
    parser.add_argument('--use_sr', default=False, type=bool)
    parser.add_argument('--source_path', default='examples/images/mark.jpeg', type=str)
    parser.add_argument('--target_path', default='examples/images/elon_musk.jpg', type=str)
    parser.add_argument('--target_video', default='examples/videos/efiop_short.mp4', type=str)
    parser.add_argument('--out_video_name', default='examples/results/result.mp4', type=str)
    parser.add_argument('--out_image_name', default='examples/results/result.png', type=str)
    parser.add_argument('--image_to_image', default=True, type=bool, help='True for image to image swap, False for swap on video')
    args = parser.parse_args()
    
    main(args)