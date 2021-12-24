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

    # generator for face swap
    G = AEI_Net(c_id=512)
    G.eval()
    G.load_state_dict(torch.load(args.G_path, map_location=torch.device('cpu')))
    G = G.cuda()
    G = G.half()

    # arcface model for source embeddings getting
    netArc = iresnet100(fp16=False)
    netArc.load_state_dict(torch.load('arcface_model/backbone.pth'))
    netArc=netArc.cuda()
    netArc.eval()

    # model for face landmarks getting 
    handler = Handler('./coordinate_reg/model/2d106det', 0, ctx_id=0, det_size=640)

    # model for super resolution
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
    
    # get crops from source images
    print(args.source_paths)
    source = []
    try:
        for source_path in args.source_paths: 
            img = cv2.imread(source_path)
            img = crop_face(img, app, args.crop_size)[0]
            source.append(img[:, :, ::-1])
        print("Everything is ok!")
    except TypeError:
        print("Bad source images")
    
    # get full frames from video
    if args.image_to_video:
        full_frames, fps = read_video(args.target_video)
    else:
        target_full = cv2.imread(args.target_path)
        full_frames = [target_full]
    
    # get target faces that are used for swap
    print(args.target_faces_paths)
    if not args.target_faces_paths:
        target = get_target(full_frames, app, args.crop_size)
    else:
        target = []
        try:
            for target_faces_path in args.target_faces_paths: 
                img = cv2.imread(target_faces_path)
                img = crop_face(img, app, args.crop_size)[0]
                target.append(img)
        except TypeError:
            print("Bad target images")
        
    
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
    
    if args.image_to_video:
        get_final_video(final_frames_list,
                        crop_frames_list,
                        full_frames,
                        tfm_array_list,
                        args.out_video_name,
                        fps, 
                        handler)
        
        add_audio_from_another_video(args.target_video, args.out_video_name, "audio")
        print(f"Video saved with path {args.out_video_name}")
    else:
        result = get_final_image(final_frames_list[0], crop_frames_list[0], full_frames[0], tfm_array_list[0], handler)
        cv2.imwrite(args.out_image_name, result)
        print(f'Swapped Image saved with path {args.out_image_name}')     
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # dataset params
    parser.add_argument('--G_path', default='weights/G_0_035000_init_arch_arcface2.pth', type=str, help='Path to weights for G')
    parser.add_argument('--batch_size', default=40, type=int)
    parser.add_argument('--crop_size', default=224, type=int, help="Don't change this")
    parser.add_argument('--use_sr', default=False, type=bool)
    
    parser.add_argument('--multi_swap', default=True, type=bool, help="True for multi swap; you can set multiple faces in source_paths and targes_faces_paths")
    parser.add_argument('--source_paths', default=['examples/images/mark.jpeg'], nargs='+')
    parser.add_argument('--target_faces_paths', default=[], nargs='+', help="It's necessary to set the face/faces in the video to which the source face/faces is swapped. You can skip this parametr, and then any face is selected in the target video for swap.")
    
    parser.add_argument('--image_to_video', default=True, type=bool, help='True for image to video swap, False for image to image swap')
    parser.add_argument('--target_video', default='examples/videos/efiop_short.mp4', type=str, help="It's necessary for image to video swap")
    parser.add_argument('--out_video_name', default='examples/results/result.mp4', type=str, help="It's necessary for image to video swap")
    
    parser.add_argument('--target_path', default='examples/images/elon_musk.jpg', type=str, help="It's necessary for image to image swap")
    parser.add_argument('--out_image_name', default='examples/results/result.png', type=str,help="It's necessary for image to image swap")
    
    args = parser.parse_args()
    main(args)