import os
import sys
import cv2
import argparse
from insightface_func.face_detect_crop_single import Face_detect_crop
from pathlib import Path
from tqdm import tqdm

def main(args):
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))
    crop_size = 224

    dirs = os.listdir(args.path_to_dataset)
    for i in tqdm(range(len(dirs))):
        d = os.path.join(args.path_to_dataset, dirs[i])
        dir_to_save = os.path.join(args.save_path, dirs[i])
        Path(dir_to_save).mkdir(parents=True, exist_ok=True)
        
        image_names = os.listdir(d)
        for image_name in image_names:
            try:
                image_path = os.path.join(d, image_name)
                image = cv2.imread(image_path)
                cropped_image, _ = app.get(image, crop_size)
                cv2.imwrite(os.path.join(dir_to_save, image_name), cropped_image[0])
            except:
                pass
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path_to_dataset', default='./VggFace2/VGG-Face2/data/preprocess_train', type=str)
    parser.add_argument('--save_path', default='./VggFace2-crop', type=str)
    
    args = parser.parse_args()
    
    main(args)
