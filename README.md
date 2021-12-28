# SberSwap

## Results 
![](/examples/images/example1.png)

![](/examples/images/example2.png)

## Video Swap
<div>
<img src="/examples/videos/orig.webp" width="360"/>
<img src="/examples/videos/elon.webp" width="360"/>
<img src="/examples/videos/khabenskii.webp" width="360"/>
<img src="/examples/videos/mark.webp" width="360"/>
<div/>

## Installation
  
1. Clone this repository
  ```bash
  git clone https://github.com/Danyache/sber-swap.git
  cd sber-swap
  git submodule init
  git submodule update
  ```
2. Install dependent packages
  ```bash
  pip install -r requirements.txt
  ```
3. Download weights
  ```bash
  sh download_models.sh
  ```
## Usage
  1. Colab Demo or you can use jupyter notebook [SberSwapInference.ipynb](SberSwapInference.ipynb) locally
  2. Face Swap On Video
  > python inference.py 
  3. Face Swap On Image
  > python inference.py --target_path {PATH_TO_IMAGE} --image_to_image True
  
## Training
  
We also provide the training code for face swap model as follows:
  1. Download [VGGFace2 Dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face/).
  2. Crop and align faces with out detection model.
  > python preprocess_vgg.py --path_to_dataset ./VggFace2/VGG-Face2/data/preprocess_train --save_path ./VggFace2-crop
  3. Start training. 
  > python train.py --run_name {YOUR_RUN_NAME}

We provide a lot of different options for the training. More info about each option you can find in `train.py` file. If you would like to use wandb logging of the experiments, you should login to wandb first -- `wandb login`.
  
### Tips:
  1. For first epochs we suggest not to use eye detection loss if you train from scratch
  2. In case of finetuning model you can variate losses coefficients to make result look more like source identity, or vice versa, save features and attributes of target face
  
