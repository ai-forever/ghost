[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9851423)] [[Habr](https://habr.com/ru/company/sberbank/blog/645919/)]

# 👻 GHOST: Generative High-fidelity One Shot Transfer 

Our paper ["GHOST—A New Face Swap Approach for Image and Video Domains"](https://ieeexplore.ieee.org/abstract/document/9851423) has been published on IEEE Xplore.

<p align="center">
  Google Colab Demo
</p>
<p align="center">
  <a href="https://colab.research.google.com/drive/1B-2JoRxZZwrY2eK_E7TB5VYcae3EjQ1f">
  <img src="https://colab.research.google.com/assets/colab-badge.svg"/>
  </a>
</p>

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
  git clone https://github.com/sberbank-ai/sber-swap.git
  cd sber-swap
  git submodule init
  git submodule update
  ```
2. Install dependent packages
  ```bash
  pip install -r requirements.txt
  ```
  If it is not possible to install onnxruntime-gpu, try onnxruntime instead  
  
3. Download weights
  ```bash
  sh download_models.sh
  ```
## Usage
  1. Colab Demo <a href="https://colab.research.google.com/drive/1B-2JoRxZZwrY2eK_E7TB5VYcae3EjQ1f"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> or you can use jupyter notebook [SberSwapInference.ipynb](SberSwapInference.ipynb) locally
  2. Face Swap On Video
  
  Swap to one specific person in the video. You must set face from the target video (for example, crop from any frame).
  ```bash
  python inference.py --source_paths {PATH_TO_IMAGE} --target_faces_paths {PATH_TO_IMAGE} --target_video {PATH_TO_VIDEO}
  ```
  Swap to many person in the video. You must set multiple faces for source and the corresponding multiple faces from the target video.
  ```bash
  python inference.py --source_paths {PATH_TO_IMAGE PATH_TO_IMAGE ...} --target_faces_paths {PATH_TO_IMAGE PATH_TO_IMAGE ...} --target_video {PATH_TO_VIDEO}
  ```
  3. Face Swap On Image
  
  You may set the target face, and then source will be swapped on this person, or you may skip this parameter, and then source will be swapped on any person in the image.
  ```bash
  python inference.py --target_path {PATH_TO_IMAGE} --image_to_image True
  ```
  
## Training
  
We also provide the training code for face swap model as follows:
  1. Download [VGGFace2 Dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face/).
  2. Crop and align faces with out detection model.
  ```bash
  python preprocess_vgg.py --path_to_dataset {PATH_TO_DATASET} --save_path {SAVE_PATH}
  ```
  3. Start training. 
  ```bash
  python train.py --run_name {YOUR_RUN_NAME}
  ```
We provide a lot of different options for the training. More info about each option you can find in `train.py` file. If you would like to use wandb logging of the experiments, you should login to wandb first  `--wandb login`.
  
### Tips:
  1. For first epochs we suggest not to use eye detection loss and scheduler if you train from scratch.
  2. In case of finetuning model you can variate losses coefficients to make result look more like source identity, or vice versa, save features and attributes of target face.
  3. You can change backbone for attribute encoder and num_blocks of AAD ResBlk using parameters `--backbone` and `--num_blocks`.
  4. For finetuning model you can use our pretrain weights for generator and discriminator that are in folder `weights`. We provide weights for models with unet backbone and 1-3 blocks in AAD ResBlk. The main model is model with 2 blocks in AAD ResBlk.
  
