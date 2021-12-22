# sber-swap

## Usage
1. Colab Demo

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
## Training
  
We also provide the training code for face swap model as follows:
  1. Download [VGGFace2 Dataset] (https://www.robots.ox.ac.uk/~vgg/data/vgg_face/).
  2. Crop and align faces with out detection model.
  3. Start training. 
  > python train.py
