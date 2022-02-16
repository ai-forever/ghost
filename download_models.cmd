# load arcface
wget -P ./arcface_model https://github.com/sberbank-ai/sber-swap/releases/download/arcface/backbone.pth
wget -P ./arcface_model https://github.com/sberbank-ai/sber-swap/releases/download/arcface/iresnet.py

# load landmarks detector
wget -P ./insightface_func/models/antelope https://github.com/sberbank-ai/sber-swap/releases/download/antelope/glintr100.onnx
wget -P ./insightface_func/models/antelope https://github.com/sberbank-ai/sber-swap/releases/download/antelope/scrfd_10g_bnkps.onnx

# load G and D models with 1, 2, 3 blocks
# model with 2 blocks is main
wget -P ./weights https://github.com/sberbank-ai/sber-swap/releases/download/sber-swap-v2.0/G_unet_2blocks.pth
wget -P ./weights https://github.com/sberbank-ai/sber-swap/releases/download/sber-swap-v2.0/D_unet_2blocks.pth

wget -P ./weights https://github.com/sberbank-ai/sber-swap/releases/download/sber-swap-v2.0/G_unet_1block.pth
wget -P ./weights https://github.com/sberbank-ai/sber-swap/releases/download/sber-swap-v2.0/D_unet_1block.pth

wget -P ./weights https://github.com/sberbank-ai/sber-swap/releases/download/sber-swap-v2.0/G_unet_3blocks.pth
wget -P ./weights https://github.com/sberbank-ai/sber-swap/releases/download/sber-swap-v2.0/D_unet_3blocks.pth

# load model for eyes loss
wget -P ./AdaptiveWingLoss/AWL_detector https://github.com/sberbank-ai/sber-swap/releases/download/awl_detector/WFLW_4HG.pth

# load super res model
wget -P ./weights https://github.com/sberbank-ai/sber-swap/releases/download/super-res/10_net_G.pth
