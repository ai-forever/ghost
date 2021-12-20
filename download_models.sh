# load arcface
wget -P ./arcface_model https://github.com/Danyache/sber-swap/releases/download/arcface/backbone.pth
wget -P ./arcface_model https://github.com/Danyache/sber-swap/releases/download/arcface/iresnet.py

# load landmarks detector
wget -P ./insightface_func/models/antelope https://github.com/Danyache/sber-swap/releases/download/antelope/glintr100.onnx
wget -P ./insightface_func/models/antelope https://github.com/Danyache/sber-swap/releases/download/antelope/scrfd_10g_bnkps.onnx

# load model itself
wget -P ./weights https://github.com/Danyache/sber-swap/releases/download/sber-swap-v1.0/G_0_035000_init_arch_arcface2.pth

# load model for eyes loss
wget -P ./AdaptiveWingLoss/AWL_detector https://github.com/Danyache/sber-swap/releases/download/awl_detector/WFLW_4HG.pth

# load super res model
wget -P ./weights https://github.com/Danyache/sber-swap/releases/download/super-res/10_net_G.pth
