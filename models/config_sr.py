import sys

class TestOptions(object):
    name = 'weights'
    results_dir = './results/'
    gpu_ids = [0]
    crop_size = 256
    dataset_mode = 'test'
    which_epoch = '10'

    aspect_ratio = 1.0
    checkpoints_dir = ''
    
    init_type = 'xavier'
    init_variance = 0.02
    isTrain = False
    is_test = True
    semantic_nc = 3
    
    model = 'pix2pix'
    netG = 'lipspade'
    nef = 16
    ngf = 48 
    
    norm_G = 'spectralspadesyncbatch3x3'
    num_upsampling_layers = 'normal'
    phase = 'test'
    use_vae = False
    z_dim = 256
