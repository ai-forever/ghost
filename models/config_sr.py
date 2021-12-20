import sys

class TestOptions(object):
    name = 'weights'
    results_dir = './results/'
    gpu_ids = [0]
    crop_size = 256
    dataset_mode = 'test'
    which_epoch = '10'

    aspect_ratio = 1.0
    batchSize = 1
    cache_filelist_read = True
    cache_filelist_write = True
    checkpoints_dir = ''
    coco_no_portraits = False
    contain_dontcare_label = False

    display_winsize = 256
    how_many = sys.maxsize
    how_many = 10
    init_type = 'xavier'
    init_variance = 0.02
    isTrain = False
    is_test = True
    label_nc = 3
    output_nc = 3
    semantic_nc = 3
    load_from_opt_file = False
    max_dataset_size = sys.maxsize
    
    model = 'pix2pix'
    nThreads = 0
    netG = 'lipspade'
    nef = 16
    ngf = 48 
    no_flip = True
    no_instance = True
    no_pairing_check = False

    norm_D = 'spectralinstance'
    norm_E = 'spectralinstance'
    norm_G = 'spectralspadesyncbatch3x3'
    num_upsampling_layers = 'normal'
    phase = 'test'
    prd_resize = 256
    #preprocess_mode = 'resize_and_crop'
    serial_batches = True
    use_vae = False
    z_dim = 256
