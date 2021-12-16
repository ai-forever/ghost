"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
import os
#import data  # only run from basic level!
import copy  # deepcopy

class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh, self.scale_ratio = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        # 20200211 test 4x with only 3 stage

        self.ups = nn.ModuleList([
            SPADEResnetBlock(16 * nf, 8 * nf, opt),
            SPADEResnetBlock(8 * nf, 4 * nf, opt),
            SPADEResnetBlock(4 * nf, 2 * nf, opt),
            SPADEResnetBlock(2 * nf, 1 * nf, opt)  # here
            ])

        self.to_rgbs = nn.ModuleList([
            nn.Conv2d(8 * nf, 3, 3, padding=1),
            nn.Conv2d(4 * nf, 3, 3, padding=1),
            nn.Conv2d(2 * nf, 3, 3, padding=1),
            nn.Conv2d(1 * nf, 3, 3, padding=1)      # here
            ])

        self.up = nn.Upsample(scale_factor=2)
        
    # 20200309 interface for flexible encoder design
    # and mid-level loss control!
    # For basic network, it's just a 16x downsampling
    def encode(self, input):
        h, w = input.size()[-2:]
        sh, sw = h//2**self.scale_ratio, w//2**self.scale_ratio
        x = F.interpolate(input, size=(sh, sw))
        return self.fc(x) # 20200310: Merge fc into encoder

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        # 20200211 Yang Lingbo with respect to phase
        scale_ratio = num_up_layers
        #scale_ratio = 4  #here
        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh, scale_ratio

    def forward(self, input, seg=None):
        '''
            20200307: Dangerous Change
            Add separable forward to allow different 
            input and segmentation maps...
            
            To return to original, simply add
            seg = input at begining, and disable the seg parameter.
            
            20200308: A more elegant solution:
            @ Allow forward to take default parameters.
            @ Allow customizable input encoding
            
            20200310: Merge fc into encode, since encoder directly outputs processed feature map.
            
            [TODO] @ Allow customizable segmap encoding?
        '''
        
        if seg is None:
            seg = input # Interesting change...
            
        # For basic generator, 16x downsampling.
        # 20200310: Merge fc into encoder
        x = self.encode(input)
        #print(x.shape, input.shape, seg.shape)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        
        if self.opt.is_test:
            phase = len(self.to_rgbs)
        else:
            phase = self.opt.train_phase+1

        for i in range(phase):
            x = self.up(x)
            x = self.ups[i](x, seg)
        
        x = self.to_rgbs[phase-1](F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x
        
    def mixed_guidance_forward(self, input, seg=None, n=0, mode='progressive'):
        '''
            mixed_forward: input and seg are different images
            For the first n levels (including encoder)
            we use input, for the rest we use seg.
            
            If mode = 'progressive', the output's like: AAABBB
            If mode = 'one_plug', the output's like:    AAABAA
            If mode = 'one_ablate', the output's like:  BBBABB
        '''
        
        if seg is None:
            return self.forward(input)
            
        if self.opt.is_test:
            phase = len(self.to_rgbs)
        else:
            phase = self.opt.train_phase+1
        
        if mode == 'progressive':
            n = max(min(n, 4 + phase), 0)
            guide_list = [input] * n + [seg] * (4+phase-n)
        elif mode == 'one_plug':
            n = max(min(n, 4 + phase-1), 0)
            guide_list = [seg] * (4+phase)
            guide_list[n] = input
        elif mode == 'one_ablate':
            if n > 3+phase:
                return self.forward(input)
            guide_list = [input] * (4+phase)
            guide_list[n] = seg
        
        x = self.encode(guide_list[0])
        x = self.head_0(x, guide_list[1])

        x = self.up(x)
        x = self.G_middle_0(x, guide_list[2])
        x = self.G_middle_1(x, guide_list[3])
        
        for i in range(phase):
            x = self.up(x)
            x = self.ups[i](x, guide_list[4+i])
        
        x = self.to_rgbs[phase-1](F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x

class HiFaceGANGenerator(SPADEGenerator):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh, self.scale_ratio = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt, 16 * nf)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt, 16 * nf)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt, 16 * nf)

        # 20200211 test 4x with only 3 stage

        self.ups = nn.ModuleList([
            SPADEResnetBlock(16 * nf, 8 * nf, opt, 8 * nf),
            SPADEResnetBlock(8 * nf, 4 * nf, opt, 4 * nf),
            SPADEResnetBlock(4 * nf, 2 * nf, opt, 2 * nf),
            SPADEResnetBlock(2 * nf, 1 * nf, opt, 1 * nf)  # here
            ])

        self.to_rgbs = nn.ModuleList([
            nn.Conv2d(8 * nf, 3, 3, padding=1),
            nn.Conv2d(4 * nf, 3, 3, padding=1),
            nn.Conv2d(2 * nf, 3, 3, padding=1),
            nn.Conv2d(1 * nf, 3, 3, padding=1)      # here
            ])

        self.up = nn.Upsample(scale_factor=2)
        self.encoder = ContentAdaptiveSuppresor(opt, self.sw, self.sh, self.scale_ratio)
        
    def nested_encode(self, x):
        return self.encoder(x)

    def forward(self, input):
        xs = self.nested_encode(input)
        x = self.encode(input)
        '''
        print([_x.shape for _x in xs])
        print(x.shape)
        print(self.head_0)
        '''
        x = self.head_0(x, xs[0])

        x = self.up(x)
        x = self.G_middle_0(x, xs[1])
        x = self.G_middle_1(x, xs[1])
        
        if self.opt.is_test:
            phase = len(self.to_rgbs)
        else:
            phase = self.opt.train_phase+1

        for i in range(phase):
            x = self.up(x)
            x = self.ups[i](x, xs[i+2])
        
        x = self.to_rgbs[phase-1](F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x


class ContentAdaptiveSuppresor(BaseNetwork):
    def __init__(self, opt, sw, sh, n_2xdown,
            norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.sw = sw
        self.sh = sh
        self.max_ratio = 16
        self.n_2xdown = n_2xdown
        
        # norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        
        # 20200310: Several Convolution (stride 1) + LIP blocks, 4 fold
        ngf = opt.ngf
        kw = 3
        pw = (kw - 1) // 2
        
        self.head = nn.Sequential(
            nn.Conv2d(opt.semantic_nc, ngf, kw, stride=1, padding=pw, bias=False),
            norm_layer(ngf),
            nn.ReLU(),
        )
        cur_ratio = 1
        for i in range(n_2xdown):
            next_ratio = min(cur_ratio*2, self.max_ratio)
            model = [
                SimplifiedLIP(ngf*cur_ratio),
                nn.Conv2d(ngf*cur_ratio, ngf*next_ratio, kw, stride=1, padding=pw),
                norm_layer(ngf*next_ratio),
            ]
            cur_ratio = next_ratio
            if i < n_2xdown - 1: 
                model += [nn.ReLU(inplace=True)]
            setattr(self, 'encoder_%d' % i, nn.Sequential(*model))
            
    def forward(self, x):
        # 20200628: Note the features are arranged from small to large
        x = [self.head(x)]
        for i in range(self.n_2xdown):
            net = getattr(self, 'encoder_%d' % i)
            x = [net(x[0])] + x
        return x

        
#########################################
# Below are deprecated codes
#
# 20200309: LIP for local importance pooling
# 20200311: Self-supervised mask encoder
#   Author: lingbo.ylb
# Quick trial, to be reformated later.
# 20200324: Nah forget about it...
#########################################


def lip2d(x, logit, kernel=3, stride=2, padding=1):
    weight = logit.exp()
    return F.avg_pool2d(x*weight, kernel, stride, padding) / F.avg_pool2d(weight, kernel, stride, padding)
    

class SoftGate(nn.Module):
    COEFF = 12.0
    
    def __init__(self):
        super(SoftGate, self).__init__()
    
    def forward(self, x):
        return torch.sigmoid(x).mul(self.COEFF)

class SimplifiedLIP(nn.Module):
    def __init__(self, channels):
        super(SimplifiedLIP, self).__init__()

        rp = channels

        self.logit = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            SoftGate()
        )
        '''
        OrderedDict((
            ('conv', nn.Conv2d(channels, channels, 3, padding=1, bias=False)),
            ('bn', nn.InstanceNorm2d(channels, affine=True)),
            ('gate', SoftGate()),
        ))
        '''

    def init_layer(self):
        self.logit[0].weight.data.fill_(0.0)

    def forward(self, x):
        frac = lip2d(x, self.logit(x))
        return frac
        
class LIPEncoder(BaseNetwork):
    def __init__(self, opt, sw, sh, n_2xdown,
            norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.sw = sw
        self.sh = sh
        self.max_ratio = 16
        
        # norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        
        # 20200310: Several Convolution (stride 1) + LIP blocks, 4 fold
        ngf = opt.ngf
        kw = 3
        pw = (kw - 1) // 2
        
        model = [
            nn.Conv2d(opt.semantic_nc, ngf, kw, stride=1, padding=pw, bias=False),
            norm_layer(ngf),
            nn.ReLU(),
        ]
        cur_ratio = 1
        for i in range(n_2xdown):
            next_ratio = min(cur_ratio*2, self.max_ratio)
            model += [
                SimplifiedLIP(ngf*cur_ratio),
                nn.Conv2d(ngf*cur_ratio, ngf*next_ratio, kw, stride=1, padding=pw),
                norm_layer(ngf*next_ratio),
            ]
            cur_ratio = next_ratio
            if i < n_2xdown - 1: 
                model += [nn.ReLU(inplace=True)]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)

class LIPSPADEGenerator(SPADEGenerator):
    '''
        20200309: SPADEGenerator with a learnable feature encoder
        Encoder design: Local Importance-based Pooling (Ziteng Gao et.al.,ICCV 2019)
    '''
    def __init__(self, opt):
        super().__init__(opt)
        self.lip_encoder = LIPEncoder(opt, self.sw, self.sh, self.scale_ratio)
        
    def encode(self, x):
        return self.lip_encoder(x)


class NoiseClassPredictor(nn.Module):
    '''
        Input: nc*sw*sw tensor, either from clean or corrupted images
        Output: n-dim tensor indicating the loss type (or intensity?)
    '''
    def __init__(self, opt, sw, nc, outdim):
        super().__init__()
        nbottleneck = 256
        middim = 256
        #　Compact info
        conv = [
            nn.Conv2d(nc, nbottleneck, 1, stride=1),
            nn.InstanceNorm2d(nbottleneck),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        # sw should be probably 16, downsample to 4 and convert to 1
        while sw > 4:
            sw = sw // 2
            conv += [
                nn.Conv2d(nbottleneck, nbottleneck, 3, stride=2, padding=1),
                nn.InstanceNorm2d(nbottleneck),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            
            
        self.conv = nn.Sequential(*conv)
        
        indim = sw * sw * nbottleneck
        self.fc = nn.Sequential(
            nn.Linear(indim, middim),
            nn.BatchNorm1d(middim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(middim, outdim),
            # nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0],-1)
        return self.fc(x)
        

class NoiseIntensityPredictor(nn.Module):
    '''
        Input: nc*sw*sw tensor, either from clean or corrupted images
        Output: 1-dim tensor indicating the loss intensity
    '''
    def __init__(self, opt, sw, nc, outdim):
        super().__init__()
        nbottleneck = 256
        middim = 256
        #　Compact info
        conv = [
            nn.Conv2d(nc, nbottleneck, 1, stride=1),
            nn.BatchNorm2d(nbottleneck),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        # sw should be probably 16, downsample to 4 and convert to 1
        while sw > 4:
            sw = sw // 2
            conv += [
                nn.Conv2d(nbottleneck, nbottleneck, 3, stride=2, padding=1),
                nn.BatchNorm2d(nbottleneck),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            
            
        self.conv = nn.Sequential(*conv)
        
        indim = sw * sw * nbottleneck
        self.fc = nn.Sequential(
            nn.Linear(indim, middim),
            nn.BatchNorm1d(middim),
            #nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(middim, outdim),
            #nn.Dropout(0.5),
            # nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return x.squeeze()


class SubAddGenerator(SPADEGenerator):
    '''
        20200311: 
        This generator contains a complete set
        of self-supervised training scheme
        that requires a separate dataloader.
        
        The self-supervised pre-training is 
        implemented as a clean interface.
        SubAddGenerator::train_E(self, dataloader, epochs)
        
        For the upperlevel Pix2Pix_Model, 
        two things to be done:
        A) Run the pretrain script
        B) Save encoder and adjust learning rate.
        
        -----------------------------------------
        20200312:
        Pre-test problem: The discriminator is hard to test real vs fake
        Also, using residual of feature maps ain't work...
        
        Cause: The feature map is too close to be properly separated.
        
        Try to test on one single reduction: arbitrary ratio of downsampling
        try to estimate the reduction ratio?
    '''
    def __init__(self, opt):
        super().__init__(opt)
        self.encoder = LIPEncoder(opt, self.sw, self.sh, self.scale_ratio)
    
        self.dis_nc = self.opt.ngf * min(16, 2**self.scale_ratio)
        # intensity is a scalar
        self.discriminator = NoiseIntensityPredictor(opt, self.sw, self.dis_nc, 1)
        if opt.isTrain:
            self.attach_dataloader(opt)
            self.noise_dim = opt.noise_dim
            
            self.l1_loss = nn.L1Loss()
            self.gan_loss = nn.MSELoss()
        
            #self.discriminator = NoiseClassPredictor(opt, self.sw, self.dis_nc, 
            #    self.noise_dim + 1)  # add a new label for clean images
            
            #self.gan_loss = nn.CrossEntropyLoss()
            
            beta1, beta2 = opt.beta1, opt.beta2
            if opt.no_TTUR:
                G_lr, D_lr = opt.lr, opt.lr
            else:
                G_lr, D_lr = opt.lr / 2, opt.lr * 2

            self.optimizer_E = torch.optim.Adam(
                self.encoder.parameters(), lr=G_lr, betas=(beta1, beta2)
            )
            self.optimizer_D = torch.optim.Adam(
                self.discriminator.parameters(), lr=D_lr/2, betas=(beta1, beta2)
            )
    
    def _create_auxiliary_opt(self, opt):
        '''
            Create auxiliary options
            change necessary params
            --- dataroot
            --- dataset_mode
        '''
        aux_opt = copy.deepcopy(opt)  # just for safety
        aux_opt.dataroot = opt.dataroot_assist
        aux_opt.dataset_mode = 'assist'
        aux_opt.batchSize = 4
        aux_opt.nThreads = 4
        return aux_opt
    
    def attach_dataloader(self, opt):
        aux_opt = self._create_auxiliary_opt(opt)
        self.loader = data.create_dataloader(aux_opt)
    
    def encode(self, x):
        return self.encoder(x)
    
    def process_input(self, data):
        self.clean = data['clean'].cuda()
        self.noisy = data['noisy'].cuda()
        # for BCELoss, class label is just int.
        self.noise_label = data['label'].cuda()
        # label design...
        # clean label should be 0 or [1,0,0...0]?
        # for BCELoss, class label is just 0.
        #self.clean_label = torch.zeros_like(self.noise_label)
        self.clean_label = torch.ones_like(self.noise_label)
        
    def update_E(self):
        bundle_in = torch.cat((self.clean, self.noisy), dim=0)
        bundle_out = self.encode(bundle_in)
        nb = bundle_in.shape[0] // 2
        F_real, F_fake = bundle_out[:nb], bundle_out[nb:]
        
        pred_fake = self.discriminator(F_fake)
        loss_l1 = self.l1_loss(F_fake, F_real)
        loss_gan = self.gan_loss(pred_fake, self.clean_label)
        loss_sum = loss_l1 * 10 + loss_gan
        
        self.optimizer_E.zero_grad()
        loss_sum.backward()
        self.optimizer_E.step()
        
        self.loss_l1 = loss_l1.item()
        self.loss_gan_E = loss_gan.item()
        self.loss_sum = loss_sum.item()
        
    def update_D(self):
    
        with torch.no_grad():
            bundle_in = torch.cat((self.clean, self.noisy), dim=0)
            bundle_out = self.encode(bundle_in)
            nb = bundle_in.shape[0] // 2
            #F_real, #F_fake = bundle_out[:nb], bundle_out[nb:]
            F_fake = bundle_out[nb:] / (bundle_out[:nb] + 1e-6)
            F_real = torch.ones_like(F_fake, requires_grad=False)
            
        pred_real = self.discriminator(F_real)
        loss_real = self.gan_loss(pred_real, self.clean_label)
        pred_fake = self.discriminator(F_fake.detach())
        loss_fake = self.gan_loss(pred_fake, self.noise_label)
        loss_sum = (loss_real + loss_fake * self.opt.noise_dim) / 2
        
        self.optimizer_D.zero_grad()
        loss_sum.backward()
        self.optimizer_D.step()
        
        self.loss_gan_D_real = loss_real.item()
        self.loss_gan_D_fake = loss_fake.item()
        
    def debug_D(self):
        with torch.no_grad():
            bundle_in = torch.cat((self.clean, self.noisy), dim=0)
            bundle_out = self.encode(bundle_in)
            nb = bundle_in.shape[0] // 2
            F_real, F_fake = bundle_out[:nb], bundle_out[nb:]
            F_res = F_fake - F_real # try to predict the residual, it's easier
            #F_real = torch.zeros_like(F_real) # real res == 0
            
        
        pred_real = self.discriminator(F_real)#.argmax(dim=1)
        pred_fake = self.discriminator(F_res.detach())#.argmax(dim=1)
        print(pred_real, pred_fake)
        real_acc = (pred_real == 0).sum().item() / pred_real.shape[0]
        fake_acc = (pred_fake == self.noise_label).sum().item() / pred_fake.shape[0]
        print(real_acc, fake_acc)
        
    def log(self, epoch, i):
        logstring = '   Epoch [%d] iter [%d]: ' % (epoch, i)
        logstring += 'l1: %.4f ' % self.loss_l1
        logstring += 'gen: %.4f ' % self.loss_gan_E
        logstring += 'E_sum: %.4f ' % self.loss_sum
        logstring += 'dis_real: %.4f ' % self.loss_gan_D_real
        logstring += 'dis_fake: %.4f' % self.loss_gan_D_fake
        print(logstring)
        
    def train_E(self, epochs):
        pretrained_ckpt_dir = os.path.join(
            self.opt.checkpoints_dir, self.opt.name,
            'pretrained_net_E_%d.pth' % epochs
        )
        print(pretrained_ckpt_dir)
        
        print('======= Stage I: Subtraction =======')
        if os.path.isfile(pretrained_ckpt_dir):
            state_dict_E = torch.load(pretrained_ckpt_dir)
            self.encoder.load_state_dict(state_dict_E)
            print('======= Load cached checkpoints %s' % pretrained_ckpt_dir)
        else:
            print('======= total epochs: %d ' % epochs)
            for epoch in range(1,epochs+1):
                for i, data in enumerate(self.loader):
                    self.process_input(data)
                    
                    self.update_E()
                    self.update_D()
                    
                    if i % 10 == 0:
                        self.log(epoch, i)  # output losses and thing.
                        
                print('Epoch [%d] finished' % epoch)
                # just save the latest.
                torch.save(self.encoder.state_dict(), os.path.join(
                    self.opt.checkpoints_dir, self.opt.name, 'pretrained_net_E_%d.pth' % epoch
                ))
                
class ContrasiveGenerator(SPADEGenerator):
    def __init__(self, opt):
        super().__init__(opt)
        self.encoder = LIPEncoder(opt, self.sw, self.sh, self.scale_ratio)
        
        if opt.isTrain:
            self.attach_dataloader(opt)
            self.noise_dim = opt.noise_dim
            
            self.l1_loss = nn.L1Loss()
        
            beta1, beta2 = opt.beta1, opt.beta2
            self.optimizer_E = torch.optim.Adam(
                self.encoder.parameters(), lr=opt.lr, betas=(beta1, beta2)
            )
    
    def _create_auxiliary_opt(self, opt):
        '''
            Create auxiliary options
            change necessary params
            --- dataroot
            --- dataset_mode
        '''
        aux_opt = copy.deepcopy(opt)  # just for safety
        aux_opt.dataroot = opt.dataroot_assist
        aux_opt.dataset_mode = 'assist'
        aux_opt.batchSize = 8
        aux_opt.nThreads = 4
        return aux_opt
    
    def attach_dataloader(self, opt):
        aux_opt = self._create_auxiliary_opt(opt)
        self.loader = data.create_dataloader(aux_opt)
    
    def encode(self, x):
        return self.encoder(x)
    
    def process_input(self, data):
        self.clean = data['clean'].cuda()
        self.noisy = data['noisy'].cuda()
        
    def update_E(self):
        bundle_in = torch.cat((self.clean, self.noisy), dim=0)
        bundle_out = self.encode(bundle_in)
        nb = bundle_in.shape[0] // 2
        F_real, F_fake = bundle_out[:nb], bundle_out[nb:]
        loss_l1 = self.l1_loss(F_fake, F_real)
        
        self.optimizer_E.zero_grad()
        loss_l1.backward()
        self.optimizer_E.step()
        
        self.loss_l1 = loss_l1.item()
        
    def log(self, epoch, i):
        logstring = '   Epoch [%d] iter [%d]: ' % (epoch, i)
        logstring += 'l1: %.4f ' % self.loss_l1
        print(logstring)
        
    def train_E(self, epochs):
        pretrained_ckpt_dir = os.path.join(
            self.opt.checkpoints_dir, self.opt.name,
            'pretrained_net_E_%d.pth' % epochs
        )
        print(pretrained_ckpt_dir)
        
        print('======= Stage I: Subtraction =======')
        if os.path.isfile(pretrained_ckpt_dir):
            state_dict_E = torch.load(pretrained_ckpt_dir)
            self.encoder.load_state_dict(state_dict_E)
            print('======= Load cached checkpoints %s' % pretrained_ckpt_dir)
        else:
            print('======= total epochs: %d ' % epochs)
            for epoch in range(1,epochs+1):
                for i, data in enumerate(self.loader):
                    self.process_input(data)
                    self.update_E()
                    
                    if i % 10 == 0:
                        self.log(epoch, i)  # output losses and thing.
                        
                print('Epoch [%d] finished' % epoch)
                # just save the latest.
                torch.save(self.encoder.state_dict(), os.path.join(
                    self.opt.checkpoints_dir, self.opt.name, 'pretrained_net_E_%d.pth' % epoch
                ))

