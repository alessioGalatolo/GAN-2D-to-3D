import torch
import torch.nn as nn
from GAN2Shape.stylegan2 import Generator, Discriminator
from GAN2Shape import networks
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

class GAN2Shape(nn.Module):
    def __init__(self, config):
        super().__init__()

        ## Networks
        self.generator = Generator(config.get('gan_size'),
                                   config.get('z_dim'), 8,
                                   channel_multiplier=config.get('channel_multiplier'))
        self.discriminator = Discriminator(config.get('gan_size'),
                                           channel_multiplier=config.get('channel_multiplier'))
        gan_ckpt = torch.load(config.get('gan_ckpt_path'))
        self.generator.load_state_dict(gan_ckpt['g_ema'], strict=False)
        self.generator = self.generator.cuda()
        self.generator.eval()
        self.discriminator.load_state_dict(gan_ckpt['d'], strict=False)
        self.discriminator = self.discriminator.cuda()
        self.discriminator.eval()

        self.image_size = config.get('image_size')
        self.step = 1
        self.prior = self.init_prior_shape("box")

        self.lighting_net = networks.LightingNet(self.image_size).cuda()
        self.viewpoint_net = networks.ViewpointNet(self.image_size).cuda()
        self.depth_net = networks.DepthNet(self.image_size).cuda()
        self.albedo_net = networks.AlbedoNet(self.image_size).cuda()

        self.offset_encoder_net = networks.OffsetEncoder(self.image_size).cuda()
        self.pspnet = networks.PSPNet(layers=50, classes=21, pretrained=False).cuda()
        pspnet_checkpoint = torch.load('checkpoints/parsing/pspnet_voc.pth')
        self.pspnet.load_state_dict(pspnet_checkpoint['state_dict'],
                                    strict=False)
        ## Misc
        self.max_depth=1.1
        self.min_depth=0.9
        self.depth_rescaler = lambda d: (1+d)/2 *self.max_depth + (1-d)/2 *self.min_depth

    def init_optimizers(self):
        pass

    def init_prior_shape(self, type="box"):
        with torch.no_grad():
            height, width = self.image_size, self.image_size
            center_x, center_y = int(width / 2), int(height / 2)
            if type=="box":        
                box_height, box_width = int(height*0.7*0.5), int(width*0.7*0.5)
                prior = torch.zeros([1,height,width])
                prior[0, center_y - box_height:center_y+box_height,center_x-box_width:center_x+box_width] = 1
                return prior
            else:
                return torch.ones([1,height,width])

    def forward(self, data):
        # call the appropriate step
        getattr(self, f'forward_step{self.step}')(data)
        self.step = ((self.step + 1) % 3) + 1

    def forward_step1(self, data_batch):
        print('Doing step 1')
        depth_raw = self.model.depth_net(data_batch)
        depth_centered = depth_raw - depth_raw.view(1,1,-1).mean(2).view(1,1,1,1)
        depth = torch.tanh(depth_centered).squeeze(0)



    def forward_step2(self, data):
        print('Doing step 2')
        pass

    def forward_step3(self, data):
        print('Doing step 3')
        pass

    def backward(self):
        pass

    def plot_predicted_depth_map(self, data, device, img_idx=0):
        depth_raw = self.depth_net(data[img_idx].to(device))
        depth_centered = depth_raw - depth_raw.view(1,1,-1).mean(2).view(1,1,1,1)
        depth = torch.tanh(depth_centered)
        depth = self.depth_rescaler(depth)[0,0,:].cpu().numpy()
        x = np.arange(0, self.image_size, 1)
        y = np.arange(0, self.image_size, 1)
        X, Y = np.meshgrid(x, y)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(X, Y, depth, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        plt.show()



