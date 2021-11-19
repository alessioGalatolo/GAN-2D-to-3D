import torch
import torch.nn as nn
from .stylegan2 import Generator, Discriminator
from GAN2Shape import networks


class GAN2Shape(nn.Module):
    def __init__(self, config):
        super().__init__()
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

        self.image_size = 128

        self.lighting_net = networks.LightingNet(self.image_size)
        self.viewpoint_net = networks.ViewpointNet(self.image_size)
        self.depth_net = networks.DepthNet(self.image_size)
        self.albedo_net = networks.AlbedoNet(self.image_size)

        self.offset_encoder_net = networks.OffsetEncoder(self.image_size)

    def init_optimizers(self):
        pass

    def forward(self, data):
        # call the appropriate step
        getattr(self, f'forward_step{self.step}')(data)
        self.step = (self.step + 1 % 3) + 1

    def forward_step1(self, data):
        pass

    def forward_step2(self, data):
        pass

    def forward_step3(self, data):
        pass

    def backward(self):
        pass
