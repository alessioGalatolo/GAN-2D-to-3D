import torch
import torch.nn as nn
from .stylegan2 import Generator, Discriminator
import networks


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

        self.lighting_net = networks.LightingNet(self.image_size)
        self.viewpoint_net = networks.ViewpointNet(self.image_size)
        self.depth_net = networks.DepthNet(self.image_size)
        self.albedo_net = networks.AlbedoNet(self.image_size)

    def init_optimizers(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass
