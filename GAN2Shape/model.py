# FIXME: This is a placeholder for the stylegan implementation

import torch
import torch.nn as nn
from .stylegan2 import Generator, Discriminator


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

    def init_optimizers(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass
