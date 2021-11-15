import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import activation


"""
Network dependencies for the networks in arxiv/2011.00844.

    V: ViewpointNet -> Encoder -> nn.Module
    L: LightingNet -> Encoder -> nn.Module

    D: DepthNet -> EncoderDecoder -> nn.Module
    A: AlbedoNet -> EncoderDecoder -> nn.Module

    E: OffsetEncoder -> {nn.Module, ResBlock}

    ResBlock -> nn.Module
"""


class Encoder(nn.Module):
    """
    The Encoder used in ViewpointNet and LightingNet.
    See Table 5 in arxiv/2011.00844.
    """

    def __init__(self, cin, cout, size):
        super().__init__()
        nf = max(4096 // size, 16)
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, cout, kernel_size=1, stride=1, padding=0, bias=False)]
        network += [nn.Tanh()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0), -1)


class ViewpointNet(Encoder):
    def __init__(self, image_size):
        super().__init__(cin=3, cout=6, size=image_size)

    def forward(self, x):
        super().forward(input=x)


class LightingNet(Encoder):
    def __init__(self, image_size):
        super().__init__(cin=3, cout=4, size=image_size)

    def forward(self, x):
        super().forward(input=x)


class EncoderDecoder(nn.Module):
    """
    The EncoderDecoder used in DepthNet and AlbedoNet.
    See Table 6 in arxiv/2011.00844.
    """

    def __init__(self, cin, cout, size, activation, zdim=256):
        super().__init__()
        nf = max(4096 // size, 16)
        gn_base = 8 if size >= 128 else 16
        # downsampling
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(gn_base, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(gn_base*2, nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(gn_base*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)]
        # upsampling
        network += [
            nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(gn_base*4, nf*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(gn_base*4, nf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(gn_base*2, nf*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(gn_base*2, nf*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(gn_base, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(gn_base, nf),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(gn_base, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(gn_base, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, cout, kernel_size=5, stride=1, padding=2, bias=False)]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input)


class DepthNet(EncoderDecoder):
    def __init__(self, image_size):
        super().__init__(cin=3, cout=1, size=image_size, activation=None)

    def forward(self, x):
        super().forward(input=x)


class AlbedoNet(EncoderDecoder):
    def __init__(self, image_size):
        super().__init__(cin=3, cout=3, size=image_size, activation=nn.Tanh)

    def forward(self, x):
        super().forward(input=x)


class ResBlock(nn.Module):
    """
    The residual block used in the GAN offset encoder E.
    See Table 8 in arxiv/2011.00844.
    """

    def __init__(self, cin, cout):
        super().__init__()
        residual_path=[
            nn.ReLU(),
            nn.Conv2d(cin,cout, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(cout,cout, kernel_size=3, stride=1, padding=1)]
        identity_path=[
            nn.AvgPool2d(stride=2),
            nn.Conv2d(cin,cout, kernel_size=1, stride=1, padding=0)
        ]
        self.res_path=nn.Sequential(*residual_path)
        self.identity_path=nn.Sequential(*identity_path)
        

    def forward(self, x):
        x_identity = self.identity_path(x)
        x_res = self.res_path(x)
        out = x_identity + x_res
        return out


class OffsetEncoder(nn.Module):
    """
    The GAN offset encoder E.
    See Table 7 in arxiv/2011.00844.
    """

    def __init__(self, image_size=128, cin=3, cout=512, activation=None):
        super().__init__()
        allowed_sizes=[64,128]
        assert(image_size in allowed_sizes)
        nf=16 #should this be an input param?

        network_part1=[
            nn.Conv2d(cin, 2*nf, kernel_size=4, stride=2, padding=1), #the GAN2Shape repo had 1*nf out channels here but that isn't consistent with table 7 if nf=16
            nn.ReLU(),
            ResBlock(2*nf,4*nf),
            ResBlock(4*nf,8*nf),
            ResBlock(8*nf,16*nf)]

        if image_size==128:
            network_part2=[
                ResBlock(16*nf,32*nf),
                nn.Conv2d(32*nf,64*nf, kernel_size=4, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(64*nf,cout, kernel_size=1, stride=1, padding=0)]

        elif image_size==64:
            network_part2=[
                nn.Conv2d(16*nf,32*nf, kernel_size=4, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(32*nf,cout/2, kernel_size=1, stride=1, padding=0)]

        network = network_part1 + network_part2
        if activation is not None:
            network+=[activation()]
        self.network=nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)
        
