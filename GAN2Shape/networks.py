import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, cin, cout, size, nf):
        super().__init__()
        # TODO

    def forward(self, input):
        # TODO
        pass


class ViewpointNet(Encoder):
    def __init__(self):
        super().__init__()
        # TODO

    def forward(self, x):
        # TODO
        pass


class LightingNet(Encoder):
    def __init__(self):
        super().__init__()
        # TODO

    def forward(self, x):
        # TODO
        pass


class EncoderDecoder(nn.Module):
    """
    The EncoderDecoder used in DepthNet and AlbedoNet.
    See Table 6 in arxiv/2011.00844.
    """

    def __init__(self, cin, cout, k, s, p):
        super().__init__()
        # TODO

    def forward(self, input):
        # TODO
        pass


class DepthNet(EncoderDecoder):
    def __init__(self):
        super().__init__()
        # TODO

    def forward(self, x):
        # TODO
        pass


class AlbedoNet(EncoderDecoder):
    def __init__(self):
        super().__init__()
        # TODO

    def forward(self, x):
        # TODO
        pass


class ResBlock(nn.Module):
    """
    The residual block used in the GAN offset encoder E.
    See Table 8 in arxiv/2011.00844.
    """

    def __init__(self):
        super().__init__()
        # TODO

    def forward(self, x):
        # TODO
        pass


class OffsetEncoder(nn.Module):
    """
    The GAN offset encoder E.
    See Table 7 in arxiv/2011.00844.
    """

    def __init__(self):
        super().__init__()
        # TODO

    def forward(self, x):
        # TODO
        pass
