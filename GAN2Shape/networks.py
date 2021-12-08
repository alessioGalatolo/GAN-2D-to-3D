import torch
import torch.nn as nn
import torch.nn.functional as F
from gan2shape.resnet import resnet50, resnet101, resnet152
from gan2shape.debug_grad_updates import *
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
            nn.Conv2d(nf*8, nf*16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*16, nf*16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*16, cout, kernel_size=1, stride=1, padding=0, bias=False)]
        network += [nn.Tanh()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0), -1)


class ViewpointNet(Encoder):
    def __init__(self, image_size, debug=False):
        super().__init__(cin=3, cout=6, size=image_size)
        self.debug = debug
        self.alert_func = Alert_View()

    def forward(self, x):
        out = super().forward(input=x)
        if self.debug:
            out = self.alert_func.apply(out)
        return out


class LightingNet(Encoder):
    def __init__(self, image_size, debug=False):
        super().__init__(cin=3, cout=4, size=image_size)
        self.debug = debug
        self.alert_func = Alert_Light()
    def forward(self, x):
        out = super().forward(input=x)
        if self.debug:
            out = self.alert_func.apply(out)
        return out


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
    def __init__(self, image_size, debug=False):
        super().__init__(cin=3, cout=1, size=image_size, activation=None)
        self.debug = debug
        self.alert_func = Alert_Depth()
    def forward(self, x):
        out = super().forward(input=x)
        if self.debug:
            out = self.alert_func.apply(out)
        return out


class AlbedoNet(EncoderDecoder):
    def __init__(self, image_size, debug=False):
        super().__init__(cin=3, cout=3, size=image_size, activation=nn.Tanh)
        self.debug = debug
        self.alert_func = Alert_Albedo()

    def forward(self, x):
        out = super().forward(input=x)
        if self.debug:
            out = self.alert_func.apply(out)
        return out


class ResBlock(nn.Module):
    """
    The residual block used in the GAN offset encoder E.
    See Table 8 in arxiv/2011.00844.
    """

    def __init__(self, cin, cout):
        super().__init__()
        residual_path = [
            nn.ReLU(),
            nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1)]
        identity_path = [
            nn.AvgPool2d(stride=2, kernel_size=2),
            nn.Conv2d(cin, cout, kernel_size=1, stride=1, padding=0)
        ]
        self.res_path = nn.Sequential(*residual_path)
        self.identity_path = nn.Sequential(*identity_path)

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

    def __init__(self, image_size=128, cin=3, cout=512, activation=None, debug=False):
        super().__init__()
        allowed_sizes = [64, 128]
        assert(image_size in allowed_sizes)
        nf = 16  # should this be an input param?
        self.debug = debug
        self.alert_func = Alert_OffsetEncoder()

        network_part1 = [
            nn.Conv2d(cin, 2*nf, kernel_size=4, stride=2, padding=1),  # the GAN2Shape repo had 1*nf out channels here but that isn't consistent with table 7 if nf=16
            nn.ReLU(),
            ResBlock(2*nf, 4*nf),
            ResBlock(4*nf, 8*nf),
            ResBlock(8*nf, 16*nf)]

        if image_size == 128:
            network_part2 = [
                ResBlock(16*nf, 32*nf),
                nn.Conv2d(32*nf, 64*nf, kernel_size=4, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(64*nf, cout, kernel_size=1, stride=1, padding=0)]

        elif image_size == 64:
            network_part2 = [
                nn.Conv2d(16*nf, 32*nf, kernel_size=4, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(32*nf, cout/2, kernel_size=1, stride=1, padding=0)]

        network = network_part1 + network_part2
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, x):
        # The authors do a reshape here for whatever reason
        out = self.network(x).reshape(x.size(0), -1)
        if self.debug:
            out = self.alert_func.apply(out)
        return out
        # return self.network(x)


class PPM(nn.Module):
    """
        Code author: Hengshuang Zhao
        Original repo: https://github.com/hszhao/semseg
    """

    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PSPNet(nn.Module):
    """
        Code author: Hengshuang Zhao
        Original repo: https://github.com/hszhao/semseg
    """

    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
        super(PSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion

        if layers == 50:
            resnet = resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = resnet101(pretrained=pretrained)
        else:
            resnet = resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )

    def forward(self, x, y=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x
