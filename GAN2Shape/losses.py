from GAN2Shape.stylegan2 import PerceptualLoss
import torch
import torch.nn as nn


class DiscriminatorLoss():
    def __init__(self, ftr_num=4, data_parallel=False):
        self.data_parallel = data_parallel
        self.ftr_num = ftr_num

    def __call__(self, D, fake_img, real_img, mask=None):
        if self.data_parallel:
            with torch.no_grad():
                d, real_feature = nn.parallel.data_parallel(
                    D, real_img.detach(), self.ftr_num)
            d, fake_feature = nn.parallel.data_parallel(D, fake_img, self.ftr_num)
        else:
            with torch.no_grad():
                d, real_feature = D(real_img.detach(), self.ftr_num)
            d, fake_feature = D(fake_img, self.ftr_num)
        losses = []
        ftr_num = self.ftr_num if self.ftr_num is not None else len(fake_feature)
        for i in range(ftr_num):
            loss = torch.abs(fake_feature[i] - real_feature[i])
            if mask is not None:
                b, c, h, w = loss.shape
                _, _, hm, wm = mask.shape
                sh, sw = hm//h, wm//w
                mask0 = nn.functional.avg_pool2d(mask,
                                                 kernel_size=(sh, sw),
                                                 stride=(sh, sw)).expand_as(loss)
                loss = (loss * mask0).sum() / mask0.sum()
            else:
                loss = loss.mean()
            losses += [loss]
        return sum(losses)


class PhotometricLoss():
    EPS = 1e-7

    def __call__(self, image1, image2, mask=None, conf_sigma=None):
        loss = (image1 - image2).abs()
        if conf_sigma is not None:
            loss = loss * 2**0.5 / (conf_sigma + self.EPS) + (conf_sigma + self.EPS).log()
        if mask is not None:
            mask = mask.expand_as(loss)
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss


class SmoothLoss():

    def __call__(self, pred_map):
        if type(pred_map) not in [tuple, list]:
            pred_map = [pred_map]

        loss = 0
        weight = 1

        for scaled_map in pred_map:
            dx, dy = self.gradient(scaled_map)
            dx2, dxdy = self.gradient(dx)
            dydx, dy2 = self.gradient(dy)
            loss += (dx2.abs().mean()
                     + dxdy.abs().mean()
                     + dydx.abs().mean()
                     + dy2.abs().mean()) * weight
            weight /= 2.3
        return loss

    def gradient(self, pred):
        if pred.dim() == 4:
            pred = pred.reshape(-1, pred.size(2), pred.size(3))
        D_dy = pred[:, 1:] - pred[:, :-1]
        D_dx = pred[:, :, 1:] - pred[:, :, :-1]
        return D_dx, D_dy
