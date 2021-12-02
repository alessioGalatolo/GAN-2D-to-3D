from GAN2Shape.stylegan2 import PerceptualLoss
import torch
import torch.nn as nn

# TODO: decide on whether to have a class for the losses or a function

class DiscriminatorLoss(object):
    
    def __init__(self, ftr_num=None, data_parallel=False):
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
                mask0 = nn.functional.avg_pool2d(mask, kernel_size=(sh,sw), stride=(sh,sw)).expand_as(loss)
                loss = (loss * mask0).sum() / mask0.sum()
            else:
                loss = loss.mean()
            losses += [loss]
        return sum(losses)

    def set_ftr_num(self, ftr_num):
        self.ftr_num = ftr_num

# class DiscriminatorLoss():
#     ...


class PhotometricLoss():
    ...


class SmoothLoss():
    ...
