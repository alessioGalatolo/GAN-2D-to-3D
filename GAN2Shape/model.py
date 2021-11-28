import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from GAN2Shape.stylegan2 import Generator, Discriminator
from GAN2Shape import networks
from GAN2Shape.renderer import Renderer
from GAN2Shape.losses import PerceptualLoss, PhotometricLoss, DiscriminatorLoss
from GAN2Shape import utils


class GAN2Shape(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Networks
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
        self.prior = self.init_prior_shape("box").cuda()

        self.lighting_net = networks.LightingNet(self.image_size).cuda()
        self.viewpoint_net = networks.ViewpointNet(self.image_size).cuda()
        self.depth_net = networks.DepthNet(self.image_size).cuda()
        self.albedo_net = networks.AlbedoNet(self.image_size).cuda()

        self.offset_encoder_net = networks.OffsetEncoder(self.image_size).cuda()
        self.pspnet = networks.PSPNet(layers=50, classes=21, pretrained=False).cuda()
        pspnet_checkpoint = torch.load('checkpoints/parsing/pspnet_voc.pth')
        self.pspnet.load_state_dict(pspnet_checkpoint['state_dict'],
                                    strict=False)

        # Misc
        self.max_depth = 1.1
        self.min_depth = 0.9
        self.border_depth = 0.7*self.max_depth + 0.3*self.min_depth
        self.lam_perc = 1
        self.lam_smooth = 0.01
        self.lam_regular = 0.01
        self.xyz_rotation_range = config.get('xyz_rotation_range', 60)
        self.xy_translation_range = config.get('xy_translation_range', 0.1)
        self.z_translation_range = config.get('z_translation_range', 0.1)
        self.crop = None  # TODO
        self.truncation = config.get('truncation', 1)
        if self.truncation < 1:
            with torch.no_grad():
                self.mean_latent = self.generator.mean_latent(4096)
        else:
            self.mean_latent = None
        # Renderer
        self.renderer = Renderer(config, self.image_size, self.min_depth, self.max_depth)

    def rescale_depth(self, depth):
        return (1+depth)/2*self.max_depth + (1-depth)/2*self.min_depth

    def depth_net_forward(self, inputs):
        depth_raw = self.depth_net(inputs)
        depth_centered = depth_raw - depth_raw.view(1, 1, -1).mean(2).view(1, 1, 1, 1)
        depth = torch.tanh(depth_centered).squeeze(0)
        depth = self.rescale_depth(depth)
        return F.mse_loss(depth, self.prior.detach())

    def init_prior_shape(self, type="box"):
        with torch.no_grad():
            height, width = self.image_size, self.image_size
            center_x, center_y = int(width / 2), int(height / 2)
            if type == "box":
                box_height, box_width = int(height*0.7*0.5), int(width*0.7*0.5)
                prior = torch.zeros([1, height, width])
                prior[0,
                      center_y-box_height: center_y+box_height,
                      center_x-box_width: center_x+box_width] = 1
                return prior
            else:
                return torch.ones([1, height, width])

    def forward(self, data):
        # call the appropriate step
        getattr(self, f'forward_step{self.step}')(data)
        self.step = ((self.step + 1) % 3) + 1

    def forward_step1(self, inputs):
        b = 1
        h, w = self.image_size, self.image_size
        print('Doing step 1')

        # Depth
        depth_raw = self.depth_net(inputs)
        depth_centered = depth_raw - depth_raw.view(1, 1, -1).mean(2).view(1, 1, 1, 1)
        depth = torch.tanh(depth_centered).squeeze(0)
        depth = self.rescale_depth(depth)
        # TODO: add border clamping
        depth_border = torch.zeros(1, h, w-4).cuda()
        depth_border = F.pad(depth_border, (2, 2), mode='constant', value=1.02)
        depth = depth*(1-depth_border) + depth_border * self.border_depth
        # TODO: add flips?

        # Viewpoint
        view = self.viewpoint_net(inputs)
        # TODO: add mean and flip?
        view_trans = torch.cat([
            view[:, :3] * math.pi/180 * self.xyz_rotation_range,
            view[:, 3:5] * self.xy_translation_range,
            view[:, 5:] * self.z_translation_range], 1)
        self.renderer.set_transform_matrices(view_trans)

        # Albedo
        albedo = self.albedo_net(inputs)
        # TODO: add flips?

        # Lighting
        lighting = self.lighting_net(inputs)
        lighting_a = lighting[:, :1] / 2+0.5  # ambience term
        lighting_b = lighting[:, 1:2] / 2+0.5  # diffuse term
        lighting_dxy = lighting[:, 2:]
        lighting_d = torch.cat([lighting_dxy, torch.ones(lighting.size(0), 1).cuda()], 1)
        lighting_d = lighting_d / ((lighting_d**2).sum(1, keepdim=True))**0.5  # diffuse light direction

        # Shading
        normal = self.renderer.get_normal_from_depth(depth)
        diffuse_shading = (normal * lighting_d.view(-1, 1, 1, 3)).sum(3).clamp(min=0).unsqueeze(1)
        shading = lighting_a.view(-1, 1, 1, 1) + lighting_b.view(-1, 1, 1, 1) * diffuse_shading
        texture = (albedo/2+0.5) * shading * 2 - 1

        recon_depth = self.renderer.warp_canon_depth(depth)
        recon_normal = self.renderer.get_normal_from_depth(recon_depth)

        grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(recon_depth)
        margin = (self.max_depth - self.min_depth) / 2

        # invalid border pixels have been clamped at max_depth+margin
        recon_im_mask = (recon_depth < self.max_depth+margin).float()
        recon_im_mask = recon_im_mask.unsqueeze(1).detach()
        recon_im = F.grid_sample(texture, grid_2d_from_canon, mode='bilinear').clamp(min=-1, max=1)

        # Loss
        # TODO: we could potentially implement these losses ourselves
        loss_l1_im = utils.photometric_loss(recon_im[:b], inputs, mask=recon_im_mask[:b]) # FIXME: use our loss
        loss_perc_im = self.PerceptualLoss(recon_im[:b] * recon_im_mask[:b],
                                           inputs * recon_im_mask[:b])
        loss_perc_im = torch.mean(loss_perc_im)
        loss_smooth = utils.smooth_loss(depth) + utils.smooth_loss(diffuse_shading)
        loss_total = loss_l1_im + self.lam_perc * loss_perc_im + self.lam_smooth * loss_smooth

        return loss_total

    def forward_step2(self, data):
        print('Doing step 2')
        with torch.no_grad():
            pseudo_im, mask = self.sample_pseudo_imgs(self.batchsize)

        proj_im, offset = self.generator.invert(pseudo_im, self.truncation, self.mean_latent)
        if self.crop is not None:  # FIXME: move resize, crop
            proj_im = utils.resize(proj_im, [self.origin_size, self.origin_size])
            proj_im = utils.crop(proj_im, self.crop)
        proj_im = utils.resize(proj_im, [self.image_size, self.image_size])

        self.loss_l1 = utils.photometric_loss(proj_im, pseudo_im, mask=mask)  # FIXME: use our loss
        self.loss_rec = (self.discriminator, self.proj_im, pseudo_im, mask=mask)
        self.loss_latent_norm = torch.mean(offset ** 2)
        loss_total = self.loss_l1 + self.loss_rec + self.lam_regular * self.loss_latent_norm

        return loss_total

    def forward_step3(self, data):
        print('Doing step 3')
        ...

    def plot_predicted_depth_map(self, data, img_idx=0):
        with torch.no_grad():
            depth_raw = self.depth_net(data[img_idx].cuda())
            depth_centered = depth_raw - depth_raw.view(1, 1, -1).mean(2).view(1, 1, 1, 1)
            depth = torch.tanh(depth_centered)
            depth = self.rescale_depth(depth)[0, 0, :].cpu().numpy()
            x = np.arange(0, self.image_size, 1)
            y = np.arange(0, self.image_size, 1)
            X, Y = np.meshgrid(x, y)
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.plot_surface(X, Y, depth, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
            plt.show()

    def sample_pseudo_imgs(self, batchsize):
        b, h, w = batchsize, self.image_size, self.image_size

        # random lighting conditions
        # here we do not use self.sample_view_light, but use uniform distributions instead
        x_min, x_max, y_min, y_max, diffuse_min, diffuse_max, alpha = self.rand_light
        rand_light_dxy = torch.FloatTensor(b, 2).cuda()
        rand_light_dxy[:, 0].uniform_(x_min, x_max)
        rand_light_dxy[:, 1].uniform_(y_min, y_max)
        rand_light_d = torch.cat([rand_light_dxy, torch.ones(b, 1).cuda()], 1)
        rand_light_d = rand_light_d / ((rand_light_d**2).sum(1, keepdim=True))**0.5
        rand_diffuse_shading = (self.normal[0, None] * rand_light_d.view(-1, 1, 1, 3))\
            .sum(3).clamp(min=0).unsqueeze(1)
        rand = torch.FloatTensor(b, 1, 1, 1).cuda().uniform_(diffuse_min, diffuse_max)
        rand_diffuse = (self.light_b[0, None].view(-1, 1, 1, 1) + rand) * rand_diffuse_shading
        rand_shading = self.light_a[0, None].view(-1, 1, 1, 1) + alpha * rand + rand_diffuse
        rand_light_im = (self.albedo[0, None]/2+0.5) * rand_shading * 2 - 1

        depth = self.depth[0, None]
        if self.use_mask:
            mask = self.canon_mask.expand(b, 3, h, w)
        else:
            mask = torch.ones(b, 3, h, w).cuda()

        # random viewpoints
        rand_views = self.sample_view_light(b, 'view')
        rand_views_trans = torch.cat([
            rand_views[:, :3] * math.pi/180 * self.xyz_rotation_range,
            rand_views[:, 3:5] * self.xy_translation_range,
            rand_views[:, 5:] * self.z_translation_range], 1)
        pseudo_im, mask = self.renderer.render_given_view(rand_light_im, depth.expand(b, h, w),
                                                          view=rand_views_trans,
                                                          mask=mask, grid_sample=True)
        pseudo_im, mask = pseudo_im, mask[:, 0, None, ...]
        return pseudo_im.clamp(min=-1, max=1), mask.contiguous()

    def sample_view_light(self, num, sample_type='view'):
        samples = []
        for i in range(num):
            if sample_type == 'view':
                sample = self.view_mvn.sample()[None, :]
                sample[0, 1] *= self.view_scale
                samples.append(sample)
            else:
                samples.append(self.light_mvn.sample()[None, :])
        samples = torch.cat(samples, dim=0)
        return samples
