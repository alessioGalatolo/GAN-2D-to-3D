import math
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

from torchvision import transforms

from GAN2Shape import utils
from GAN2Shape import networks
from GAN2Shape.renderer import Renderer
from GAN2Shape.stylegan2 import Generator, Discriminator
from GAN2Shape.losses import PerceptualLoss, PhotometricLoss, DiscriminatorLoss, SmoothLoss


class GAN2Shape(nn.Module):
    def __init__(self, config, debug=False):
        super().__init__()
        self.z_dim = config.get('z_dim')
        self.debug = debug
        # Networks
        self.generator = Generator(config.get('gan_size'),
                                   self.z_dim, 8,
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
        self.collected = None
        self.prior = self.init_prior_shape("box").cuda()

        self.lighting_net = networks.LightingNet(self.image_size, self.debug).cuda()
        self.viewpoint_net = networks.ViewpointNet(self.image_size, self.debug).cuda()
        self.depth_net = networks.DepthNet(self.image_size, self.debug).cuda()
        self.albedo_net = networks.AlbedoNet(self.image_size, self.debug).cuda()        
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
        self.use_mask = config.get('use_mask', True)
        self.relative_encoding = config.get('relative_encoding', False)
        self.transformer = config.get('transformer')
        self.rand_light = config.get('rand_light', [-1, 1, -0.2, 0.8, -0.1, 0.6, -0.6])
        self.truncation = config.get('truncation', 1)
        if self.truncation < 1:
            with torch.no_grad():
                self.mean_latent = self.generator.mean_latent(4096)
        else:
            self.mean_latent = None
        # Renderer
        self.renderer = Renderer(config, self.image_size, self.min_depth, self.max_depth)

        view_mvn_path = config.get('view_mvn_path', 'checkpoints/view_light/view_mvn.pth')
        light_mvn_path = config.get('light_mvn_path', 'checkpoints/view_light/light_mvn.pth')
        view_scale = config.get('view_scale', 1)
        self.view_light_sampler = ViewLightSampler(view_mvn_path, light_mvn_path, view_scale)

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

    def forward_step1(self, images, latents, collected):
        b = 1
        h, w = self.image_size, self.image_size
        print('Doing step 1')

        # Depth
        depth_raw = self.depth_net(images)
        depth_centered = depth_raw - depth_raw.view(1, 1, -1).mean(2).view(1, 1, 1, 1)
        depth = torch.tanh(depth_centered).squeeze(0)
        depth = self.rescale_depth(depth)
        # TODO: add border clamping
        depth_border = torch.zeros(1, h, w-4).cuda()
        depth_border = F.pad(depth_border, (2, 2), mode='constant', value=1.02)
        depth = depth*(1-depth_border) + depth_border * self.border_depth
        # TODO: add flips?

        # Viewpoint
        view = self.viewpoint_net(images)
        # TODO: add mean and flip?
        view_trans = torch.cat([
            view[:, :3] * math.pi/180 * self.xyz_rotation_range,
            view[:, 3:5] * self.xy_translation_range,
            view[:, 5:] * self.z_translation_range], 1)
        self.renderer.set_transform_matrices(view_trans)

        # Albedo
        albedo = self.albedo_net(images)
        # TODO: add flips?

        # Lighting
        lighting = self.lighting_net(images)
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
        # FIXME: why is above var not used?

        grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(recon_depth)
        margin = (self.max_depth - self.min_depth) / 2

        # invalid border pixels have been clamped at max_depth+margin
        recon_im_mask = (recon_depth < self.max_depth+margin).float()
        recon_im_mask = recon_im_mask.unsqueeze(1).detach()
        recon_im = F.grid_sample(texture, grid_2d_from_canon, mode='bilinear').clamp(min=-1, max=1)

        # Loss
        loss_l1_im = PhotometricLoss()(recon_im[:b], images, mask=recon_im_mask[:b])
        loss_perc_im = PerceptualLoss()(recon_im[:b] * recon_im_mask[:b], images * recon_im_mask[:b])
        loss_perc_im = torch.mean(loss_perc_im)
        loss_smooth = SmoothLoss()(depth) + SmoothLoss()(diffuse_shading)
        loss_total = loss_l1_im + self.lam_perc * loss_perc_im + self.lam_smooth * loss_smooth

        #FIXME include use_mask bool?
        # if use_mask == false:
        # if use_mask is false set canon_mask to None
        canon_mask = None
        collected = (normal, lighting_a, lighting_b, albedo, depth, canon_mask)
        return loss_total, collected

    def forward_step2(self, images, latents, collected):
        batch_size = len(images)
        F1_d = 2  # FIXME
        print('Doing step 2')
        origin_size = images.size(0)
        # unpack collected
        # I realized we need to detach them from the computational graph in step 2
        *tensors, canon_mask = collected
        for t in tensors:
            t.detach()
        normal, light_a, light_b, albedo, depth = tensors


        with torch.no_grad():
            pseudo_im, mask = self.sample_pseudo_imgs(batch_size, normal,
                                                      light_a, light_b,
                                                      albedo, depth,
                                                      canon_mask)

            gan_im, _ = self.generator([latents], input_is_w=True,
                                       truncation_latent=self.mean_latent,
                                       truncation=self.truncation, randomize_noise=False)
            gan_im = gan_im.clamp(min=-1, max=1)
            # FIXME: add back cropping
            # gan_im = utils.resize(gan_im, [origin_size, origin_size])
            # gan_im = utils.crop(gan_im, self.crop)
            gan_im = utils.resize(gan_im, [self.image_size, self.image_size])
            center_w = self.generator.style_forward(torch.zeros(1, self.z_dim).cuda())
            center_h = self.generator.style_forward(torch.zeros(1, self.z_dim).cuda(), depth=8-F1_d)

        latent_projection = self.latent_projection(pseudo_im, gan_im, latents, center_w, center_h)
        projected_image, offset = self.generator.invert(pseudo_im,
                                                        latent_projection,
                                                        self.truncation,
                                                        self.mean_latent)
        # FIXME: add back cropping
        # projected_image = utils.resize(projected_image, [origin_size, origin_size])
        # projected_image = utils.crop(projected_image, self.crop)
        projected_image = utils.resize(projected_image, [self.image_size, self.image_size])
        self.loss_l1 = PhotometricLoss()(projected_image, pseudo_im, mask=mask)

        self.loss_rec = DiscriminatorLoss()(self.discriminator, projected_image, pseudo_im, mask=mask)
        self.loss_latent_norm = torch.mean(offset ** 2)
        loss_total = self.loss_l1 + self.loss_rec + self.lam_regular * self.loss_latent_norm

        collected = projected_image.cpu(), mask.cpu()
        return loss_total, collected

    def forward_step3(self, images, latents, collected):
        """
        What needs to happen:
        (Learning the 3D shape)

          -  include the orginial image among the samples to train on:
             samples = [original image, projected samples]

          -  for each sample I-tilde_i:

                 the networks
                     viewpoint net V
                     lighting net L
                 predict the instance-specific
                     view v-tilde_i
                     lighting l-tilde_i

                 the networks
                     depth net D
                     albedo net A
                 with input: original sample I
                 output: the shared
                     depth v-tilde
                     albedo a-tilde

                 feed all of this into the reconstruction loss,
                 whatever it may be, and optimize for the \theta's.

                 with m=batch size?
                 and \lambda_2 as the regularization coefficient.

          -  anything to return?
        """

        print('Doing step 3')

        I = images  # assuming that "images" is actually just one image at a time in the trainer.py loop
        projected_samples, masks = collected  # assuming forward_step2() returns a list of size m of images
        samples = [I, *projected_samples]  # FIXME: Unpacking will probably not work out of the box like this

        batch_size = len(samples)

        losses = []

        d = self.depth_net(I)   # D(I)
        a = self.albedo_net(I)  # A(I)

        # FIXME: Add a mask for I to the masks list for the zip to work!
        # What mask would be appropriate for the orginial image I?
        for i, mask in zip(samples, masks):
            l = self.lighting_net(i)   # L(i)
            v = self.viewpoint_net(i)  # V(i)

            # FIXME: Feed the renderer a sensible image
            # as opposed to just `l` (in case `l` is not an image)
            i_rendered, _ = self.renderer.render_given_view(
                                l,
                                d.expand(b, h, w),  # FIXME: just filling in mindlessly for now -- probably wrong
                                view=v,
                                mask=mask,
                                grid_sample=True)

            # FIXME: define and/or use the proper reconstruction_loss
            # FIXME: Make sure that what you are appending to `losses` is a number
            # or at least something summable using sum()
            losses.append(reconstruction_loss(i, i_rendered))

        loss_total = (1 / batch_size) * sum(losses) + self.lam_smooth * SmoothLoss()(d)

        # FIXME: use the optimizer correctly, or drop the line below completely in case the optim.step() function
        # would already be taking care of updating the theta parameters until the argmin converged/stoped
        # The thetas are the parameters (weights) of each of the 4 networks
        theta_D, theta_A, theta_V, theta_L = argmin(loss_total, {'params': [theta_D, theta_A, theta_V, theta_L]})

        collected = None
        return loss_total, collected

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

    def latent_projection(self, image, gan_im, latent, center_w, center_h):
        F1_d = 2  # FIXME: don't know what this is for
        offset = self.offset_encoder_net(image)
        if self.relative_encoding:
            offset = offset - self.offset_encoder_net(gan_im)
        hidden = offset + center_h
        offset = self.generator.style_forward(hidden, skip=8-F1_d) - center_w
        latent = latent + offset
        return offset, latent

    def sample_pseudo_imgs(self, batchsize, normal, light_a, light_b, albedo, depth, canon_mask=None):
        b, h, w = batchsize, self.image_size, self.image_size

        # random lighting conditions
        # here we do not use self.sample_view_light, but use uniform distributions instead
        x_min, x_max, y_min, y_max, diffuse_min, diffuse_max, alpha = self.rand_light
        rand_light_dxy = torch.FloatTensor(b, 2).cuda()
        rand_light_dxy[:, 0].uniform_(x_min, x_max)
        rand_light_dxy[:, 1].uniform_(y_min, y_max)
        rand_light_d = torch.cat([rand_light_dxy, torch.ones(b, 1).cuda()], 1)
        rand_light_d = rand_light_d / ((rand_light_d**2).sum(1, keepdim=True))**0.5
        rand_diffuse_shading = (normal[0, None] * rand_light_d.view(-1, 1, 1, 3))\
            .sum(3).clamp(min=0).unsqueeze(1)
        rand = torch.FloatTensor(b, 1, 1, 1).cuda().uniform_(diffuse_min, diffuse_max)
        rand_diffuse = (light_b[0, None].view(-1, 1, 1, 1) + rand) * rand_diffuse_shading
        rand_shading = light_a[0, None].view(-1, 1, 1, 1) + alpha * rand + rand_diffuse
        rand_light_im = (albedo[0, None]/2+0.5) * rand_shading * 2 - 1

        depth = depth[0, None]
        if canon_mask is not None:
            mask = canon_mask.expand(b, 3, h, w)
        else:
            mask = torch.ones(b, 3, h, w).cuda()

        # random viewpoints
        rand_views = self.view_light_sampler.sample(b, 'view')
        rand_views_trans = torch.cat([
            rand_views[:, :3] * math.pi/180 * self.xyz_rotation_range,
            rand_views[:, 3:5] * self.xy_translation_range,
            rand_views[:, 5:] * self.z_translation_range], 1)
        pseudo_im, mask = self.renderer.render_given_view(rand_light_im, depth.expand(b, h, w),
                                                          view=rand_views_trans,
                                                          mask=mask, grid_sample=True)
        pseudo_im, mask = pseudo_im, mask[:, 0, None, ...]
        return pseudo_im.clamp(min=-1, max=1), mask.contiguous()


class ViewLightSampler():
    def __init__(self, view_mvn_path, light_mvn_path, view_scale):
        view_mvn = torch.load(view_mvn_path)
        light_mvn = torch.load(light_mvn_path)
        self.view_mean = view_mvn['mean'].cuda()
        self.light_mean = light_mvn['mean'].cuda()
        self.view_scale = view_scale
        self.view_dist = MultivariateNormal(view_mvn['mean'].cuda(), view_mvn['cov'].cuda())
        self.light_dist = MultivariateNormal(light_mvn['mean'].cuda(), light_mvn['cov'].cuda())

    def _sample(self, sample_type):
        dist = getattr(self, f'{sample_type}_dist')
        sample = dist.sample()[None, :]
        if sample_type == 'view':
            sample[0, 1] *= self.view_scale
        return sample

    def sample(self, n=1, sample_type='view'):
        samples = []
        for _ in range(n):
            samples.append(self._sample(sample_type))
        samples = torch.cat(samples, dim=0)
        return samples
