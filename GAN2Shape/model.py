import math
from glob import glob
import logging
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from GAN2Shape import utils
from GAN2Shape import networks
from GAN2Shape.renderer import Renderer
from GAN2Shape.stylegan2 import Generator, Discriminator
from GAN2Shape.losses import PerceptualLoss, PhotometricLoss, DiscriminatorLoss, SmoothLoss


class GAN2Shape(nn.Module):
    NETS = ['lighting', 'viewpoint', 'depth', 'albedo', 'offset_encoder']

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

        self.lighting_net = networks.LightingNet(self.image_size, self.debug).cuda()
        self.viewpoint_net = networks.ViewpointNet(self.image_size, self.debug).cuda()
        self.depth_net = networks.DepthNet(self.image_size, self.debug).cuda()
        self.albedo_net = networks.AlbedoNet(self.image_size, self.debug).cuda()
        self.offset_encoder_net = networks.OffsetEncoder(self.image_size, debug=self.debug).cuda()

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

        self.ckpt_paths = config.get('our_nets_ckpts')

        # losses
        self.smooth_loss = SmoothLoss()
        self.perceptual_loss = PerceptualLoss(model='net-lin', net='vgg',
                                              use_gpu=True,
                                              gpu_ids=[torch.device('cuda:0')])
        self.discriminator_loss = DiscriminatorLoss()
        self.photometric_loss = PhotometricLoss()

    def rescale_depth(self, depth):
        return (1+depth)/2*self.max_depth + (1-depth)/2*self.min_depth

    def depth_net_forward(self, inputs, prior):
        depth_raw = self.depth_net(inputs).squeeze(1)
        depth = depth_raw - depth_raw.view(1, 1, -1).mean(2).view(1, 1, 1, 1)
        depth = depth.tanh()
        depth = self.rescale_depth(depth)
        return F.mse_loss(depth[0], prior.detach().expand(depth.size(1), -1, -1)), depth

    def forward_step1(self, images, latents, collected, step1=True, eval=False, **kwargs):
        b = 1
        h, w = self.image_size, self.image_size
        if self.debug:
            logging.info('Doing step 1')

        # Depth
        # TODO: add flips?
        if step1:
            with torch.no_grad():
                depth_raw = self.depth_net(images)
        else:
            depth_raw = self.depth_net(images)
        depth = self.get_clamped_depth(depth_raw.squeeze(1), h, w)

        # Viewpoint
        if step1:
            with torch.no_grad():
                view = self.viewpoint_net(images)
        else:
            view = self.viewpoint_net(images)
        # Add mean
        view = view + self.view_light_sampler.view_mean.unsqueeze(0)

        view_trans = self.get_view_transformation(view)
        self.renderer.set_transform_matrices(view_trans)

        # Albedo
        albedo = self.albedo_net(images)
        # TODO: add flips?

        # Lighting
        if step1:
            with torch.no_grad():
                lighting = self.lighting_net(images)
        else:
            lighting = self.lighting_net(images)
        # Add mean
        lighting = lighting + self.view_light_sampler.light_mean.unsqueeze(0)
        lighting_a, lighting_b, lighting_d = self.get_lighting_directions(lighting)

        # Shading
        normal = self.renderer.get_normal_from_depth(depth)
        diffuse_shading, texture = self.get_shading(normal, lighting_a,
                                                    lighting_b, lighting_d, albedo)

        recon_depth = self.renderer.warp_canon_depth(depth)
        recon_normal = self.renderer.get_normal_from_depth(recon_depth)
        # FIXME: why is above var not used?

        grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(recon_depth)
        margin = (self.max_depth - self.min_depth) / 2

        # invalid border pixels have been clamped at max_depth+margin
        recon_im_mask = (recon_depth < self.max_depth+margin).float()
        recon_im_mask = recon_im_mask.unsqueeze(1).detach()
        recon_im = F.grid_sample(texture, grid_2d_from_canon, mode='bilinear').clamp(min=-1, max=1)

        # Only used at test time
        if eval:
            return recon_im, recon_depth

        # Loss
        loss_l1_im = self.photometric_loss(recon_im[:b], images, mask=recon_im_mask[:b])
        loss_perc_im = self.perceptual_loss(recon_im[:b] * recon_im_mask[:b],
                                            images * recon_im_mask[:b])
        loss_perc_im = torch.mean(loss_perc_im)
        loss_smooth = self.smooth_loss(depth) + self.smooth_loss(diffuse_shading)
        loss_total = loss_l1_im + self.lam_perc * loss_perc_im + self.lam_smooth * loss_smooth

        # FIXME include use_mask bool?
        # if use_mask == false:
        # if use_mask is false set canon_mask to None
        if len(images) == 1:
            canon_mask = None
        else:
            canon_mask = [None]*len(images)
        collected = (normal, lighting_a, lighting_b, albedo, depth, canon_mask)
        return loss_total, collected

    def forward_step2(self, image, latent, collected, n_proj_samples=8, **kwargs):
        F1_d = 2  # number of mapping network layers used to regularize the latent offset
        if self.debug:
            logging.info('Doing step 2')
        origin_size = image.size(0)
        # unpack collected
        *tensors, canon_mask = collected
        for t in tensors:
            t.detach()
        normal, light_a, light_b, albedo, depth = tensors

        with torch.no_grad():
            pseudo_im, mask = self.sample_pseudo_imgs(n_proj_samples,
                                                      normal,
                                                      light_a, light_b,
                                                      albedo, depth,
                                                      canon_mask)

            gan_im, _ = self.generator([latent], input_is_w=True,
                                       truncation_latent=self.mean_latent,
                                       truncation=self.truncation, randomize_noise=False)
            gan_im = gan_im.clamp(min=-1, max=1)
            # FIXME: add back cropping
            # gan_im = utils.resize(gan_im, [origin_size, origin_size])
            # gan_im = utils.crop(gan_im, self.crop)
            gan_im = utils.resize(gan_im, [self.image_size, self.image_size])
            center_w = self.generator.style_forward(torch.zeros(1, self.z_dim).cuda())
            center_h = self.generator.style_forward(torch.zeros(1, self.z_dim).cuda(),
                                                    depth=8-F1_d)

        latent_projection = self.latent_projection(pseudo_im, gan_im, latent,
                                                   center_w, center_h, F1_d)
        projected_image, offset = self.generator.invert(pseudo_im,
                                                        latent_projection,
                                                        self.truncation,
                                                        self.mean_latent)
        # FIXME: add back cropping
        # projected_image = utils.resize(projected_image, [origin_size, origin_size])
        # projected_image = utils.crop(projected_image, self.crop)
        projected_image = utils.resize(projected_image, [self.image_size, self.image_size])
        self.loss_l1 = self.photometric_loss(projected_image, pseudo_im, mask=mask)

        self.loss_rec = self.discriminator_loss(self.discriminator,
                                                projected_image,
                                                pseudo_im, mask=mask)
        self.loss_latent_norm = torch.mean(offset ** 2)
        loss_total = self.loss_l1 + self.loss_rec + self.lam_regular * self.loss_latent_norm
        collected = projected_image.detach().cpu(), mask.detach().cpu()
        return loss_total, collected

    def forward_step3(self, images, latents, collected, **kwargs):
        if self.debug:
            logging.info('Doing step 3')

        # --------- Extract Albedo and Depth from the original image ----------
        projected_samples, masks = collected
        permutation = torch.randperm(len(projected_samples))
        projected_samples[permutation]
        masks[permutation]
        projected_samples, masks = projected_samples.cuda(), masks.cuda()

        step1_loss, collected = self.forward_step1(images, None, None, step1=False)
        normal, _, _, albedo, depth, _ = collected

        # --------- Extract View and Light from the projected sample ----------
        b = len(projected_samples)

        # View
        view = self.viewpoint_net(projected_samples)  # V(i)
        # Add mean
        view = view + self.view_light_sampler.view_mean.unsqueeze(0)
        view_trans = self.get_view_transformation(view)
        self.renderer.set_transform_matrices(view_trans)

        # Lighting
        light = self.lighting_net(projected_samples)   # L(i)
        # Add mean
        light = light + self.view_light_sampler.light_mean.unsqueeze(0)

        light_a, light_b, light_d = self.get_lighting_directions(light)

        # Shading
        diffuse_shading, texture = self.get_shading(normal, light_a,
                                                    light_b, light_d, albedo)

        depth = depth.expand(b,
                             self.image_size,
                             self.image_size)
        recon_depth = self.renderer.warp_canon_depth(depth)
        grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(recon_depth)
        margin = (self.max_depth - self.min_depth) / 2

        # invalid border pixels have been clamped at max_depth+margin
        recon_im_mask = (recon_depth < self.max_depth+margin).float()
        recon_im_mask = recon_im_mask.unsqueeze(1).detach() * masks
        recon_im = F.grid_sample(texture, grid_2d_from_canon, mode='bilinear')\
            .clamp(min=-1, max=1)

        # Loss
        loss_l1_im = self.photometric_loss(recon_im[:b], projected_samples, mask=recon_im_mask[:b])
        loss_perc_im = self.perceptual_loss(recon_im[:b] * recon_im_mask[:b],
                                            projected_samples * recon_im_mask[:b])
        loss_perc_im = torch.mean(loss_perc_im)
        loss_total = step1_loss + loss_l1_im + self.lam_perc * loss_perc_im

        return loss_total, None

    def latent_projection(self, image, gan_im, latent, center_w, center_h, F1_d):
        offset = self.offset_encoder_net(image)
        if self.relative_encoding:
            offset = offset - self.offset_encoder_net(gan_im)
        hidden = offset + center_h
        offset = self.generator.style_forward(hidden, skip=8-F1_d) - center_w
        latent = latent + offset
        return offset, latent

    def sample_pseudo_imgs(self, n_images, normal,
                           light_a, light_b, albedo,
                           depth, canon_mask=None):
        h, w = self.image_size, self.image_size  # assume square

        # random lighting conditions
        # here we do not use self.sample_view_light, but use uniform distributions instead
        x_min, x_max, y_min, y_max, diffuse_min, diffuse_max, alpha = self.rand_light
        rand_light_dxy = torch.FloatTensor(n_images, 2).cuda()
        rand_light_dxy[:, 0].uniform_(x_min, x_max)
        rand_light_dxy[:, 1].uniform_(y_min, y_max)
        rand_light_d = torch.cat([rand_light_dxy, torch.ones(n_images, 1).cuda()], 1)
        rand_light_d = rand_light_d / ((rand_light_d**2).sum(1, keepdim=True))**0.5
        rand_diffuse_shading = (normal[0, None] * rand_light_d.view(-1, 1, 1, 3))\
            .sum(3).clamp(min=0).unsqueeze(1)
        rand = torch.FloatTensor(n_images, 1, 1, 1).cuda().uniform_(diffuse_min, diffuse_max)
        rand_diffuse = (light_b[0, None].view(-1, 1, 1, 1) + rand) * rand_diffuse_shading
        rand_shading = light_a[0, None].view(-1, 1, 1, 1) + alpha * rand + rand_diffuse
        rand_light_im = (albedo[0, None]/2+0.5) * rand_shading * 2 - 1

        depth = depth[0, None]
        if canon_mask is not None:
            mask = canon_mask.expand(n_images, 3, h, w)
        else:
            mask = torch.ones(n_images, 3, h, w).cuda()

        # random viewpoints
        rand_views = self.view_light_sampler.sample(n_images, 'view')
        rand_views_trans = torch.cat([
            rand_views[:, :3] * math.pi/180 * self.xyz_rotation_range,
            rand_views[:, 3:5] * self.xy_translation_range,
            rand_views[:, 5:] * self.z_translation_range], 1)
        pseudo_im, mask = self.renderer.render_given_view(rand_light_im,
                                                          depth.expand(n_images, h, w),
                                                          view=rand_views_trans,
                                                          mask=mask, grid_sample=True)
        pseudo_im, mask = pseudo_im, mask[:, 0, None, ...]
        return pseudo_im.clamp(min=-1, max=1), mask.contiguous()

    def get_view_transformation(self, view):
        view_trans = torch.cat([
            view[:, :3] * math.pi/180 * self.xyz_rotation_range,
            view[:, 3:5] * self.xy_translation_range,
            view[:, 5:] * self.z_translation_range], 1)
        return view_trans

    def get_clamped_depth(self, depth_raw, h, w, clamp_border=True):
        depth_centered = depth_raw - depth_raw.view(1, -1).mean(1).view(1, 1, 1)
        depth = torch.tanh(depth_centered)
        depth = self.rescale_depth(depth)
        if clamp_border:
            depth_border = torch.zeros(1, h, w-4).cuda()
            depth_border = F.pad(depth_border, (2, 2), mode='constant', value=1.02)
            depth = depth*(1-depth_border) + depth_border * self.border_depth
        return depth

    def get_lighting_directions(self, lighting):
        lighting_a = lighting[:, :1] / 2+0.5  # ambience term
        lighting_b = lighting[:, 1:2] / 2+0.5  # diffuse term
        lighting_dxy = lighting[:, 2:]
        lighting_d = torch.cat([lighting_dxy, torch.ones(lighting.size(0), 1).cuda()], 1)
        lighting_d = lighting_d / ((lighting_d**2).sum(1, keepdim=True))**0.5
        return lighting_a, lighting_b, lighting_d

    def get_shading(self, normal, lighting_a, lighting_b, lighting_d, albedo):
        diffuse_shading = (normal * lighting_d.view(-1, 1, 1, 3)).sum(3).clamp(min=0).unsqueeze(1)
        shading = lighting_a.view(-1, 1, 1, 1) + lighting_b.view(-1, 1, 1, 1) * diffuse_shading
        texture = (albedo/2+0.5) * shading * 2 - 1

        return diffuse_shading, texture

    def evaluate_results(self, image):
        with torch.no_grad():
            recon_im, recon_depth = self.forward_step1(image, None, None, eval=True)
            depth_raw = self.depth_net(image).squeeze(1)
            recon_depth = self.get_clamped_depth(depth_raw, self.image_size,
                                                 self.image_size, clamp_border=False)
        return recon_im, recon_depth

    def reset_params(self, net):
        for layers in net.children():
            for layer in layers:
                if hasattr(layer, 'reset_parameters'):
                    logging.info("Resetting layer")
                    layer.reset_parameters()

    def reinitialize_model(self):
        logging.warning(">>>RESETTING ALL WEIGHTS<<<")
        self.reset_params(self.lighting_net)
        self.reset_params(self.viewpoint_net)
        self.reset_params(self.depth_net)
        self.reset_params(self.albedo_net)
        self.reset_params(self.offset_encoder_net)

    def save_checkpoint(self, img_idx, stage, total_it, category='car'):
        try:
            nets = GAN2Shape.NETS
            now = datetime.datetime.now()
            now = now.strftime("%Y_%m_%d_%H_%M")  # descending order for sorting
            for net in nets:
                save_dict = {'total_it': total_it,
                             'dataset': category,
                             'model_state_dict': getattr(self, f'{net}_net').state_dict()}

                # full path
                filename = self.build_checkpoint_path(self.ckpt_paths['VLADE_nets'],
                                                      category, net, img_idx, stage,
                                                      total_it, now)
                # path without filename
                save_path = filename.rsplit('/', maxsplit=1)[0]
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                with open(filename, 'wb') as f:
                    torch.save(save_dict, f)
        except Exception as e:
            logging.error("Error: ", e)
            logging.error(">>>Saving failed... continuing training<<<")

    def load_from_checkpoints(self, path_base, category):
        paths, indices = self.build_checkpoint_path(path_base, category)
        for path, img_idx in zip(paths, indices):
            self.load_from_checkpoint(path)
            yield img_idx

    def load_from_checkpoint(self, filename_path):
        nets = GAN2Shape.NETS
        device = torch.device('cuda')
        for net in nets:
            filename = filename_path(net)
            with open(filename, 'rb') as f:
                checkpoint = torch.load(f, map_location=device)
            getattr(self, f'{net}_net').load_state_dict(checkpoint['model_state_dict'])

    def build_checkpoint_path(self, base, category, net=None,
                              img_idx="*", stage="*", total_it="*", time="*", general=False):
        if net is not None:
            return f'{base}/{category}/{net}_image_{img_idx}_stage_{stage}_{total_it}_it_{time}.pth'
        net = GAN2Shape.NETS[0]
        path = f'{base}/{category}/{net}_image_*_stage_*_*_it_*.pth'
        # look for checkpoints
        possible_paths = glob(path)
        assert possible_paths
        paths = []
        img_ids = []
        for path in possible_paths:
            splits = path.split(net)
            beginning = splits[0]
            end = splits[1]
            paths.append(lambda x, b=beginning, e=end: f'{b}{x}{e}')
            words = path.split('_')
            if not general:
                image_index = words.index('image')
                img_ids.append(int(words[image_index+1]))
        return paths, img_ids


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


class MaskingModel():
    CATEGORIES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                  'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                  'horse', 'motorbike', 'person', 'pottedplant',
                  'sheep', 'sofa', 'train', 'tvmonitor']  # FIXME: add face
    CATEGORY2NUMBER = {category: i+1 for i, category in enumerate(CATEGORIES)}

    def __init__(self, category):
        # parsing/masking model
        self.category = category
        if category == 'face':
            self.mask_net = networks.BiSeNet(n_classes=19)
            self.mask_net.load_state_dict(torch.load('checkpoints/parsing/bisenet.pth'))
        else:
            self.mask_net = networks.PSPNet(layers=50, classes=21, pretrained=False)
            temp = torch.nn.DataParallel(self.mask_net)
            checkpoint = torch.load('checkpoints/parsing/pspnet_voc.pth')
            temp.load_state_dict(checkpoint['state_dict'], strict=False)
            self.mask_net = temp.module
        self.mask_net = self.mask_net.cuda()
        self.mask_net.eval()

    def confidence_mask(self, image, depth=None):
        with torch.no_grad():
            # # FIXME: only if car, cat
            # image = image / 2 + 0.5
            # image[:, 0].sub_(0.485).div_(0.229)
            # image[:, 1].sub_(0.456).div_(0.224)
            # image[:, 2].sub_(0.406).div_(0.225)
            image_size = image.shape[-1]
            size = 512 if self.category == 'face' else 473
            image = utils.resize(image, [size, size])
            out = self.mask_net(image)
            if self.category in MaskingModel.CATEGORIES:
                category_number = MaskingModel.CATEGORY2NUMBER[self.category]
                mask = out[:, category_number: category_number+1, ...]
            elif self.category == 'face':
                mask = torch.sum(out[:, 1:13, ...], dim=1, keepdim=True)
            else:
                out = out[:, 0:1, ...]
                mask = torch.ones(out.size())

            mask -= mask.min().item()
            mask /= mask.max().item()
            return utils.resize(mask, [image_size, image_size])

    def image_mask(self, image, depth=None):
        with torch.no_grad():
            # # FIXME: only if car, cat
            # image = image / 2 + 0.5
            # image[:, 0].sub_(0.485).div_(0.229)
            # image[:, 1].sub_(0.456).div_(0.224)
            # image[:, 2].sub_(0.406).div_(0.225)
            image_size = image.shape[-1]
            size = 512 if self.category == 'face' else 473
            image = utils.resize(image, [size, size])
            out = self.mask_net(image)
            if self.category in MaskingModel.CATEGORIES:
                out = out.argmax(dim=1, keepdim=True)
                mask = (out == MaskingModel.CATEGORY2NUMBER[self.category])
            elif self.category == 'face':
                out = torch.cat((out[:, :17, ...], out[:, 18:, ...]), dim=1)
                out = out.argmax(dim=1, keepdim=True)
                mask_all = ((out >= 1) == (out != 16))
                mask_face = ((out >= 1) == (out <= 13))
                mask = (mask_all & mask_face)
            else:
                mask = torch.ones(out.size(), dtype=torch.bool)

            if not torch.any(mask):
                logging.warning(f'Did not find any {self.category} in image {image}')
                mask = torch.ones(out.size(), dtype=torch.bool)
        if depth is not None:
            depth = utils.resize(depth, [size, size])
            depth[~(mask[0])] = np.nan
            mask = depth
        mask = mask.float()
        mask = utils.resize(mask, [image_size, image_size])
        return mask
