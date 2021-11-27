import math
from tqdm import tqdm
import torch
import torch.nn as nn
from GAN2Shape.stylegan2 import Generator, Discriminator
from GAN2Shape import networks
from GAN2Shape.renderer import Renderer
import GAN2Shape.utils as utils
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


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

        # Renderer
        self.renderer = Renderer(config, self.image_size, self.min_depth, self.max_depth)

    def rescale_depth(self, depth):
        return (1+depth)/2*self.max_depth + (1-depth)/2*self.min_depth

    def pretrain_depth_net(self, data, plot_example=None):
        depth_net_params = filter(lambda p: p.requires_grad,
                                  self.depth_net.parameters())
        optim = torch.optim.Adam(depth_net_params, lr=0.0001,
                                 betas=(0.9, 0.999), weight_decay=5e-4)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim,T_0=10,eta_min=0.0001)
        train_loss = []
        print("Pretraining depth net on prior shape")
        iterator = tqdm(range(len(data)))
        for i in iterator:
            data_batch = data[i]
            inputs = data_batch.cuda()
            depth_raw = self.depth_net(inputs)
            depth_centered = depth_raw - depth_raw.view(1, 1, -1).mean(2).view(1, 1, 1, 1)
            depth = torch.tanh(depth_centered).squeeze(0)
            depth = self.rescale_depth(depth)
            loss = F.mse_loss(depth, self.prior.detach())
            optim.zero_grad()
            loss.backward()
            optim.step()
            if i % 10 == 0:
                with torch.no_grad():
                    iterator.set_description("Loss = " + str(loss.cpu()))
                    train_loss.append(loss.cpu())
            # scheduler.step()
        # plt.plot(train_loss)
        # plt.title("Pretrain prior - loss / 10 steps")
        # plt.show()
        if plot_example is not None:
            with torch.no_grad():
                self.plot_predicted_depth_map(data, img_idx=plot_example)
        return train_loss

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

    def forward_step1(self, data_batch):
        return
        b = 1
        h, w = self.image_size, self.image_size
        print('Doing step 1')

        # Depth
        depth_raw = self.depth_net(inputs)
        depth_centered = depth_raw - depth_raw.view(1, 1, -1).mean(2).view(1, 1, 1, 1)
        depth = torch.tanh(depth_centered).squeeze(0)
        depth = self.model.depth_rescaler(depth)
        # TODO: add border clamping
        depth_border = torch.zeros(1, h, w-4).cuda()
        depth_border = F.pad(depth_border, (2, 2), mode='constant', value=1.02)
        depth = self.depth*(1-depth_border) + depth_border * self.border_depth
        # TODO: add flips?

        # Viewpoint
        view = self.viewpoint_net(data_batch)
        # TODO: add mean and flip?
        view_trans = torch.cat([
            view[:, :3] * math.pi/180 * self.xyz_rotation_range,
            view[:, 3:5] * self.xy_translation_range,
            view[:, 5:] * self.z_translation_range], 1)
        self.renderer.set_transform_matrices(view_trans)

        # Albedo
        albedo = self.albedo_net(data_batch)
        # TODO: add flips?

        # Lighting
        lighting = self.lighting_net(data_batch)
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
        loss_l1_im = utils.photometric_loss(recon_im[:b], data_batch, mask=recon_im_mask[:b])
        loss_perc_im = self.PerceptualLoss(recon_im[:b] * recon_im_mask[:b],
                                           data_batch * recon_im_mask[:b])
        loss_perc_im = torch.mean(loss_perc_im)
        loss_smooth = utils.smooth_loss(depth) + utils.smooth_loss(diffuse_shading)
        loss_total = loss_l1_im + self.lam_perc * loss_perc_im + self.lam_smooth * loss_smooth

        return loss_total

    def forward_step2(self, data):
        print('Doing step 2')
        pass

    def forward_step3(self, data):
        print('Doing step 3')
        pass

    def backward(self):
        pass

    def plot_predicted_depth_map(self, data, img_idx=0):
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
