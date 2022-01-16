import math
import torch
from GAN2Shape import utils
from GAN2Shape.model import MaskingModel


class PriorGenerator():
    def __init__(self, image_size, category, prior,
                 noise_threshold=0.7, near=0.91, far=1.02):
        self.image_size = image_size
        self.category = category
        self.prior = prior
        if not hasattr(self, f'_{prior}_prior'):
            raise NotImplementedError()
        self.noise_threshold = noise_threshold
        self.near = near
        self.far = far
        self.base_prior = torch.Tensor(1, self.image_size, self.image_size).fill_(far)
        self.masking_model = MaskingModel(self.category)

    def __call__(self, image, device='cuda', *args, **kwargs):
        with torch.no_grad():
            prior = getattr(self, f'_{self.prior}_prior')(image, *args, **kwargs)
            return prior.to(device)

    def _box_prior(self, _):
        center_x, center_y = int(self.image_size / 2), int(self.image_size / 2)
        box_height, box_width = int(self.image_size*0.5*0.5), int(self.image_size*0.8*0.5)
        prior = torch.zeros([1, self.image_size, self.image_size])
        prior[0,
              center_x-box_width: center_x+box_width,
              center_y-box_height: center_y+box_height] = 1
        return prior

    def _masked_box_prior(self, image):
        # same as box but only project object
        mask = self.masking_model.image_mask(image)[0].cpu()

        # cut noise in mask
        noise = mask < self.noise_threshold
        mask[noise] = 0
        mask = (mask - self.noise_threshold) / (1 - self.noise_threshold)

        prior = self.far - self.base_prior * mask
        return prior

    def _smoothed_box_prior(self, image):
        # Smoothed masked_box
        prior = self._masked_box_prior(image)

        # Smoothing through repeated convolution
        kernel_size = 11
        pad = 5
        n_convs = 3
        conv = torch.nn.Conv2d(in_channels=1, out_channels=1,
                               kernel_size=kernel_size, stride=1,
                               padding=0)
        filt = torch.ones(1, 1, kernel_size, kernel_size)
        filt = filt / torch.norm(filt)
        conv.weight = torch.nn.Parameter(filt)
        prior = prior.unsqueeze(0)
        for i in range(n_convs):
            prior = conv(prior)
            # Rescale depth values to appropriate range
            prior = self.near + ((prior - torch.min(prior))*(self.far - self.near))\
                / (torch.max(prior) - torch.min(prior))
            # Pad result with 'far' to keep the image size
            prior = torch.nn.functional.pad(prior, tuple([pad]*4), value=self.far)

        return prior.squeeze(0)

    def _ellipsoid_prior(self, image):
        radius = 0.4
        mask = self.masking_model.image_mask(image)[0, 0] >= self.noise_threshold
        max_y, min_y, max_x, min_x = utils.get_mask_range(mask)

        # if self.category in ['car', 'church']:
        #     max_y = max_y + (max_y - min_y) / 6

        r_pixel = (max_x - min_x) / 2
        ratio = (max_y - min_y) / (max_x - min_x)
        c_x = (max_x + min_x) / 2
        c_y = (max_y + min_y) / 2

        i, j = torch.meshgrid(torch.linspace(0, self.image_size-1, self.image_size),
                              torch.linspace(0, self.image_size-1, self.image_size))
        i = (i - self.image_size/2) / ratio + self.image_size/2
        temp = math.sqrt(radius**2 - (radius - (self.far - self.near))**2)
        dist = torch.sqrt((i - c_y)**2 + (j - c_x)**2)
        area = dist <= r_pixel
        dist_rescale = dist / r_pixel * temp
        depth = radius - torch.sqrt(torch.abs(radius ** 2 - dist_rescale ** 2)) + self.near
        prior = torch.clone(self.base_prior)
        prior[0, area] = depth[area]
        return prior
