import torch
import torch.nn.functional as F
from os import makedirs


def create_results_folder():
    makedirs('results/plots', exist_ok=True)
    makedirs('results/htmls', exist_ok=True)
    # Add more if necessary


def resize(image, size):
    dim = image.dim()
    if dim == 3:
        image = image.unsqueeze(1)
    b, _, h, w = image.shape
    if size[0] > h:
        image = F.interpolate(image, size, mode='bilinear')
    elif size[0] < h:
        image = F.interpolate(image, size, mode='area')
    if dim == 3:
        image = image.squeeze(1)
    return image


def crop(tensor, crop_size):
    size = tensor.size(2)   # assume h=w
    margin = (size - crop_size) // 2
    tensor = tensor[:, :, margin:margin+crop_size, margin:margin+crop_size]
    return tensor


def get_mask_range(mask):
    h_range = torch.arange(0, mask.size(0))
    w_range = torch.arange(0, mask.size(1))
    grid = torch.stack(torch.meshgrid([h_range, w_range]), 0).float()
    max_y = torch.max(grid[0, mask])
    min_y = torch.min(grid[0, mask])
    max_x = torch.max(grid[1, mask])
    min_x = torch.min(grid[1, mask])
    return max_y, min_y, max_x, min_x
