import argparse
import yaml
from torchvision import transforms
from torch import cuda
from gan2shape.model import GAN2Shape
from gan2shape.dataset import ImageDataset, LatentDataset
from plotting import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate GAN 2D to 3D shape')
    # parser.add_argument('--ckpt_path',
    #                     dest='CKPT_PATH',
    #                     help='path of the saved weights')
    parser.add_argument('--config-file',
                        dest='CONFIG',
                        default='config.yml',
                        help='path of the config yaml file')
    args = parser.parse_args()
    # read configuration
    with open(args.CONFIG, 'r') as config_file:
        config = yaml.safe_load(config_file)
    if not cuda.is_available():
        print("A CUDA-enables GPU is required to run this model")
        exit(1)

    transform = transforms.Compose(
            [
                transforms.Resize(config.get('image_size')),
                transforms.ToTensor()
            ]
        )

    config['transform'] = transform
    images = ImageDataset(config.get('root_path'), transform=transform)
    latents = LatentDataset(config.get('root_path'))
    model = GAN2Shape(config)

    category = config.get('category')
    base_path = config.get('our_nets_ckpts')['VLADE_nets']
    stage = config.get('stage', '*')
    iteration = config.get('iteration', '*')
    time = config.get('time', '*')

    img_idx = 2
    model.load_from_checkpoint(base_path, category, stage, iteration, time)
    recon_im, recon_depth = model.evaluate_results(images[img_idx].cuda())
    recon_im, recon_depth = recon_im.cpu(), recon_depth.cpu()
    # plot_originals(images[plot_index], block=True)
    # plot_reconstructions(recon_im, recon_depth, block=True)
    plotly_3d_depth(recon_depth, recon_im)
