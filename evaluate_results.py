import argparse
import yaml
from torchvision import transforms
from torch import cuda
from gan2shape.model import GAN2Shape
from gan2shape.dataset import ImageDataset, LatentDataset
from gan2shape.trainer import Trainer
from gan2shape import utils
from plotting import *
import torch


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

    

    img_idx = 0
    model.load_from_checkpoint(base_path, category, stage, iteration, time)
    img1 = images[img_idx].unsqueeze(0)

    recon_im, recon_depth = model.evaluate_results(img1.cuda())
    recon_im, recon_depth = recon_im.cpu(), recon_depth.cpu()
    # plot_originals(images[img_idx], block=True)
    # plot_reconstructions(recon_im, recon_depth, block=True)

    trainer = Trainer(model = GAN2Shape, model_config=config)
    size = 473
    img = utils.resize(img1, [size, size])
    # FIXME: only if car, cat
    img = img / 2 + 0.5
    img[:, 0].sub_(0.485).div_(0.229)
    img[:, 1].sub_(0.456).div_(0.224)
    img[:, 2].sub_(0.406).div_(0.225)
    model.mask_net = model.mask_net.cuda()
    out = model.mask_net(img.cuda())
    out = out.argmax(dim=1, keepdim=True)
    mask = out.float()
    mask =  utils.resize(mask, [img1.shape[-1],img1.shape[-1]])

    recon_depth[0,mask[0,0]!=7] = np.NaN
    plotly_3d_depth(recon_depth, recon_im, img_idx=img_idx, save=True)
