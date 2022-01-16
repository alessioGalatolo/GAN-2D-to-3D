import argparse
import yaml
from torchvision import transforms
from torch import cuda
from GAN2Shape.model import GAN2Shape, MaskingModel
from GAN2Shape.dataset import ImageDataset
from GAN2Shape import utils
import numpy as np
from plotting import *
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate GAN 2D to 3D shape')
    # parser.add_argument('--ckpt_path',
    #                     dest='CKPT_PATH',
    #                     help='path of the saved weights')
    parser.add_argument('--config-file',
                        dest='CONFIG',
                        default='config.yml',
                        help='path of the config yaml file')
    parser.add_argument('--generalize',
                        dest='GENERALIZE',
                        action='store_true',
                        default=False,
                        help='If to run training procedure that favors generalization')
    parser.add_argument('--record-loss',
                        dest='RECORD_LOSS',
                        default=None,
                        help='Set to "filename" (not full path) to save the step1 loss list in "results/loss_lists/filename"')
    args = parser.parse_args()
    # read configuration
    with open(args.CONFIG, 'r') as config_file:
        config = yaml.safe_load(config_file)
    if not cuda.is_available():
        print("A CUDA-enables GPU is required to run this model")
        exit(1)

    utils.create_results_folder()
    transform = transforms.Compose(
            [
                transforms.Resize(config.get('image_size')),
                transforms.ToTensor()
            ]
        )
    subset = config.get('image_subset', None)
    category = config.get('category')
    base_path = config.get('our_nets_ckpts')['VLADE_nets']
    stage = config.get('stage', '*')
    iteration = config.get('iteration', '*')
    time = config.get('time', '*')

    images = ImageDataset(config.get('root_path'), transform=transform,
                          subset=subset)
    model = GAN2Shape(config)
    masking_model = MaskingModel(category)

    if not args.GENERALIZE:
        generator = model.load_from_checkpoints(base_path, category)
        if subset is not None:
            print('>>> Warning: Evaluating the instance-specific model with subset is probably not what you want.')
    else:
        print('>>> Using general model for all predictions')
        if subset is None:
            print('>>> Error: Subset must be specified for the general model. Exiting ...')
            quit()
        paths, _ = model.build_checkpoint_path(base_path, category, general=args.GENERALIZE)
        model.load_from_checkpoint(paths[-1])

        generator = np.arange(len(subset))

    loss_list = []
    for img_idx in tqdm(generator):
        img1 = images[img_idx].unsqueeze(0)
        recon_im, recon_depth = model.evaluate_results(img1.cuda())
        if args.RECORD_LOSS is not None:
            loss_step1, _ = model.forward_step1(img1.cuda(), None, None, step1=True, eval=False)
            loss_list.append(loss_step1.detach().cpu().item())
        recon_im, recon_depth = recon_im.cpu(), recon_depth.cpu()
        if args.GENERALIZE:
            plt_idx = subset[img_idx]
        else:
            plt_idx = img_idx
        # plot_originals(images[img_idx].unsqueeze(0), block=True)
        plot_reconstructions(recon_im, recon_depth, block=False, im_idx=str(plt_idx))

        recon_depth = masking_model.image_mask(img1.cuda(), recon_depth)

        plotly_3d_animate(recon_depth, texture=img1, img_idx=plt_idx, save=True, show=False, create_gif=False)

    if args.RECORD_LOSS is not None:
        loss_list = np.array(loss_list)
        statistical_box_plot(loss_list)
        mean = np.mean(loss_list)
        std = np.std(loss_list)
        print('mean = ', mean)
        print('std = ', std)
        np.save('results/loss_lists/step1_' + args.RECORD_LOSS + '_model', loss_list)
