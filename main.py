import argparse
import yaml
from torchvision import transforms
from torch import cuda
from gan2shape.trainer import Trainer
from gan2shape.model import GAN2Shape
from gan2shape.dataset import ImageDataset, LatentDataset
from plotting import plot_originals
import logging
import time


def main():
    parser = argparse.ArgumentParser(description='Run a GAN 2D to 3D shape')
    parser.add_argument('--config-file',
                        dest='CONFIG',
                        default='config.yml',
                        help='path of the config yaml file')
    parser.add_argument('--wandb',
                        dest='WANDB',
                        action='store_true',
                        default=False)
    parser.add_argument('--save-ckpts',
                        dest='SAVE_CKPTS',
                        action='store_true',
                        default=False,
                        help='Save model weights after each stage')
    parser.add_argument('--debug',
                        dest='DEBUG',
                        action='store_true',
                        default=False,
                        help='Debug the model')
    parser.add_argument('--log-file',
                        dest='LOG_FILE',
                        default=None,
                        help='name of the logging file')
    args = parser.parse_args()

    if not cuda.is_available():
        print("A CUDA-enables GPU is required to run this model")
        exit(1)

    # read configuration
    with open(args.CONFIG, 'r') as config_file:
        config = yaml.safe_load(config_file)

    if args.WANDB:
        import wandb
        wandb.init(project=" gan-2d-to-3d", entity="dd2412-group42", config=config)

    # setup logging
    logging.basicConfig(filename=args.LOG_FILE,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO)

    # load/transform data
    transform = transforms.Compose(
            [
                transforms.Resize(config.get('image_size')),
                transforms.ToTensor()
            ]
        )

    if not args.SAVE_CKPTS:
        print(">>> Warning, not saving checkpoints. \n If this is a real run you want to rerun with --save-ckpts <<<")
        time.sleep(0.5)

    images = ImageDataset(config.get('root_path'), transform=transform)
    latents = LatentDataset(config.get('root_path'))
    # set configuration
    trainer = Trainer(model=GAN2Shape, model_config=config,
                      debug=args.DEBUG, plot_intermediate=True,
                      log_wandb=args.WANDB, save_ckpts=args.SAVE_CKPTS)


    # plot_originals(images)
    trainer.fit(images, latents)
    return


if __name__ == "__main__":
    main()
