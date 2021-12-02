import argparse
import yaml
from os import path
from torchvision import transforms
from torch import cuda
from GAN2Shape.trainer import Trainer
from GAN2Shape.model import GAN2Shape
from GAN2Shape.dataset import ImageDataset, LatentDataset


def main():
    parser = argparse.ArgumentParser(description='Run a GAN 2D to 3D shape')
    parser.add_argument('--config-file',
                        dest='CONFIG',
                        default='config.yml',
                        help='path of the config yaml file')
    args = parser.parse_args()

    if not cuda.is_available():
        print("A CUDA-enables GPU is required to run this model")
        exit(1)

    # read configuration
    with open(args.CONFIG, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # load/transform data
    transform = transforms.Compose(
            [
                transforms.CenterCrop(config.get('image_size')),
                transforms.Resize(config.get('image_size')),
                transforms.ToTensor()
            ]
        )
    config['transform'] = transform
    images = ImageDataset(config.get('root_path'), transform=transform)
    latents = LatentDataset(config.get('root_path'))
    # set configuration
    trainer = Trainer(model=GAN2Shape, model_config=config)
    trainer.fit(images, latents, plot_depth_map=False)


if __name__ == "__main__":
    main()
