import argparse
import yaml
from torchvision import transforms
from GAN2Shape.trainer import Trainer
from GAN2Shape.model import GAN2Shape
from GAN2Shape.dataset import GenericDataset


def main():
    parser = argparse.ArgumentParser(description='Run a GAN 2D to 3D shape')
    parser.add_argument('--config-file',
                        dest='CONFIG',
                        default='config.yml',
                        help='path of the config yaml file')
    args = parser.parse_args()

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
    dataset = GenericDataset(config.get('root_path'), transform=transform)
    # set configuration
    trainer = Trainer(model=GAN2Shape, model_config=config)
    trainer.fit(dataset)


if __name__ == "__main__":
    main()
