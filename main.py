import argparse
import yaml
from GAN2Shape.trainer import Trainer
from GAN2Shape.model import GAN2Shape


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

    # set configuration
    trainer = Trainer(model=GAN2Shape, model_config=config)
    trainer.fit()


if __name__ == "__main__":
    main()
