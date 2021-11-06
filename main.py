import argparse
import configparser
from GAN2Shape.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Run a GAN 2D to 3D shape')
    parser.add_argument('--config-file',
                        dest='CONFIG',
                        default='config.ini',
                        help='path of the config file')
    args = parser.parse_args()

    # read configuration
    config = configparser.ConfigParser()
    config.read(args.CONFIG)

    # set configuration
    trainer = Trainer(n_epochs=1)
    trainer.fit()


if __name__ == "__main__":
    main()
