import argparse
import configparser


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


if __name__ == "__main__":
    main()
