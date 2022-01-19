import argparse
import yaml
from os import path
from torchvision import transforms
from torch import cuda
from GAN2Shape.trainer import Trainer, GeneralizingTrainer, GeneralizingTrainer2
from GAN2Shape.model import GAN2Shape
from GAN2Shape.dataset import ImageLatentDataset
from GAN2Shape.utils import create_results_folder
import logging
import time


def main():
    parser = argparse.ArgumentParser(description='Run a GAN 2D to 3D shape')
    parser.add_argument('--config-file',
                        dest='CONFIG',
                        default='config.yml',
                        help='path of the config yaml file')
    parser.add_argument('--category',
                        dest='CATEGORY',
                        default=None,
                        help='The object on which to run GAN2Shape, will use adequate config files')
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
    parser.add_argument('--load-pretrained',
                        dest='LOAD_PRETRAINED',
                        action='store_true',
                        default=False,
                        help='Load pretrained weights before training')
    parser.add_argument('--generalize',
                        dest='GENERALIZE',
                        action='store_true',
                        default=False,
                        help='If to run training procedure that favors generalization')
    parser.add_argument("--images",
                        dest="IMAGES",
                        action="append",
                        type=int,
                        default=None,
                        nargs="+",
                        help="Image numbers on which to run the method")
    args = parser.parse_args()

    if not cuda.is_available():
        print("A CUDA-enables GPU is required to run this model")
        exit(1)

    if args.CATEGORY is not None:
        category = args.CATEGORY
        with open('minimal_config.yml', 'r') as minimal_config_file,\
             open(path.join("configs", f'{category}.yml'), 'r') as specific_config_file:
            minimal_config = yaml.safe_load(minimal_config_file)
            specific_config = yaml.safe_load(specific_config_file)
            config = {**minimal_config, **specific_config}  # python 3.5+
            config['category'] = category
    else:
        # read given configuration
        with open(args.CONFIG, 'r') as config_file:
            config = yaml.safe_load(config_file)
            category = config.get('category')

    if args.WANDB:
        import wandb
        wandb.init(project=" gan-2d-to-3d",
                   entity="dd2412-group42",
                   config=config)
        config = wandb.config

    # setup logging
    logging.basicConfig(filename=args.LOG_FILE,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO)
    create_results_folder()
    # load/transform data
    transform = transforms.Compose(
            [
                transforms.Resize(config.get('image_size')),
                transforms.ToTensor()
            ]
        )

    load_dict = None
    if args.LOAD_PRETRAINED:
        load_dict = {
            'category': config.get('category'),
            'base_path': config.get('our_nets_ckpts')['VLADE_nets'],
            'stage': config.get('stage', '*'),
            'iteration': config.get('iteration', '*'),
            'time': config.get('time', '*')
        }

    if not args.SAVE_CKPTS:
        print(">>> Warning, not saving checkpoints.")
        print("If this is a real run you want to rerun with --save-ckpts <<<")
        time.sleep(0.5)

    data_folder = path.join(config.get('root_path'), category)
    subset = args.IMAGES
    if subset is not None:
        subset = [image for image_list in subset for image in image_list]
    images_latents = ImageLatentDataset(data_folder,
                                        transform=transform,
                                        subset=subset)

    # set configuration
    trainer_config = {
        'model': GAN2Shape, 'model_config': config,
        'debug': args.DEBUG, 'plot_intermediate': True,
        'log_wandb': args.WANDB, 'save_ckpts': args.SAVE_CKPTS,
        'load_dict': load_dict
    }

    if args.GENERALIZE:
        trainer = GeneralizingTrainer2(**trainer_config)
        # the original method totals = [{'step1': 1300, 'step2': 2200, 'step3': 1800}]
        # hence the choice of the below setting for n_epochs = 100
        stages = [{'step1': 13, 'step2': 22, 'step3': 18}]
        # stages = [{'step1': 1, 'step2': 1, 'step3': 1}]
        if subset is not None:
            print(">>> Warning, using a subset with a generalizing trainer.")
            print("It is always better to use the whole dataset.<<<")
    else:
        trainer = Trainer(**trainer_config)
        stages = [{'step1': 700, 'step2': 700, 'step3': 600},
                  {'step1': 200, 'step2': 500, 'step3': 400},
                  {'step1': 200, 'step2': 500, 'step3': 400},
                  {'step1': 200, 'step2': 500, 'step3': 400}]

    trainer.fit(images_latents, stages=stages, batch_size=config.get('batch_size', 2))
    return


if __name__ == "__main__":
    main()
