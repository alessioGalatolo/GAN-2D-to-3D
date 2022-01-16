import argparse
import yaml
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
    args = parser.parse_args()

    if not cuda.is_available():
        print("A CUDA-enables GPU is required to run this model")
        exit(1)

    # read configuration
    with open(args.CONFIG, 'r') as config_file:
        config = yaml.safe_load(config_file)

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

    images_latents = ImageLatentDataset(config.get('root_path'),
                                        transform=transform,
                                        subset=config.get('image_subset', None)
                                        )

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
        stages = {  False:  {'step1': 13, 'step2': 1, 'step3': 1, #initialization epochs
                    'n_init_iterations': config.get('n_init_iterations', 1)}, 
                    True:   {'step1': 1, 'step2': 1, 'step3': 18, #regular epochs
                    'n_init_iterations': 1} 
                    }
        if 'image_subset' in config:
            print(">>> Warning, using a subset with a generalizing trainer.")
            print("It is always better to use the whole dataset.<<<")        
    else:
        trainer = Trainer(**trainer_config)
        # 1 epoch of step 1 -> n_epochs_init epochs of init iterations -> proceed with the original stages
        first_stage = [{'step1': 700, 'step2': 0, 'step3': 0, 'n_init_iterations': 0}]
        init_stages = [{'step1': 1, 'step2': 1, 'step3': 1, #initialization epochs
                    'n_init_iterations': config.get('n_init_iterations', 1)}] * config.get('n_epochs_init', 1)

        normal_stages = [{'step1': 1, 'step2': 700, 'step3': 600, 'n_init_iterations': 1},
                        {'step1': 200, 'step2': 500, 'step3': 400, 'n_init_iterations': 1},
                        {'step1': 200, 'step2': 500, 'step3': 400, 'n_init_iterations': 1},
                        {'step1': 200, 'step2': 500, 'step3': 400, 'n_init_iterations': 1}]
        stages = first_stage + init_stages + normal_stages

    trainer.fit(images_latents, stages=stages, batch_size=config.get('batch_size', 2))
    return


if __name__ == "__main__":
    main()
