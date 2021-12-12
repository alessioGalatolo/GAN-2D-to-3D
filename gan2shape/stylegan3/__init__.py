import sys
import os
from pathlib import Path

current_path = os.getcwd()

module_path = Path(__file__).parent
sys.path.append(str(module_path.resolve()))
os.chdir(module_path)

from training.networks_stylegan2 import Discriminator
from training.networks_stylegan2 import Generator as GeneratorV2
from training.networks_stylegan3 import Generator as GeneratorV3
from legacy import load_network_pkl
from torch_utils.misc import copy_params_and_buffers
from dnnlib.util import open_url

os.chdir(current_path)
