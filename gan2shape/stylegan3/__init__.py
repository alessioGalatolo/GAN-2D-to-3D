import sys
import os
from pathlib import Path

current_path = os.getcwd()

module_path = Path(__file__).parent
sys.path.append(str(module_path.resolve()))
os.chdir(module_path)

from training.networks_stylegan2 import Discriminator
from training.networks_stylegan3 import Generator

os.chdir(current_path)
