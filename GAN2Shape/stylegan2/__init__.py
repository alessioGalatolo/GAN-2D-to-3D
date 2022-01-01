import sys
import os
from pathlib import Path

current_path = os.getcwd()

module_path = Path(__file__).parent / 'stylegan2-pytorch'
sys.path.append(str(module_path.resolve()))
os.chdir(module_path)

from model import Generator, Discriminator
from lpips import PerceptualLoss

os.chdir(current_path)
