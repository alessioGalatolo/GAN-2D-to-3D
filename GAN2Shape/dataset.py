from os import path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root_dir, list_filename='list.txt', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        list_path = path.join(self.root_dir, list_filename)
        self.file_list = pd.read_csv(list_path, header=None)

    def __len__(self):
        return int(len(self.file_list))

    def __getitem__(self, index):
        img_path = path.join(self.root_dir, self.file_list[0][index])
        with Image.open(img_path) as image:
            if self.transform is not None:
                image = self.transform(image)
            # image = image[None, :]
            image = image.unsqueeze(0)
            image = image * 2 - 1
            return image


class LatentDataset(Dataset):
    def __init__(self, root_dir, list_filename='list.txt', latent_folder='latents'):
        self.root_dir = root_dir
        self.latent_folder = latent_folder
        list_path = path.join(self.root_dir, list_filename)
        self.file_list = pd.read_csv(list_path, header=None)

    def __len__(self):
        return int(len(self.file_list))

    def __getitem__(self, index):
        latent_file = self.file_list[0][index].split('.')[0] + '.pt'
        latent_path = path.join(self.root_dir, self.latent_folder, latent_file)
        with torch.no_grad():
            latent = torch.load(latent_path, map_location='cpu')
            if type(latent) is dict:
                latent = latent['latent']
            if latent.dim() == 1:
                latent = latent.unsqueeze(0)
            return latent
