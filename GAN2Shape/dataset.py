from os import path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root_dir, list_filename='list.txt', transform=None, subset=None):
        self.root_dir = root_dir
        self.transform = transform
        list_path = path.join(self.root_dir, list_filename)
        self.file_list = pd.read_csv(list_path, header=None)
        if subset is not None:
            try:
                self.file_list = self.file_list.iloc[subset].reset_index(drop=True)
            except IndexError as e:
                print(e, ": Invalid image subset indices specified \nExiting ...")
                quit()

    def __len__(self):
        return int(len(self.file_list))

    def __getitem__(self, index):
        img_path = path.join(self.root_dir, self.file_list[0][index])
        with Image.open(img_path) as image:
            if self.transform is not None:
                image = self.transform(image)
            # image = image[None, :]
            image = image * 2 - 1
            return image


class LatentDataset(Dataset):
    def __init__(self, root_dir, list_filename='list.txt', latent_folder='latents', subset=None):
        self.root_dir = root_dir
        self.latent_folder = latent_folder
        list_path = path.join(self.root_dir, list_filename)
        self.file_list = pd.read_csv(list_path, header=None)
        if subset is not None:
            try:
                self.file_list = self.file_list.iloc[subset].reset_index(drop=True)
            except IndexError as e:
                print(e, ": Invalid image subset indices specified \nExiting ...")
                quit()

    def __len__(self):
        return int(len(self.file_list))

    def __getitem__(self, index):
        latent_file = self.file_list[0][index].split('.')[0] + '.pt'
        latent_path = path.join(self.root_dir, self.latent_folder, latent_file)
        with torch.no_grad():
            latent = torch.load(latent_path, map_location='cpu')
            if type(latent) is dict:
                latent = latent['latent']
            return latent


class ImageLatentDataset(Dataset):
    def __init__(self, root_dir, list_filename='list.txt',
                 transform=None, latent_folder='latents', subset=None):
        self.image_dataset = ImageDataset(root_dir, list_filename, transform, subset)
        self.latent_dataset = LatentDataset(root_dir, list_filename, latent_folder, subset)
        assert len(self.image_dataset) == len(self.latent_dataset)

    def __len__(self):
        return int(len(self.image_dataset))

    def __getitem__(self, index):
        return self.image_dataset[index], self.latent_dataset[index], index
