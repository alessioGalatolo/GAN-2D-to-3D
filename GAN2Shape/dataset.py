from os import path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class GenericDataset(Dataset):
    def __init__(self, root_dir, list_filename='list.txt', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        list_path = path.join(self.root_dir, list_filename)
        self.file_list = pd.read_csv(list_path, header=None)

    def __len__(self):
        return int(len(self.file_list))

    def __getitem__(self, index):
        img_path = path.join(self.root_dir, self.file_list[0][index])
        with Image.open(img_path)as image:
            if self.transform:
                image = self.transform(image)
            # image = image[None, :]
            image = image.unsqueeze(0)
            return image
