import os
import sys

import pytorch_lightning as pl

from glob import glob
from PIL import Image

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, Normalize, ToTensor, Resize


import numpy as np


class FFHQDataModule(pl.LightningDataModule):
    def __init__(self, opts):
        super(FFHQDataModule, self).__init__()
        self.opts = opts

        dataset = _FFHQDataset(self.opts.dataset_dir, self.opts.output_size)
        size = dataset.__len__()
        train_size = int(0.96 * (size))
        val_size = int(size - train_size)
        self.train_set, self.val_set = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        dataloader = DataLoader(self.train_set, batch_size=self.opts.batch_size, num_workers=self.opts.workers,
                                shuffle=True)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_set, batch_size=self.opts.batch_size, num_workers=self.opts.workers,
                                shuffle=True)
        return dataloader


class _FFHQDataset(Dataset):
    def __init__(self, data_dir, image_size):
        super(_FFHQDataset, self).__init__()

        self.transform = Compose([
            Resize(image_size),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.data_dir = data_dir

        self.image_list = self.populate_path_list(self.data_dir)
        self.image_list = self.image_list.astype(np.string_)
        print(self.image_list.shape)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        image = Image.open(self.image_list[index])
        image = self.transform(image)

        return image

    @staticmethod
    def populate_path_list(path):
        types = ('*.jpg', '*.png')

        path_list = []
        for files in types:
            path_list.extend(sorted(glob(os.path.join(path, files))))
        path_list = sorted(path_list)

        path_list = np.array(path_list)

        return path_list


if __name__ == "__main__":
    dataset = _FFHQDataset(sys.argv[1], 256)
    image = dataset.__getitem__(0)
    print(image.shape)