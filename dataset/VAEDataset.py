import os
import sys
import numpy as np
import pickle as pkl
import pytorch_lightning as pl
import torch

from glob import glob
from torch.utils.data import Dataset, DataLoader, random_split


class ExpressionDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        dataset = _ExpressionDataset(self.data_dir)
        size = dataset.__len__()
        train_size = int(0.95 * (size))
        val_size = int(size - train_size)
        self.train_set, self.val_set = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        dataloader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return dataloader


class _ExpressionDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()

        self.expression_coefficients = np.load(data_dir)
        self.expression_coefficients = torch.from_numpy(self.expression_coefficients)

    def __getitem__(self, index):
        expression = self.expression_coefficients[index]
        return expression

    def __len__(self):
        return self.expression_coefficients.shape[0]

    def get_shape(self):
        return self.expression_coefficients.shape


if __name__ == "__main__":
    dataset = _ExpressionDataset(sys.argv[1])
    print(dataset.__len__())