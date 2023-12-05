from torchvision import datasets
from torchvision import transforms as T

import torch

import torch.utils.data as data_utils


class Dataset:
    def init(self):
        self.train_ds = datasets.CIFAR10('./data', train=True, download=False,
            transform=T.Compose([
                T.ToTensor(),
                T.Resize(256), T.CenterCrop(128),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ]))

        ids = torch.arange(2000)
        self.train_ds = data_utils.Subset(self.train_ds, ids)

    def train_dataset(self):
        return self.train_ds

    def test_dataset(self):
        return None
