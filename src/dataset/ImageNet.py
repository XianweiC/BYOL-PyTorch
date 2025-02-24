import numpy as np

import torch
from torch.utils import data
from torchvision import datasets


class ImageNet(data.Dataset):
    def __init__(self, root='', train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        if self.train:
            self.dataset = datasets.ImageNet(root, split='train')
        else:
            self.dataset = datasets.ImageNet(root, split='val')

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = np.array(img)

        if self.transform is not None:
            img = self.transform(image=img)['image']
            label = torch.tensor(label).long()
        return img, label

    def __len__(self):
        return len(self.dataset)


class FastImageNet(data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

    def __getitem__(self, index):
        return

    def __len__(self):
        return
