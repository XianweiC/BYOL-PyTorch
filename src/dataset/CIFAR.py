import os
import numpy as np

import torch
from torch.utils import data
from torchvision import datasets


class CIFAR10(data.Dataset):
    def __init__(self, root='/Users/xianweicao/Documents/workspace/SSL/data/CIFAR', train=True, transform=None):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=False)
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = np.array(img)

        if self.transform is not None:
            img = self.transform(image=img)['image']
            label = torch.tensor(label).long()

        # return img, label, name
        return img, label

    def __len__(self):
        return len(self.dataset)


class CIFAR100(data.Dataset):
    def __init__(self, root='/Users/xianweicao/Documents/workspace/SSL/data/CIFAR', train=True, transform=None):
        self.dataset = datasets.CIFAR100(root=root, train=train)
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = np.array(img)

        if self.transform is not None:
            img = self.transform(image=img)['image']
            label = torch.tensor(label).long()

        # return img, label, name
        return img, label

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    train_set = CIFAR10(train=True)
    train_set = data.DataLoader(train_set, batch_size=256, shuffle=False, num_workers=0)
    i = 0
    for i in range(1000):
        for img, label in train_set:
            # i += 1
            print(img)
            print(label)
        print(i)
