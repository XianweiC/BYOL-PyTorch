import numpy as np

import torch
from torch.utils import data
from torchvision import datasets


class STL10(data.Dataset):
    def __init__(self, root='/Users/xianweicao/Documents/workspace/Dataset/', train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        if self.train:
            self.dataset = datasets.STL10(self.root, split='train', download=False)
        else:
            self.dataset = datasets.STL10(self.root, split='test', download=False)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = np.array(img)

        if self.transform is not None:
            img = self.transform(image=img)['image']
            label = torch.tensor(label).long()
        return img, label

    def __len__(self):
        return len(self.dataset)
