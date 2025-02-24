import numpy as np

from torch.utils import data


class TwoViewTransformDataset(data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        img = np.array(img)

        img1 = self.transform(image=img)['image']
        img2 = self.transform(image=img)['image']

        return img1, img2

    def __len__(self):
        return len(self.dataset)