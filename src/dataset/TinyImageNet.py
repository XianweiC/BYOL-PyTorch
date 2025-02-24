import os
import cv2

import torch
from torch.utils import data


class TinyImageNet(data.Dataset):
    '''
    Tiny ImageNet dataset.
    It contains 100000 samples for training. The class is 200.
    10000 samples for validation and testing.
    Each sample has a size (64, 64, 3).
    '''
    def __init__(self, root, split='train', transform=None):
        super(TinyImageNet, self).__init__()

        with open(os.path.join(root, 'wnids.txt')) as f:
            wnids = f.read().splitlines()
        self.id2label = {ids: i for i, ids in enumerate(wnids)}

        with open(os.path.join(root, 'words.txt')) as f:
            words = f.read().splitlines()
        self.id2name = {line.split('\t')[0]: line.split('\t')[1] for line in words}

        if split == 'train':
            dataset = []
            self.dir = os.path.join(root, 'train')
            for subdir in os.listdir(self.dir):
                for filename in os.listdir(os.path.join(self.dir, subdir, 'images')):
                    path = os.path.join(self.dir, subdir, 'images', filename)
                    label = self.id2label[subdir]
                    dataset.append([path, label])

        elif split == 'val':
            self.dir = os.path.join(root, 'val')
            with open(os.path.join(self.dir, 'val_annotations.txt')) as f:
                val_annotations = f.read().splitlines()
            val_annotations = {line.split('\t')[0]: line.split('\t')[1] for line in val_annotations}
            dataset = []
            for filename, cls in val_annotations.items():
                path = os.path.join(self.dir, 'images', filename)
                label = self.id2label[cls]
                dataset.append([path, label])

        # NOTE: TODO: Only use when evaluating model performance
        # elif split == 'test':
        #     self.dir = os.path.join(root, 'test')

        else:
            raise NotImplementedError

        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = cv2.imread(img)[:, :, ::-1]

        if self.transform is not None:
            img = self.transform(image=img)['image']
            label = torch.tensor(label).long()

        return img, label

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    from tqdm import tqdm
    train_set = TinyImageNet('/Users/xianweicao/Documents/workspace/Dataset/tiny-imagenet-200', 'train')
    i = 0
    label_max = 0
    for img, label in tqdm(train_set):
        i += 1
        if label > label_max:
            label_max = label
        print(img.shape, label)
