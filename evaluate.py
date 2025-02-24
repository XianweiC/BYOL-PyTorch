from ml_collections import config_dict
import numpy as np
import timm

import torch
from torch import nn
from torch.utils import data
from torchvision import models

from src.dataset import CIFAR10, get_transforms
from src.utils import setup_system
from engine import validate_each_classes

if __name__ == '__main__':
    device = torch.device('mps')

    batch_size = 256
    random_seed = 2025
    num_classes = 10
    setup_system(random_seed)

    param = torch.load('checkpoints/Supervised/CIFAR-10/2025_1000_256_2048_256_256/192208/model-449.pth')
    model = models.resnet18(weights=None, num_classes=num_classes).to(device)
    model.load_state_dict(param, strict=True)

    # data
    # transform_train = get_transforms('train')
    transform_test = get_transforms('test')
    train_set = CIFAR10(train=True, transform=transform_test)
    test_set = CIFAR10(train=False, transform=transform_test)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # criterion
    criterion = config_dict.ConfigDict()
    criterion.ce_loss = nn.CrossEntropyLoss()

    num_classes = len(train_set.dataset.classes)
    id2cls = [cls for cls in train_set.dataset.class_to_idx.keys()]

    # train set Acc.
    acc, class_accuracies = validate_each_classes(model, train_loader, criterion, num_classes, device)
    print(f'Top1: {acc} %')
    for idx, acc in class_accuracies.items():
        print(f'{id2cls[idx]}: {acc}')

    # test set Acc.
    acc, class_accuracies = validate_each_classes(model, test_loader, criterion, num_classes, device)
    print(f'Top1: {acc} %')
    for idx, acc in class_accuracies.items():
        print(f'{id2cls[idx]}: {acc}')

    # Radar map
    # example_values = [v * 100. for v in class_accuracies.values()]  # Values range from 0 to 1 (percentage)
    # example_labels = [cls for cls in id2cls]

    # plot_n_polygon(example_labels, example_values)
