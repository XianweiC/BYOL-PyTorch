import argparse
import os
import sys
import numpy as np
from sklearn import preprocessing

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from src.dataset import STL10, ImageNet, get_transforms
from src.model import ResNet, MLP
from src.utils import get_features_from_encoder


if __name__ == "__main__":

    batch_size = 512

    train_set = STL10(root='/Users/xianweicao/Documents/workspace/Dataset/STL-10', train=True, transform=get_transforms('val', 96, 96))
    test_set = STL10(root='/Users/xianweicao/Documents/workspace/Dataset/STL-10', train=False, transform=get_transforms('val', 96, 96))

    print("Input shape:", train_set[0][0].shape)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=False)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device('mps')
    encoder = nn.Sequential(ResNet(), MLP(output_channels=128))
    # encoder = ResNet()

    params = torch.load('checkpoints/BYOL/STL-10/2025_40_256_2048_256_256/113741/model-final.pth', weights_only=True)
    print(encoder.load_state_dict(params, strict=False))
    encoder = nn.Sequential(*list(encoder.children())[:-1])
    encoder = encoder.to(device)

    input_dim = 512
    output_dim = 10
    logreg = nn.Linear(input_dim, output_dim)
    logreg = logreg.to(device)

    encoder.eval()
    x_train, y_train = get_features_from_encoder(encoder, train_loader, device)
    x_test, y_test = get_features_from_encoder(encoder, test_loader, device)

    if len(x_train.shape) > 2:
        x_train = torch.mean(x_train, dim=[2, 3])
        x_test = torch.mean(x_test, dim=[2, 3])

    print("Training data shape:", x_train.shape, y_train.shape)
    print("Testing data shape:", x_test.shape, y_test.shape)


    def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test):
        train = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

        test = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False)
        return train_loader, test_loader

    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train.cpu().numpy())
    x_train = scaler.transform(x_train.cpu().numpy()).astype(np.float32)
    x_test = scaler.transform(x_test.cpu().numpy()).astype(np.float32)
    x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)

    train_loader, test_loader = create_data_loaders_from_arrays(x_train, y_train, x_test, y_test)

    optimizer = torch.optim.Adam(logreg.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    eval_every_n_epochs = 10

    for epoch in range(200):
        #     train_acc = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            logits = logreg(x)
            predictions = torch.argmax(logits, dim=1)

            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

        total = 0
        if epoch % eval_every_n_epochs == 0:
            correct = 0
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                logits = logreg(x)
                predictions = torch.argmax(logits, dim=1)

                total += y.size(0)
                correct += (predictions == y).sum().item()

            acc = 100 * correct / total
            print(f"Testing accuracy: {np.mean(acc)}")
