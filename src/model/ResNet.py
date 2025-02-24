import torch
from torch import nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(self, ssl=True, num_classes=None):
        super(ResNet, self).__init__()
        resnet = models.resnet18(weights=None)
        # resnet = models.resnet50(weights=None)
        # resnet = models.resnet101(weights=None)

        if ssl:
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        else:
            assert num_classes is not None
            self.encoder = resnet

    def forward(self, x):
        return self.encoder(x).squeeze()


if __name__ == '__main__':
    model = ResNet()
    input = torch.randn(2, 3, 32, 32)
    output = model(input)
    print(output.shape)
    model = ResNet(ssl=False, num_classes=10)
    input = torch.randn(2, 3, 32, 32)
    output = model(input)
    print(output.shape)
