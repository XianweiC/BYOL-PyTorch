from torch import nn


class MLP(nn.Module):
    def __init__(self, input_channels=512, hidden_size=512, output_channels=128):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_channels, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_channels),
        )

    def forward(self, x):
        return self.net(x)


class LinearProbing(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProbing, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()

    def forward(self, x):
        return self.fc(x)
