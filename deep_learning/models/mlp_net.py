import torch
import torch.nn as nn


class MLPPlain(nn.Module):
    """
    Plain MLP with configurable depth (num_layers).
    No skip connections.
    """
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=5):
        super().__init__()

        layers = []
        # first layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # middle layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # output head
        layers.append(nn.Linear(hidden_dim, 32))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(32, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """
    Standard 2-layer MLP residual block.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        out = out + identity
        out = self.act(out)
        return out


class MLPResNet(nn.Module):
    """
    MLP with residual blocks.
    Depth controlled by num_blocks.
    """
    def __init__(self, input_dim=7, hidden_dim=64, num_blocks=3):
        super().__init__()

        self.fc_in = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])

        self.fc_out1 = nn.Linear(hidden_dim, 32)
        self.fc_out2 = nn.Linear(32, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc_in(x))

        for block in self.blocks:
            x = block(x)

        x = self.act(self.fc_out1(x))
        x = self.fc_out2(x)
        return x
