import torch
import torch.nn as nn


class CNN1DPlain(nn.Module):
    """
    Plain 1D CNN with configurable depth (num_layers).
    No skip connections.
    """
    def __init__(self, input_dim=7, hidden_channels=32, num_layers=3):
        super().__init__()

        self.input_dim = input_dim

        layers = []
        in_channels = 1

        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = hidden_channels

        self.conv = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, L)
        x = self.conv(x)
        x = self.head(x)
        return x


class CNN1DResidualBlock(nn.Module):
    """
    Standard 2-layer Conv1D residual block.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out = out + identity
        out = self.act(out)
        return out


class CNN1DResNet(nn.Module):
    """
    1D CNN with residual blocks.
    Depth controlled by num_blocks.
    """
    def __init__(self, input_dim=7, hidden_channels=32, num_blocks=3):
        super().__init__()

        self.input_dim = input_dim

        # input projection: 1 → hidden_channels
        self.conv_in = nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()

        # residual blocks
        self.blocks = nn.ModuleList([
            CNN1DResidualBlock(hidden_channels) for _ in range(num_blocks)
        ])

        # output head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, L)
        x = self.act(self.conv_in(x))

        for block in self.blocks:
            x = block(x)

        x = self.head(x)
        return x
