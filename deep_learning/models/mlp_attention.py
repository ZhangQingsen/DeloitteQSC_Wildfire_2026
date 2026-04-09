import torch
import torch.nn as nn


class FeatureAttention(nn.Module):
    """
    Feature-wise attention (SENet-style) for tabular data.
    Input:  (B, D)
    Output: (B, D), each feature reweighted by a learned gate in [0, 1].
    """
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        gate = self.fc(x)   # (B, D)
        return x * gate     # feature-wise reweight


class MLPAttentionPlain(nn.Module):
    """
    Plain MLP with a feature-wise attention front-end.
    Depth controlled by num_layers (all Linear+ReLU, no residual).
    """
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=3):
        super().__init__()

        self.att = FeatureAttention(input_dim)

        layers = []
        # first layer: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # middle layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # output head
        layers.append(nn.Linear(hidden_dim, 32))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(32, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.att(x)
        return self.net(x)


class ResidualBlock(nn.Module):
    """
    Standard 2-layer MLP residual block: x -> FC -> ReLU -> FC -> +x -> ReLU
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


class MLPAttentionRes(nn.Module):
    """
    MLP-ResNet with feature-wise attention.
    Depth controlled by num_blocks (each block is a ResidualBlock).
    """
    def __init__(self, input_dim=7, hidden_dim=64, num_blocks=3):
        super().__init__()

        self.att = FeatureAttention(input_dim)

        self.fc_in = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])

        self.fc_out1 = nn.Linear(hidden_dim, 32)
        self.fc_out2 = nn.Linear(32, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.att(x)
        x = self.act(self.fc_in(x))

        for blk in self.blocks:
            x = blk(x)

        x = self.act(self.fc_out1(x))
        x = self.fc_out2(x)
        return x
