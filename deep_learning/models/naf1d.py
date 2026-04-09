import torch
import torch.nn as nn

class SimpleGate(nn.Module):
    def forward(self, x):
        # x: (B, C, L)
        c = x.size(1)
        x1, x2 = x[:, :c//2, :], x[:, c//2:, :]
        return x1 * x2


class SCA1D(nn.Module):
    """
    Simple Channel Attention for 1D:
    global average pooling over length, then 1x1 conv (FC) + sigmoid
    """
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, L)
        w = x.mean(dim=2, keepdim=True)  # GAP over length
        w = self.fc(w)                   # (B, C, 1)
        return x * w


class NAFBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

        self.pw1 = nn.Conv1d(channels, channels * 2, kernel_size=1)   # pointwise
        self.dw  = nn.Conv1d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2)  # depthwise
        self.sg  = SimpleGate()
        self.sca = SCA1D(channels)

        self.pw2 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, L)
        identity = x

        # LayerNorm over channel dim → 先换成 (B, L, C)
        b, c, l = x.shape
        y = x.permute(0, 2, 1)          # (B, L, C)
        y = self.norm(y)
        y = y.permute(0, 2, 1)          # (B, C, L)

        y = self.pw1(y)
        y = self.dw(y)
        y = self.sg(y)                  # (B, C, L)
        y = self.sca(y)
        y = self.pw2(y)

        return y + identity

class NAFNet1DPlain(nn.Module):
    def __init__(self, input_dim=7, channels=32, num_layers=3):
        super().__init__()
        self.input_dim = input_dim

        self.proj_in = nn.Conv1d(1, channels, kernel_size=1)

        self.blocks = nn.ModuleList([
            NAFBlock1D(channels) for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)          # (B, 1, L)
        x = self.proj_in(x)         # (B, C, L)
        for blk in self.blocks:
            x = blk(x)
        x = self.head(x)
        return x


class NAFNet1DRes(nn.Module):
    def __init__(self, input_dim=7, channels=32, num_blocks=3):
        super().__init__()
        self.input_dim = input_dim

        self.proj_in = nn.Conv1d(1, channels, kernel_size=1)

        self.blocks = nn.ModuleList([
            NAFBlock1D(channels) for _ in range(num_blocks)
        ])

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)          # (B, 1, L)
        x = self.proj_in(x)         # (B, C, L)
        for blk in self.blocks:
            x = blk(x)
        x = self.head(x)
        return x
