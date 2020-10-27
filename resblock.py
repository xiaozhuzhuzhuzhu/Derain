import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, n_channels=32, kernel_size=3 ,stride_size=1, padding_size=1):
        super(ResBlock, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size, stride_size, padding_size),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, kernel_size, stride_size, padding_size),
            nn.ReLU(),
        )

    def forward(self, x):
        x = torch.relu(self.resblock(x) + x)
        return x