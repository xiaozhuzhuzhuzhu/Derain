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


class ResBlockByWeight(nn.Module):
    def __init__(self, stride_size=1, dilation=1):
        super(ResBlockByWeight, self).__init__()
        self.stride = stride_size
        self.dilation = dilation

    def forward(self, x, w1, w2, b1, b2):
        input = x

        # kernel size
        kH, kW = w1.shape[2:4]
        # padding size
        pH = kH // 2 + self.dilation - 1
        pW = kW // 2 + self.dilation - 1
        x = torch.nn.functional.conv2d(x, w1, bias=b1, stride=self.stride, padding=(pH, pW), dilation=self.dilation)
        x = torch.relu(x)

        # kernel size
        kH, kW = w2.shape[2:4]
        # padding size
        pH = kH // 2 + self.dilation - 1
        pW = kW // 2 + self.dilation - 1
        x = torch.nn.functional.conv2d(x, w2, bias=b2, stride=self.stride, padding=(pH, pW), dilation=self.dilation)
        x = torch.relu(x)

        return torch.relu(x + input)
