import torch
from torch import nn


class ConvGru(nn.Module):
    def __init__(self, in_channels, kernel_size, padding_mode="reflect"):
        super(ConvGru, self).__init__()
        padding = kernel_size // 2
        self.convr = nn.Sequential(
            nn.Conv2d(in_channels=2 * in_channels,
                      out_channels=in_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      padding_mode=padding_mode),
            nn.Sigmoid()
        )
        self.convz = nn.Sequential(
            nn.Conv2d(in_channels=2 * in_channels,
                      out_channels=in_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      padding_mode=padding_mode),
            nn.Sigmoid()
        )
        self.convh = nn.Sequential(
            nn.Conv2d(in_channels=2*in_channels,
                      out_channels=in_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      padding_mode=padding_mode),
            nn.Tanh()
        )

    def forward(self, x, h):
        input = torch.cat([x, h], dim=1)
        r = self.convr(input)
        z = self.convz(input)
        h_0 = r * h
        input_h_1 = torch.cat([x, h_0], dim=1)
        h_1 = self.convh(input_h_1)
        h = (1 - z) * h + z * h_1
        return h
