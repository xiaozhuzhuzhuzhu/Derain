import torch
import torch.nn as nn


class ConvLstm(nn.Module):
    def __init__(self, in_channels, kernel_size, padding_mode="reflect"):
        super(ConvLstm, self).__init__()
        padding = kernel_size // 2
        self.convi = nn.Sequential(
            nn.Conv2d(in_channels=2*in_channels,
                      out_channels=in_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      padding_mode=padding_mode),
            nn.Sigmoid()
        )
        self.convf = nn.Sequential(
            nn.Conv2d(in_channels=2*in_channels,
                      out_channels=in_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      padding_mode=padding_mode),
            nn.Sigmoid()
        )
        self.convg = nn.Sequential(
            nn.Conv2d(in_channels=2*in_channels,
                      out_channels=in_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      padding_mode=padding_mode),
            nn.Tanh()
        )
        self.convo = nn.Sequential(
            nn.Conv2d(in_channels=2*in_channels,
                      out_channels=in_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      padding_mode=padding_mode),
            nn.Sigmoid()
        )

    def forward(self, xh, c):
        i = self.convi(xh)
        f = self.convf(xh)
        g = self.convg(xh)
        o = self.convo(xh)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

if __name__ == '__main__':
    batch_size = 10
    channels = 32
    height = 10
    width = 10
    kernel_size = 3

    x = torch.rand(batch_size, channels, height, width)
    h = torch.rand(batch_size, channels, height, width)
    c = torch.rand(batch_size, channels, height, width)
    model = ConvLstm(in_channels=channels, kernel_size=kernel_size)
    h, c = model(torch.cat([x, h], 1), c)

    print(h.shape, c.shape)
