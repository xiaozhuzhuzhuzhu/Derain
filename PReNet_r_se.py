import torch
import torch.nn as nn
from resblock import ResBlock
from convlstm import ConvLstm
from se import SELayer


class PReNet_r(nn.Module):
    def __init__(self, recurrent_iter, fi_in_channels=3,
                 n_intermediate_channels=32, kernel_size=3, stride_size=1,
                 padding_size=1, fo_out_channels=3, n_resblocks=5, reduction=16):
        super(PReNet_r, self).__init__()
        self.iteration = recurrent_iter
        self.fi_in_channels = fi_in_channels
        self.n_intermediate_channels = n_intermediate_channels
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.padding_size = padding_size
        self.fo_out_channels = fo_out_channels
        self.n_resblocks = n_resblocks

        # initial fin
        self.fin = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * fi_in_channels,
                out_channels=n_intermediate_channels,
                kernel_size=kernel_size,
                stride=stride_size,
                padding=padding_size
            ),
            nn.ReLU(),
        )

        # initial frecurrent
        self.lstm = ConvLstm(in_channels=n_intermediate_channels, kernel_size=kernel_size)

        # initial fres
        self.res = ResBlock(
            n_channels=n_intermediate_channels,
            kernel_size=kernel_size,
            stride_size=stride_size,
            padding_size=padding_size,
        )

        # initial se
        self.se = SELayer(channel=n_intermediate_channels,
                          reduction=reduction)

        # initial fout
        self.fout = nn.Conv2d(
            in_channels=n_intermediate_channels,
            out_channels=fo_out_channels,
            kernel_size=kernel_size,
            stride=stride_size,
            padding=padding_size
        )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        x = input
        h = torch.zeros((batch_size, self.n_intermediate_channels, row, col))
        c = torch.zeros((batch_size, self.n_intermediate_channels, row, col))

        for i in range(self.iteration):
            x = torch.cat((input, x), 1)

            # fin layer
            x = self.fin(x)
            x = torch.cat([x, h], 1)

            # frecurrent layer
            h, c = self.lstm(x, c)
            x = h

            # se_resblock layer
            for j in range(self.n_resblocks):
                x = self.res(x)
                x = self.se(x)

            # fout layer
            x = self.fout(x)

        x = input + x

        return x


if __name__ == "__main__":
    batch_size, channels, height, width = 10, 3, 5, 5
    x = torch.rand(batch_size, channels, height, width)
    model = PReNet_r(6)
    y = model(x)
    print(y.shape)