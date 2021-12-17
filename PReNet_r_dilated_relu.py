import torch
import torch.nn as nn
from resblock import ResBlockByWeight
from convlstm import ConvLstm
import numpy as np


def _init_conv_w_b(w, b):
    nn.init.kaiming_normal_(w, a=np.sqrt(5))
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
    bound = 1 / np.sqrt(fan_in)
    nn.init.uniform_(b, -bound, bound)


class PReNet_r(nn.Module):
    def __init__(self, recurrent_iter, fi_in_channels=3,
                 n_intermediate_channels=32, kernel_size=3, stride_size=1,
                 padding_size=1, fo_out_channels=3, res_dilations=(1, 1, 1, 1, 1)):
        super(PReNet_r, self).__init__()
        self.iteration = recurrent_iter
        self.fi_in_channels = fi_in_channels
        self.n_intermediate_channels = n_intermediate_channels
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.padding_size = padding_size
        self.fo_out_channels = fo_out_channels

        # initial fin
        self.fin = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * fi_in_channels,
                out_channels=n_intermediate_channels,
                kernel_size=kernel_size,
                stride=stride_size,
                padding=padding_size,
            ),
            nn.ReLU(),
        )

        # initial frecurrent
        self.lstm = ConvLstm(in_channels=n_intermediate_channels, kernel_size=kernel_size)

        # initial fres
        self.resblocks = torch.nn.ModuleList([
            ResBlockByWeight(stride_size=stride_size, dilation=d) for d in res_dilations
        ])

        w1 = torch.Tensor(n_intermediate_channels, n_intermediate_channels, kernel_size, kernel_size)
        w2 = torch.Tensor(n_intermediate_channels, n_intermediate_channels, kernel_size, kernel_size)
        b1 = torch.Tensor(n_intermediate_channels)
        b2 = torch.Tensor(n_intermediate_channels)

        _init_conv_w_b(w1, b1)
        _init_conv_w_b(w2, b2)

        self.res_w1 = torch.nn.Parameter(w1, requires_grad=True)
        self.res_w2 = torch.nn.Parameter(w2, requires_grad=True)
        self.res_b1 = torch.nn.Parameter(b1, requires_grad=True)
        self.res_b2 = torch.nn.Parameter(b2, requires_grad=True)


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

            # dilated-resblock layer
            for res in self.resblocks:  # type: ResBlockByWeight
                x = res(x, self.res_w1, self.res_w2, self.res_b1, self.res_b2)

            # fout layer
            x = self.fout(x)

            # add relu
            x = torch.relu(x)

            x = input + x
            # x = torch.relu(x)

        return x


if __name__ == "__main__":
    batch_size, channels, height, width = 10, 3, 5, 5
    x = torch.rand(batch_size, channels, height, width)
    model = PReNet_r(6)
    y = model(x)

    print(y.min().item(), y.max().item())
