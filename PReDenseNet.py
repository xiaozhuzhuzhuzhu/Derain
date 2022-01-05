# 输入层+残差密连块+LSTM+输出层
import torch
import torch.nn as nn
from ResidualDenseBlock import *
from convlstm import ConvLstm


class PRDenseNet(nn.Module):
    def __init__(self, fi_in_channels=3, n_intermediate_channels=48, num_layers_m=5,
                 fo_out_channels=3, growth_rate_k=12, recurrent_iter=6, n_red=3):
        super(PRDenseNet, self).__init__()
        self.fi_in_channels = fi_in_channels
        self.n_intermediate_channels = n_intermediate_channels
        self.num_layers_m = num_layers_m
        self.fo_out_channels=fo_out_channels
        self.growth_rate_k = growth_rate_k
        self.iteration = recurrent_iter
        self.n_red = n_red

        # initial fin
        self.fin = nn.Sequential(
            nn.Conv2d(in_channels=2*fi_in_channels,
                      out_channels=n_intermediate_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.ReLU(),
        )

        # initial fred
        self.red = ResidualDenseBlock(in_channels=n_intermediate_channels,
                                      num_layers_m=num_layers_m, growth_rate_k=growth_rate_k)

        # initial frecurrent
        self.lstm = ConvLstm(in_channels=n_intermediate_channels, kernel_size=3)

        # initial fout
        self.fout = nn.Conv2d(
            in_channels=n_intermediate_channels,
            out_channels=fo_out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, input):
        x = input
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        h = torch.zeros((batch_size, self.n_intermediate_channels, row, col))
        c = torch.zeros((batch_size, self.n_intermediate_channels, row, col))

        for i in range(self.iteration):
            # 输入：两张图级联
            x = torch.cat((input, x), dim=1)

            # fin layer
            x = self.fin(x)

            # fred layer
            for j in range(self.n_red):
                x = self.red(x)

            # frecurrent layer
            x = torch.cat([x, h], 1)
            h, c = self.lstm(x, c)
            x = h

            # fout layer
            x = self.fout(x)
            x = input + x
        return x