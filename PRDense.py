# 输入层+残差密连块+输出层
import torch.nn as nn
from ResidualDenseBlock import *


class PRDense(nn.Module):
    def __init__(self, fi_in_channels=3, n_intermediate_channels=48, num_layers_m=5,
                 fo_out_channels=3, growth_rate_k=12, recurrent_iter=6):
        super(PRDense, self).__init__()
        self.fi_in_channels = fi_in_channels
        self.n_intermediate_channels = n_intermediate_channels
        self.num_layers_m = num_layers_m
        self.fo_out_channels=fo_out_channels
        self.growth_rate_k = growth_rate_k
        self.iteration = recurrent_iter

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
        for i in range(self.iteration):
            # 输入：两张图级联
            x = torch.cat((input, x), dim=1)

            # fin layer
            x = self.fin(x)

            # fred layer
            for j in range(self.num_layers_m):
                x = self.red(x)

            # fout layer
            x = self.fout(x)
            x = input + x
        return x