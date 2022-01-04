import torch.nn
from DenseBlock import *
from TransitionBlock import *


GROWTH_RATE_MULTIPLIER = 4


class ResidualDenseBlock(torch.nn.Module):

    def __init__(self, in_channels, num_layers_m, growth_rate_k):
        super(ResidualDenseBlock, self).__init__()
        self.down_sample_fn = torch.nn.AvgPool2d(kernel_size=2)
        self.dense_block = DenseBlock(in_channels=in_channels, num_layers_m=num_layers_m, growth_rate_k=growth_rate_k)
        dense_channels_out = in_channels + num_layers_m * growth_rate_k
        self.transition_block = TransitionBlock(in_channels=dense_channels_out,
                                                out_channels=growth_rate_k * GROWTH_RATE_MULTIPLIER)

    def forward(self, x):
        residual = self.down_sample_fn(x)
        x = self.dense_block(x)
        x = self.transition_block(x)
        x += residual
        return x