import torch
from DenseLayer import *


class DenseBlock(torch.nn.Module):

    def __init__(self, in_channels, num_layers_m, growth_rate_k):
        super(DenseBlock, self).__init__()
        self.dense_layers = torch.nn.ModuleList()
        channels = in_channels
        for i in range(num_layers_m):
            self.dense_layers.append(DenseLayer(in_channels=channels, out_channels=growth_rate_k))
            channels += growth_rate_k

    def forward(self, x):
        cat_input = x
        for dense_layer in self.dense_layers:
            layer_output = dense_layer(cat_input)
            cat_input = torch.cat([cat_input, layer_output], dim=1)
        return cat_input
    