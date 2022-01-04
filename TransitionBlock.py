import torch.nn
import torch.nn.functional as F


class TransitionBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, pool_kernel_size=2):
        super(TransitionBlock, self).__init__()
        self.bn_1 = torch.nn.BatchNorm2d(num_features=in_channels)
        self.conv_1_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                        bias=False)
        self.avg_pooling = torch.nn.AvgPool2d(kernel_size=pool_kernel_size)

    def forward(self, x):
        x = F.relu(self.bn_1(x))
        x = self.conv_1_1(x)
        x = self.avg_pooling(x)
        return x
    