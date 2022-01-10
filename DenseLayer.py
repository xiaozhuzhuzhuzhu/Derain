import torch.nn
import torch.nn.functional as F


class DenseLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        # self.bn_1 = torch.nn.BatchNorm2d(num_features=in_channels)
        self.conv_1_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        # self.bn_2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.conv_3_3 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                        padding=1, bias=False)

    def forward(self, x):
        # x = F.relu(self.bn_1(x))
        x = F.relu(x)
        x = self.conv_1_1(x)
        # x = F.relu(self.bn_2(x))
        x = F.relu(x)
        x = self.conv_3_3(x)
        return x


