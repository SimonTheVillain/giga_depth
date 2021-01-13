import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_shrink


# same as backbone v1 but with less weights and less computations.
class Backbone(nn.Module):
    def radius(self):
        return 15

    def __init__(self):
        super(Backbone, self).__init__()
        self.conv_start = nn.Conv2d(1, 16, 3, padding=(0, 1))  # 1
        self.conv_ds1 = nn.Conv2d(16, 32, 5, padding=(0, 2), stride=2, groups=1 + 0 * 16)  # + 2 = 3
        self.conv_1 = nn.Conv2d(32, 32, 3, padding=(0, 1), stride=1)  # + 2 x 1 = 5
        self.conv_2 = nn.Conv2d(32, 32, 3, padding=(0, 1), stride=1, groups=1 + 0 * 32)  # + 2 x 1 = 7
        self.conv_3 = nn.Conv2d(32, 32, 3, padding=(0, 1), stride=1)  # + 2 x 1 = 9
        self.conv_4 = nn.Conv2d(32, 64, 3, padding=(0, 1), stride=1, groups=1 + 0 * 64)  # + 2 x 1 = 11
        self.conv_5 = nn.Conv2d(64, 64, 3, padding=(0, 1), stride=1)  # + 2 x 1 = 13
        self.conv_6 = nn.Conv2d(64, 64, 3, padding=(0, 1), stride=1)  # + 2 x 1 = 15
        self.conv_7 = nn.Conv2d(64, 64, 3, padding=(0, 1), stride=1, groups=1 + 0 * 64)  # + 2 x 1 = 17

    def forward(self, x):
        x = F.leaky_relu(self.conv_start(x))
        x = F.leaky_relu(self.conv_ds1(x)) # downsample here
        x = F.leaky_relu(self.conv_1(x))
        x = F.leaky_relu(self.conv_2(x))
        x = F.leaky_relu(self.conv_3(x))
        x = F.leaky_relu(self.conv_4(x))
        x = F.leaky_relu(self.conv_5(x))
        x = F.leaky_relu(self.conv_6(x))
        x = F.leaky_relu(self.conv_7(x))

        return x


