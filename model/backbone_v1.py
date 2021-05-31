import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_shrink


# same as CR10_4 but with better per weight
class Backbone1(nn.Module):
    def __init__(self):
        super(Backbone1, self).__init__()
        self.conv_start = nn.Conv2d(1, 16, 3, padding=1)  # 1
        self.conv_ds1 = nn.Conv2d(16, 32, 5, padding=2, stride=2, groups=1 + 0 * 16)  # + 2 = 3
        self.conv_1 = nn.Conv2d(32, 32, 3, padding=1, stride=1)  # + 2 x 1 = 5
        self.conv_2 = nn.Conv2d(32, 32, 3, padding=1, stride=1, groups=1 + 0 * 32)  # + 2 x 1 = 7
        self.conv_3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)  # + 2 x 1 = 9
        self.conv_4 = nn.Conv2d(64, 64, 3, padding=1, stride=1, groups=1 + 0 * 64)  # + 2 x 1 = 11
        self.conv_5 = nn.Conv2d(64, 128, 3, padding=1, stride=1)  # + 2 x 1 = 13
        self.conv_6 = nn.Conv2d(128, 128, 3, padding=1, stride=1, groups=1 + 0 * 128)  # + 2 x 1 = 17


    def copy_backbone(self, other):
        if self.shallow != other.shallow:
            return

        self.conv_start.weight.data = other.conv_start.weight.serialized_data
        self.conv_start.bias.data = other.conv_start.bias.serialized_data
        self.conv_ds1.weight.data = other.conv_ds1.weight.serialized_data
        self.conv_ds1.bias.data = other.conv_ds1.bias.serialized_data
        self.conv_1.weight.data = other.conv_1.weight.serialized_data
        self.conv_1.bias.data = other.conv_1.bias.serialized_data
        self.conv_2.weight.data = other.conv_2.weight.serialized_data
        self.conv_2.bias.data = other.conv_2.bias.serialized_data
        self.conv_3.weight.data = other.conv_3.weight.serialized_data
        self.conv_3.bias.data = other.conv_3.bias.serialized_data
        self.conv_4.weight.data = other.conv_4.weight.serialized_data
        self.conv_4.bias.data = other.conv_4.bias.serialized_data
        self.conv_5.weight.data = other.conv_5.weight.serialized_data
        self.conv_5.bias.data = other.conv_5.bias.serialized_data
        self.conv_6.weight.data = other.conv_6.weight.serialized_data
        self.conv_6.bias.data = other.conv_6.bias.serialized_data

    def forward(self, x):

        x = F.leaky_relu(self.conv_start(x))
        x = F.leaky_relu(self.conv_ds1(x)) # downsample here
        x = F.leaky_relu(self.conv_1(x))
        x = F.leaky_relu(self.conv_2(x))
        x = F.leaky_relu(self.conv_3(x))
        x = F.leaky_relu(self.conv_4(x))
        x = F.leaky_relu(self.conv_5(x))
        x = F.leaky_relu(self.conv_6(x))

        return x


