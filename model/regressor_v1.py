import torch
import torch.nn as nn
import torch.nn.functional as F
import cuda_cond_mul.cond_mul.CondMul as CondMul
from model.residual_block import ResidualBlock_shrink


#https://gist.github.com/sonots/5abc0bccec2010ac69ff74788b265086
# same as CR10_4 but with better per weight
class Regressor_v1(nn.Module):
    @staticmethod
    def padding():
        return 0

    def __init__(self, classes=128, ):
        super(Regressor_v1, self).__init__()
        #per line weights for classes
        self.classifier = nn.Conv2d(128 * half_height, self.classes * half_height, 1, padding=0, groups=half_height)

    def copy_backbone(self, other):
        if self.shallow != other.shallow:
            return

        self.conv_start.weight.data = other.conv_start.weight.data
        self.conv_start.bias.data = other.conv_start.bias.data
        self.conv_ds1.weight.data = other.conv_ds1.weight.data
        self.conv_ds1.bias.data = other.conv_ds1.bias.data
        self.conv_1.weight.data = other.conv_1.weight.data
        self.conv_1.bias.data = other.conv_1.bias.data
        self.conv_2.weight.data = other.conv_2.weight.data
        self.conv_2.bias.data = other.conv_2.bias.data
        self.conv_3.weight.data = other.conv_3.weight.data
        self.conv_3.bias.data = other.conv_3.bias.data
        self.conv_4.weight.data = other.conv_4.weight.data
        self.conv_4.bias.data = other.conv_4.bias.data
        self.conv_5.weight.data = other.conv_5.weight.data
        self.conv_5.bias.data = other.conv_5.bias.data
        self.conv_6.weight.data = other.conv_6.weight.data
        self.conv_6.bias.data = other.conv_6.bias.data

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


