import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_shrink


# same as the linewise cr8_no_residual_light. It seems like a good compromise
# no residuals are needed according to my experiment
# also it might be possible to go further down with computations but first lets check this out!

class Backbone(nn.Module):
    def radius(self):
        return 17

    def __init__(self):
        super(Backbone, self).__init__()
        # kernel
        self.conv_start = nn.Conv2d(1, 16, 3, padding=(0, 1))  # 1
        self.conv_1 = nn.Conv2d(16, 32, 3, padding=(0, 1))  # + 1 = 2
        self.conv_2 = nn.Conv2d(32, 32, 3, padding=(0, 1))  # + 1 = 3
        self.conv_3_down = nn.Conv2d(32, 64, 5, padding=(0, 2), stride=(2, 2))  # + 2 = 5
        # subsampled from here!
        self.conv_4 = nn.Conv2d(64, 64, 3, padding=(0, 1))  # + 1 * 2 = 7
        self.conv_5 = nn.Conv2d(64, 64, 3, padding=(0, 1))  # + 1 * 2 = 9
        self.conv_6 = nn.Conv2d(64, 64, 3, padding=(0, 1))  # + 1 * 2 = 11
        self.conv_7 = nn.Conv2d(64, 64, 3, padding=(0, 1))  # + 1 * 2 = 13
        self.conv_8 = nn.Conv2d(64, 64, 3, padding=(0, 1))  # + 1 * 2 = 15
        self.conv_9 = nn.Conv2d(64, 128, 3, padding=(0, 1))  # + 1 * 2 = 17

    def forward(self, x):
        x = F.leaky_relu(self.conv_start(x))
        x = F.leaky_relu(self.conv_1(x))
        x = F.leaky_relu(self.conv_2(x))
        x = F.leaky_relu(self.conv_3_down(x))
        x = F.leaky_relu(self.conv_4(x))
        x = F.leaky_relu(self.conv_5(x))
        x = F.leaky_relu(self.conv_6(x))
        x = F.leaky_relu(self.conv_7(x))
        x = F.leaky_relu(self.conv_8(x))
        x = F.leaky_relu(self.conv_9(x))
        return x


