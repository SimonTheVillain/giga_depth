import torch
import torch.nn as nn
import torch.nn.functional as F




class ReluBlock(nn.Module):
    def __init__(self, channels, kernel_size, padding):
        super(ReluBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, padding_mode='same')
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, padding_mode='same')

    def forward(self, x):
        x_identity = x
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(x_identity + self.conv2(x))

        return x


class FlatNetHalfNOY(nn.Module):

    def __init__(self):
        super(FlatNetHalfNOY, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_start = nn.Conv2d(1, 64, 3, padding=1, padding_mode='same')
        self.relublock1 = ReluBlock(64, 3, 1)
        self.relublock2 = ReluBlock(64, 3, 1)
        #self.conv1 = nn.Conv2d(66, 63, 3, padding=1, padding_mode='same')
        self.relublock3 = ReluBlock(64, 3, 1)
        self.relublock4 = ReluBlock(64, 3, 1)
        #pool 2
        self.conv_end_1 = nn.Conv2d(64, 128, 5, padding=2, padding_mode='same')
        self.relublock10 = ReluBlock(128, 3, 1)
        #concatenate
        self.conv_end_2 = nn.Conv2d(128, 2, 1)  # , padding=0, padding_mode='same')
        # receptive field here should be about 32

    def forward(self, x):
        x = x[:, [0], :, :]
        ### LAYER 0
        x = F.leaky_relu(self.conv_start(x))
        x = self.relublock1(x)
        x = self.relublock2(x)
        x = self.relublock3(x)
        x = self.relublock4(x)
        ### LAYER 1
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.conv_end_1(x))
        x = self.relublock10(x)
        x = self.conv_end_2(x)
        return x


