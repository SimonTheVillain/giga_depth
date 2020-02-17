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


class FlatNet(nn.Module):

    def __init__(self):
        super(FlatNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_start = nn.Conv2d(2, 32, 3, padding=1, padding_mode='same')
        self.relublock1 = ReluBlock(32, 3, 1)
        self.relublock2 = ReluBlock(33, 3, 1)
        self.relublock3 = ReluBlock(34, 3, 1)
        self.relublock4 = ReluBlock(35, 3, 1)
        self.relublock5 = ReluBlock(36, 3, 1)
        self.relublock6 = ReluBlock(37, 3, 1)
        self.relublock7 = ReluBlock(38, 3, 1)
        self.relublock8 = ReluBlock(39, 3, 1)
        self.relublock9 = ReluBlock(40, 3, 1)
        self.conv_end = nn.Conv2d(41, 2, 1)#, padding=0, padding_mode='same')
        #receptive field here should be about 32

    def forward(self, x):
        y = x[:, [1], :, :]
        ### LAYER 0
        x = F.leaky_relu(self.conv_start(x))

        x = self.relublock1(x)
        x = torch.cat((x, y), 1)

        x = self.relublock2(x)
        x = torch.cat((x, y), 1)

        x = self.relublock3(x)
        x = torch.cat((x, y), 1)

        x = self.relublock4(x)
        x = torch.cat((x, y), 1)

        x = self.relublock5(x)
        x = torch.cat((x, y), 1)

        x = self.relublock6(x)
        x = torch.cat((x, y), 1)

        x = self.relublock7(x)
        x = torch.cat((x, y), 1)

        x = self.relublock8(x)
        x = torch.cat((x, y), 1)

        x = self.relublock9(x)
        x = torch.cat((x, y), 1)

        x = self.conv_end(x)
        return x


