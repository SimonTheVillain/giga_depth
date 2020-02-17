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


class FlatNetHalf(nn.Module):

    def __init__(self):
        super(FlatNetHalf, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_start = nn.Conv2d(2, 64, 3, padding=1, padding_mode='same')
        self.relublock1 = ReluBlock(65, 3, 1)
        self.relublock2 = ReluBlock(66, 3, 1)
        #self.conv1 = nn.Conv2d(66, 63, 3, padding=1, padding_mode='same')
        self.relublock3 = ReluBlock(67, 3, 1)
        self.relublock4 = ReluBlock(68, 3, 1)
        #pool 2
        self.conv_end_1 = nn.Conv2d(69, 126, 5, padding=2, padding_mode='same')
        self.relublock10 = ReluBlock(127, 3, 1)
        #concatenate
        self.conv_end_2 = nn.Conv2d(128, 2, 1)  # , padding=0, padding_mode='same')
        # receptive field here should be about 32

    def forward(self, x):
        y = x[:, [1], :, :]
        y_half = F.interpolate(y, scale_factor=0.5)
        ### LAYER 0
        x = F.leaky_relu(self.conv_start(x))
        x = torch.cat((x, y), 1)

        x = self.relublock1(x)
        x = torch.cat((x, y), 1)

        x = self.relublock2(x)
        x = torch.cat((x, y), 1)

        #x = F.leaky_relu(self.conv1(x))
        #x = torch.cat((x, y_half), 1)

        x = self.relublock3(x)
        x = torch.cat((x, y), 1)

        x = self.relublock4(x)
        x = F.max_pool2d(x, 2) #TODO: maybe is a convolution with a certain stride better here!
        x = torch.cat((x, y_half), 1)

        x = F.leaky_relu(self.conv_end_1(x))
        x = torch.cat((x, y_half), 1)

        x = self.relublock10(x)
        x = torch.cat((x, y_half), 1)


        x = self.conv_end_2(x)
        return x


