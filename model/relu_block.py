import torch
import torch.nn as nn
import torch.nn.functional as F




class ReluBlock(nn.Module):
    def __init__(self, channels, kernel_size, padding, layers=2):
        super(ReluBlock, self).__init__()
        self.convs = [nn.Conv2d(channels, channels, kernel_size, padding=padding, padding_mode='same')] * layers

    def forward(self, x):
        x_identity = x
        for conv in self.convs[:-1]:
            x = F.leaky_relu(conv(x))

        x = F.leaky_relu(x_identity + self.convs[-1](x))

        return x

