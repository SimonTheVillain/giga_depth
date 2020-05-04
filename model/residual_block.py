import torch
import torch.nn as nn
import torch.nn.functional as F




class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, padding, layers=2):
        super(ResidualBlock, self).__init__()
        self.convs = \
            nn.ModuleList([nn.Conv2d(channels, channels, kernel_size, padding=padding, padding_mode='same')] * layers)

    def forward(self, x):
        x_identity = x
        for conv in self.convs[:-1]:
            x = F.leaky_relu(conv(x))

        x = F.leaky_relu(x_identity + self.convs[-1](x))

        return x

class ResidualBlock_shrink(nn.Module):
    def __init__(self, channels, kernel_size, padding, layers=3, depadding=0):
        super(ResidualBlock_shrink, self).__init__()
        self.depadding = depadding
        self.convs = \
            nn.ModuleList(
                [nn.Conv2d(channels, channels, kernel_size, padding=padding, padding_mode='replicate')] * layers)

    def forward(self, x):
        x_identity = x
        for conv in self.convs[:-1]:
            x = F.leaky_relu(conv(x))
        d = self.depadding
        if d != 0:
            x_identity = x_identity[:, :, d: -d, d: -d]
        x = F.leaky_relu(x_identity + self.convs[-1](x))

        return x

class ResidualBlock_3_ResNet(nn.Module):
        def __init__(self, channels_in, channels_inter, channels_out,
                     stride=1, input_kernel_size=1, input_padding=0, batch_normalization=True):
            super(ResidualBlock_3_ResNet, self).__init__()
            self.convs = \
                nn.ModuleList( [nn.Conv2d(channels_in, channels_inter, input_kernel_size, stride=stride,
                                          padding=input_padding, padding_mode='same'),
                                nn.Conv2d(channels_inter, channels_inter, 3, padding=1, padding_mode='same'),
                                nn.Conv2d(channels_inter, channels_out, 1, padding=0, padding_mode='same')])
            if batch_normalization:
                self.batch_norms = \
                    nn.ModuleList([nn.BatchNorm2d(channels_inter),
                                   nn.BatchNorm2d(channels_inter),
                                   nn.BatchNorm2d(channels_out)])
            self.batch_normalization = batch_normalization

        def forward(self, x):
            x_identity = x
            for ind in range(0, 2):
                x = self.convs[ind](x)
                if self.batch_normalization:
                    x = self.batch_norms[ind](x)
                F.leaky_relu(x)
            x = self.convs[-1](x)
            if self.batch_normalization:
                x = self.batch_norms[-1](x)
            x[:, 0:x_identity.shape[1], :, :] += x_identity # hope this works (for sure doesn't if output shape is smaller than input shape)
            x = F.leaky_relu(x)

            return x


