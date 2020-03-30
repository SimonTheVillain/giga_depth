import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock




class Model1(nn.Module):

    def __init__(self):
        super(Model1, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_start = nn.Conv2d(2, 64, 3, padding=1, padding_mode='zeros')
        self.resi_block1 = ResidualBlock(65, 3, 1)
        self.resi_block2 = ResidualBlock(66, 3, 1)
        #self.conv1 = nn.Conv2d(66, 63, 3, padding=1, padding_mode='same')
        self.resi_block3 = ResidualBlock(67, 3, 1)
        self.resi_block4 = ResidualBlock(68, 3, 1)
        #pool 2
        self.conv_end_1 = nn.Conv2d(69, 126, 5, padding=2, padding_mode='same')
        self.resi_block10 = ResidualBlock(127, 3, 1)
        self.resi_block11 = ResidualBlock(128, 3, 1)
        #concatenate
        self.conv_end_2 = nn.Conv2d(128, 2, 3, padding=1, padding_mode='same')
        # receptive field here should be about 32

    def forward(self, x):
        y = x[:, [1], :, :]
        y_half = F.interpolate(y, scale_factor=0.5)
        ### LAYER 0
        x = F.leaky_relu(self.conv_start(x))
        x = torch.cat((x, y), 1)

        x = self.resi_block1(x)
        x = torch.cat((x, y), 1)

        x = self.resi_block2(x)
        ### LAYER 1
        x = F.max_pool2d(x, 2) #TODO: maybe is a convolution with a certain stride better here!
        x = torch.cat((x, y_half), 1)

        x = self.resi_block3(x)
        x = torch.cat((x, y_half), 1)
        x_latent = x

        x = self.resi_block4(x)
        x = torch.cat((x, y_half), 1)

        x = F.leaky_relu(self.conv_end_1(x))
        x = torch.cat((x, y_half), 1)

        x = self.resi_block10(x)
        x = torch.cat((x, y_half), 1)#TODO: remove this! it is extremely stupid!!!!!!!!!!
        x = self.resi_block11(x)
        x_latent = torch.cat((x, x_latent), 1)

        x = self.conv_end_2(x)
        return x, x_latent


