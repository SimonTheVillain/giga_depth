import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_shrink




class Model_Lines_3(nn.Module):

    def __init__(self):
        super(Model_Lines_3, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_start = nn.Conv2d(1, 64, 3, padding=0, padding_mode='none') #1
        self.resi_block1 = ResidualBlock_shrink(64, 3, 0, depadding=3) #3
        self.resi_block2 = ResidualBlock_shrink(64, 3, 0, depadding=3) #3
        #self.conv1 = nn.Conv2d(66, 63, 3, padding=1, padding_mode='same')
        self.resi_block3 = ResidualBlock_shrink(64, 3, 0, depadding=3) #3
        self.resi_block4 = ResidualBlock_shrink(64, 3, 0, depadding=3) #3
        #pool 2
        self.conv_end_1 = nn.Conv2d(64, 128, 5, padding=0, padding_mode='none') #2
        self.resi_block10 = ResidualBlock_shrink(128, 1, 0, depadding=0) #0
        self.resi_block11 = ResidualBlock_shrink(128, 1, 0, depadding=0) #0
        self.resi_block12 = ResidualBlock_shrink(128, 1, 0, depadding=0) #0
        self.resi_block13 = ResidualBlock_shrink(128, 1, 0, depadding=0) #0
        self.resi_block14 = ResidualBlock_shrink(128, 1, 0, depadding=0) #0
        self.resi_block15 = ResidualBlock_shrink(128, 1, 0, depadding=0) #0
        self.resi_block16 = ResidualBlock_shrink(128, 1, 0, depadding=0) #0
        self.resi_block17 = ResidualBlock_shrink(128, 1, 0, depadding=0) #0
        self.resi_block18 = ResidualBlock_shrink(128, 1, 0, depadding=0) #0
        #concatenate
        self.conv_end_2 = nn.Conv2d(128, 2, 1, padding=0, padding_mode='same') #0
        # receptive field here should be about 32

    def forward(self, x):
        ### LAYER 0
        x = F.leaky_relu(self.conv_start(x))

        x = self.resi_block1(x)

        x = self.resi_block2(x)

        x = self.resi_block3(x)
        #x_latent = x

        x = self.resi_block4(x)

        x = F.leaky_relu(self.conv_end_1(x))

        x = self.resi_block10(x)
        x = self.resi_block11(x)
        x = self.resi_block12(x)
        x = self.resi_block13(x)
        x = self.resi_block14(x)
        x = self.resi_block15(x)
        x = self.resi_block16(x)
        x = self.resi_block17(x)
        x = self.resi_block18(x)
        x_latent = x #torch.cat((x, x_latent), 1)

        x = self.conv_end_2(x)
        return x, x_latent


