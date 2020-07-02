import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_shrink




class Model_Lines_C_R_n(nn.Module):

    def __init__(self, classes, other=None):
        super(Model_Lines_C_R_n, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_start = nn.Conv2d(1, 64, 3, padding=0) #1
        self.resi_block1 = ResidualBlock_shrink(64, 3, 0, depadding=3) #3
        self.resi_block2 = ResidualBlock_shrink(64, 3, 0, depadding=3) #3
        #self.conv1 = nn.Conv2d(66, 63, 3, padding=1, padding_mode='same')
        self.resi_block3 = ResidualBlock_shrink(64, 3, 0, depadding=3) #3
        self.resi_block4 = ResidualBlock_shrink(64, 3, 0, depadding=3) #3
        #pool 2
        self.conv_end_1 = nn.Conv2d(64, 128, 5, padding=0) #2
        self.resi_block10 = ResidualBlock_shrink(128, 1, 0, depadding=0) #0
        self.resi_block11 = ResidualBlock_shrink(128, 1, 0, depadding=0) #0
        #concatenate
        self.conv_end_2_c = nn.Conv2d(128, 1024, 1, padding=0, padding_mode='replicate')
        self.conv_end_3_c = nn.Conv2d(1024, classes+1, 1, padding=0, padding_mode='replicate') #0

        #two similar layers for the regression part of the output
        self.conv_end_2_r = nn.Conv2d(128, 1024, 1, padding=0, padding_mode='replicate')
        self.conv_end_3_r = nn.Conv2d(1024, classes, 1, padding=0, padding_mode='replicate') #0
        # receptive field here should be about 32

        if other is not None:
            #copying over weights from the other class
            self.conv_start = other.conv_start
            self.resi_block1 = other.resi_block1
            self.resi_block2 = other.resi_block2
            self.resi_block3 = other.resi_block3
            self.resi_block4 = other.resi_block4
            self.conv_end_1 = other.conv_end_1
            self.resi_block10 = other.resi_block10
            self.resi_block11 = other.resi_block11

            self.conv_end_2_c = other.conv_end_2_c
            self.conv_end_2_r = other.conv_end_2_r
            # all except for the output since the class count might be different

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
        x_latent = x #torch.cat((x, x_latent), 1)

        x_split_point = x
        x = F.leaky_relu(self.conv_end_2_c(x))
        x = F.leaky_relu(self.conv_end_3_c(x))

        mask = x[:, [-1], :, :]
        x = F.softmax(x[:, :-1, :, :], dim=1)
        classes = x

        x = F.leaky_relu(self.conv_end_2_r(x_split_point))
        regressions = F.leaky_relu(self.conv_end_3_r(x))

        #test = torch.sum(x, axis=1)
        #print(test)
        return classes, regressions, mask, x_latent

