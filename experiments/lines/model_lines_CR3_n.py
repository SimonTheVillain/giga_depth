import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_shrink




class Model_Lines_CR3_n(nn.Module):

    def __init__(self, classes):
        super(Model_Lines_CR3_n, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_start = nn.Conv2d(1, 128, 3, padding=0) #1
        self.resi_block1 = ResidualBlock_shrink(128, 3, 0, depadding=3) #3
        self.resi_block2 = ResidualBlock_shrink(128, 3, 0, depadding=3) #3
        #self.conv1 = nn.Conv2d(66, 63, 3, padding=1, padding_mode='same')
        self.resi_block3 = ResidualBlock_shrink(128, 3, 0, depadding=3) #3
        self.resi_block4 = ResidualBlock_shrink(128, 3, 0, depadding=3) #3
        #pool 2
        self.conv_end_1 = nn.Conv2d(128, 256, 5, padding=0) #2
        self.resi_block10 = ResidualBlock_shrink(256, 1, 0, depadding=0) #0
        self.resi_block11 = ResidualBlock_shrink(256, 1, 0, depadding=0) #0
        #concatenate
        self.resi_block_end_1c = ResidualBlock_shrink(256, 1, 0, depadding=0) #0
        self.conv_end_2_c = nn.Conv2d(256, 1024, 1, padding=0, padding_mode='replicate')
        self.conv_end_3_c = nn.Conv2d(1024, classes+1, 1, padding=0, padding_mode='replicate') #0

        #two similar layers for the regression part of the output
        self.resi_block_end_1r = ResidualBlock_shrink(256, 1, 0, depadding=0) #0
        self.conv_end_2_r = nn.Conv2d(256, 1024, 1, padding=0, padding_mode='replicate')
        self.conv_end_3_r = nn.Conv2d(1024, classes, 1, padding=0, padding_mode='replicate') #0
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
        x_latent = x #torch.cat((x, x_latent), 1)

        x_split_point = x
        x = self.resi_block_end_1c(x_split_point)
        x = F.leaky_relu(self.conv_end_2_c(x))
        x = F.leaky_relu(self.conv_end_3_c(x))

        mask = x[:, [-1], :, :]
        x = F.softmax(x[:, :-1, :, :], dim=1)
        classes = x

        x = self.resi_block_end_1r(x_split_point)
        x = F.leaky_relu(self.conv_end_2_r(x))
        regressions = F.leaky_relu(self.conv_end_3_r(x))

        #test = torch.sum(x, axis=1)
        #print(test)
        return classes, regressions, mask, x_latent


