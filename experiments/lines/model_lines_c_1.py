import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_shrink




class Model_Lines_C_1(nn.Module):

    def __init__(self):
        super(Model_Lines_C_1, self).__init__()
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
        #concatenate
        self.conv_end_2 = nn.Conv2d(128, 1025, 1, padding=0, padding_mode='same')
        self.conv_end_3 = nn.Conv2d(1025, 1025, 1, padding=0, padding_mode='same') #0
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

        x = F.leaky_relu(self.conv_end_2(x))
        x = F.leaky_relu(self.conv_end_3(x))
        mask = x[:, [1024], :, :]
        x = F.softmax(x[:, :-1, :, :], dim=1)
        #test = torch.sum(x, axis=1)
        #print(test)
        x = torch.cat((x, mask), 1)
        return x, x_latent


