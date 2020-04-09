import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_3_ResNet


#Model5 should somehow resemble resnet50. memorywise this is not the best idea.
class Model5(nn.Module):

    def __init__(self):
        super(Model5, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.block_1 = nn.ModuleList([ResidualBlock_3_ResNet(2, 64, 256),
                                      ResidualBlock_3_ResNet(256, 64, 256),
                                      ResidualBlock_3_ResNet(256, 64, 256)])

        self.block_2 = nn.ModuleList([ResidualBlock_3_ResNet(256, 128, 512,
                                                             stride=2, input_kernel_size=5, input_padding=2),
                                      ResidualBlock_3_ResNet(512, 128, 512),
                                      ResidualBlock_3_ResNet(512, 128, 512),
                                      ResidualBlock_3_ResNet(512, 128, 512)])

        # this already is one resolution level too low
        self.block_3 = nn.ModuleList([ResidualBlock_3_ResNet(512, 256, 1024),#, stride=2, input_kernel_size=5, input_padding=2),
                                      ResidualBlock_3_ResNet(1024, 128, 1024),
                                      ResidualBlock_3_ResNet(1024, 128, 1024),
                                      ResidualBlock_3_ResNet(1024, 128, 1024),
                                      ResidualBlock_3_ResNet(1024, 128, 1024),
                                      ResidualBlock_3_ResNet(1024, 128, 1024)])

        self.output = nn.Conv2d(1024, 2, 1)
        # receptive field here should be about 32

    def forward(self, x):

        for res_block in self.block_1:
            x = res_block(x)

        for res_block in self.block_2:
            x = res_block(x)

        x_latent = x

        for res_block in self.block_3:
            x = res_block(x)

        return x, x_latent


