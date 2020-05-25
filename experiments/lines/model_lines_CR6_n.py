import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_shrink



#640×480×4×(1024+513+ 128×7+4×(64×7 + 1)) would result in a bit below 6GB of video memory.
# (conv_ch_up_1 would have a stride of 2)
class Model_Lines_CR6_n(nn.Module):

    def __init__(self, classes):
        self.classes = classes
        super(Model_Lines_CR6_n, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_start = nn.Conv2d(1, 64, 3, padding=0) #1
        self.resi_block1 = ResidualBlock_shrink(64, 3, 0, depadding=3) #3
        #self.conv1 = nn.Conv2d(66, 63, 3, padding=1, padding_mode='same')
        self.resi_block2 = ResidualBlock_shrink(64, 3, 0, depadding=3) #3
        self.conv_ch_up_1 = nn.Conv2d(64, 128, 5, padding=0) #2 Here we would need to downsample (stride 2)
        self.resi_block3 = ResidualBlock_shrink(128, 3, 0, depadding=3) #3
        self.resi_block4 = ResidualBlock_shrink(128, 3, 0, depadding=3) #3
        self.conv_end_1 = nn.Conv2d(128, 1024, 1, padding=0, padding_mode='replicate')
        self.conv_end_2 = nn.Conv2d(1024, classes*2+1, 1, padding=0, padding_mode='replicate')

    def forward(self, x):
        ### LAYER 0
        x = F.leaky_relu(self.conv_start(x))

        x = self.resi_block1(x)
        x = self.resi_block2(x)
        x = F.leaky_relu(self.conv_ch_up_1(x))
        x = self.resi_block3(x)
        x = self.resi_block4(x)
        x_latent = x
        x = F.leaky_relu(self.conv_end_1(x))
        x = F.leaky_relu(self.conv_end_2(x))
        classes = F.softmax(x[:, 0:self.classes, :, :], dim=1)
        regressions = x[:, self.classes:(2 * self.classes), :, :]
        mask = x[:, [-1], :, :]
        return classes, regressions, mask, x_latent


