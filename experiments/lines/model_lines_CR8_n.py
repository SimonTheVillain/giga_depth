import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_shrink, ResidualBlock_vshrink


#don't trust any of these calculations in the layers we had before...
# (conv_ch_up_1 would have a stride of 2)
class Model_Lines_CR8_n(nn.Module):

    def __init__(self, classes):
        self.classes = classes
        super(Model_Lines_CR8_n, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_start = nn.Conv2d(1, 32, 3, padding=0) #1
        self.resi_block1 = ResidualBlock_shrink(32, 3, 0, depadding=3) #3
        #self.conv1 = nn.Conv2d(66, 63, 3, padding=1, padding_mode='same')
        self.resi_block2 = ResidualBlock_shrink(32, 3, 0, depadding=3) #3
        self.conv_ch_up_1 = nn.Conv2d(32, 64, 5, padding=0) #2 here stride 2 subsampling
        self.resi_block3 = ResidualBlock_shrink(64, 3, 0, depadding=3) #3
        self.conv_end_1 = nn.Conv2d(64, 128, 7, padding=0, padding_mode='replicate') #3 or here
        # if this does not work... add one more 1x1 convolutional layer here
        self.conv_end_2 = nn.Conv2d(128, 512, 1, padding=0, padding_mode='replicate')
        self.conv_end_3 = nn.Conv2d(512, classes*2+1, 1, padding=0, padding_mode='replicate')

    def forward(self, x):
        ### LAYER 0
        x = F.leaky_relu(self.conv_start(x))

        x = self.resi_block1(x)
        x = self.resi_block2(x)
        x = F.leaky_relu(self.conv_ch_up_1(x))
        x = self.resi_block3(x)
        x_latent = x
        x = F.leaky_relu(self.conv_end_1(x))
        x = F.leaky_relu(self.conv_end_2(x))
        x = F.leaky_relu(self.conv_end_3(x))
        classes = F.softmax(x[:, 0:self.classes, :, :], dim=1)
        regressions = x[:, self.classes:(2 * self.classes), :, :]
        mask = x[:, [-1], :, :]
        return classes, regressions, mask, x_latent

# CR8 backbone!!!
class CR8_bb(nn.Module):

    def __init__(self, classes):
        self.classes = classes
        super(CR8_bb, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_start = nn.Conv2d(1, 32, 3, padding=(0, 1)) #1
        self.resi_block1 = ResidualBlock_vshrink(32, 3, padding=(0, 1), depadding=3) # + 3 = 4
        self.resi_block2 = ResidualBlock_vshrink(32, 3, padding=(0, 1), depadding=3) # + 3 = 7
        self.conv_ch_up_1 = nn.Conv2d(32, 64, 5, padding=(0, 2), stride=(2, 2)) # + 2 = 9
        # subsampled beginning here!
        self.resi_block3 = ResidualBlock_vshrink(64, 3, padding=(0, 1), depadding=3) # + 2 * 3 = 15
        self.conv_end_1 = nn.Conv2d(64, 128, 3, padding=(0, 1)) # + 2 * 2 = 17
        # if this does not work... add one more 1x1 convolutional layer here
        self.conv_end_2 = nn.Conv2d(128, 512, 1, padding=0)

    def forward(self, x):
        ### LAYER 0
        #print(x.shape)
        x = F.leaky_relu(self.conv_start(x))

        #print(x.shape)
        x = self.resi_block1(x)

        #print(x.shape)
        x = self.resi_block2(x)
        #print(x.shape)
        x = F.leaky_relu(self.conv_ch_up_1(x))
        #print(x.shape)
        x = self.resi_block3(x)
        #print(x.shape)
        x = F.leaky_relu(self.conv_end_1(x))
        #print(x.shape)
        x = F.leaky_relu(self.conv_end_2(x))
        #print(x.shape)
        return x

# CR8 regressor!!!
class CR8_reg(nn.Module):

    def __init__(self, classes):
        self.classes = classes
        super(CR8_reg, self).__init__()
        self.conv_end_3 = nn.Conv2d(512, classes*2+1, 1, padding=0, padding_mode='replicate')

    def forward(self, x, x_gt=None, mask_gt=None):
        ### LAYER 0
        x = self.conv_end_3(x)

        classes = F.softmax(x[:, 0:self.classes, :, :], dim=1)
        regressions = x[:, self.classes:(2 * self.classes), :, :]
        mask = x[:, [-1], :, :]

        inds = classes.argmax(dim=1).unsqueeze(1)

        if x_gt is None:
            regressions = torch.gather(regressions, dim=1, index=inds)
            x = (inds.type(torch.float32) + regressions) * (1.0 / float(self.classes))
            return x, mask
        else:
            inds_gt = (x_gt * self.classes).type(torch.int64)
            loss = F.cross_entropy(classes, inds_gt.squeeze(1))
            class_losses = [torch.mean(loss * mask_gt)]

            regressions = torch.gather(regressions, dim=1, index=inds_gt)
            x = (inds_gt.type(torch.float32) + regressions) * (1.0 / float(self.classes))
            return x, mask, class_losses

