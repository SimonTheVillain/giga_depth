import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_shrink


#in case we subsample rather later:
#640×480×4×(1024+513+ 128+4×(128*4 + 64×7 + 1)) would result in a bit below 7GB of video memory.
#in case we subsaple at the first pissible position:
#640×480×4×(1024+513+ 128*5+4×(64×7 + 1)) would result in a bit below 5GB of video memory.
# (conv_ch_up_1 would have a stride of 2)
class Model_CR11_hn(nn.Module):
    @staticmethod
    def padding():
        return 0

    def __init__(self, height, classes):
        super(Model_CR11_hn, self).__init__()
        self.height = height
        self.classes = classes
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel

        self.conv_start = nn.Conv2d(1, 32, 3, padding=1) # 1
        self.conv_ds_1 = nn.Conv2d(32, 32, 5, padding=2, stride=2) # half # + 2 = 3
        self.conv_d1_1 = nn.Conv2d(32, 32, 5, padding=2, stride=1) # + 2*2 =7
        #self.conv_s2_2 = nn.Conv2d(32, 32, 5, padding=2, stride=1)
        self.conv_ds_2 = nn.Conv2d(32, 64, 5, padding=2, stride=2) # quarter # + 2*2 = 11
        self.conv_d2_1 = nn.Conv2d(64, 64, 5, padding=2, stride=1) # + 4*2 = 19
        self.conv_ds_3 = nn.Conv2d(64, 128, 5, padding=2, stride=2) # eighth # + 4*2 = 27
        self.conv_d3_1 = nn.Conv2d(128, 256, 3, padding=1, stride=1)  # + 8*1 = 35
        h_8 = int(height/8)
        self.conv_d3_2 = nn.Conv2d(256 * h_8, 512 * h_8, 1, padding=0, stride=1, groups=h_8)
        self.conv_d3_3 = nn.Conv2d(512 * h_8, 64 * h_8, 1, padding=0, stride=1, groups=h_8)

        self.us2 = nn.Upsample(scale_factor=2, mode='nearest')
        #self.conv_us1_2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=0)#quarter
        #self.conv_us2_2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=0)#quarter
        #self.conv_us3_2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=0)#quarter
        #self.conv_us4_2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=0)#quarter
        self.conv_u2_1 = nn.Conv2d(64 + 64, 64, 5, padding=2, stride=1, groups=1)


        self.us1 = nn.Upsample(scale_factor=2, mode='nearest')
        #self.conv_us1_1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)#half
        #self.conv_us2_1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)#half
        #self.conv_us3_1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)#half
        #self.conv_us4_1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)#half
        self.conv_u1_1 = nn.Conv2d(64 + 32, 32, 5, padding=2, stride=1)
        self.conv_u1_2 = nn.Conv2d(32, classes*2 + 8, 5, padding=2, stride=1)



    def forward(self, x):

        h_8 = int(self.height/8)
        h_2 = int(self.height/2)
        device = x.device
        input = x
        x = F.leaky_relu(self.conv_start(x))

        #half
        x = F.leaky_relu(self.conv_ds_1(x))
        x = F.leaky_relu(self.conv_d1_1(x))
        x_half = x

        #quarter
        x = F.leaky_relu(self.conv_ds_2(x))
        x = F.leaky_relu(self.conv_d2_1(x))
        x_quarter = x

        #eigth
        x = F.leaky_relu(self.conv_ds_3(x))
        x = F.leaky_relu(self.conv_d3_1(x))

        #now the linewise part
        x = x.transpose(1, 2)
        x = x.reshape((x.shape[0], 256 * h_8, 1, x.shape[3]))
        x = F.leaky_relu(self.conv_d3_2(x))
        x = F.leaky_relu(self.conv_d3_3(x))
        x = x.reshape((x.shape[0], h_8, 64, x.shape[3]))
        x = x.transpose(1, 2)

        #fourth
        #x = F.leaky_relu(self.conv_us1_2(x))
        x = self.us2(x)
        x = torch.cat((x, x_quarter), 1)
        x = F.leaky_relu(self.conv_u2_1(x))

        # half
        #x = F.leaky_relu(self.conv_us1_1(x))
        x = self.us1(x)
        x = torch.cat((x, x_half), 1)
        x_latent = x
        x = F.leaky_relu(self.conv_u1_1(x))
        x = F.leaky_relu(self.conv_u1_2(x))

        classes = F.softmax(x[:, 0:self.classes, :, :], dim=1)
        regressions = x[:, self.classes: self.classes*2, :, :]
        mask = x[:, [self.classes*2 + 1], :, :]
        return classes, regressions, mask, x_latent


