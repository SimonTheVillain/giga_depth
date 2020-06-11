import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_shrink


#in case we subsample rather later:
#640×480×4×(1024+513+ 128+4×(128*4 + 64×7 + 1)) would result in a bit below 7GB of video memory.
#in case we subsaple at the first pissible position:
#640×480×4×(1024+513+ 128*5+4×(64×7 + 1)) would result in a bit below 5GB of video memory.
# (conv_ch_up_1 would have a stride of 2)
class Model_CR8_hsn(nn.Module):
    @staticmethod
    def padding():
        return 1+3+3+2+6+3+0+0 + 9 #18 when striding at first opportunity 27 when at second

    def __init__(self, slices, classes, image_height, pad_top=True, pad_bottom=True):
        super(Model_CR8_hsn, self).__init__()
        self.slices = slices
        self.classes = classes
        self.r = self.padding()
        self.r_top = 0
        self.r_bottom = 0
        if pad_top:
            self.r_top = self.r
        if pad_bottom:
            self.r_bottom = self.r
        self.height = int(image_height)
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel

        self.conv_start = nn.ModuleList()
        self.resi_block1 = nn.ModuleList()
        self.resi_block2 = nn.ModuleList()
        self.conv_ch_up_1 = nn.ModuleList()
        self.resi_block3 = nn.ModuleList()
        self.conv_end_1 = nn.ModuleList()
        for i in range(0, self.slices):
            self.conv_start.append(nn.Conv2d(1, 32, 3, padding=0)) #1
            self.resi_block1.append(ResidualBlock_shrink(32, 3, 0, depadding=3))
            self.resi_block2.append(ResidualBlock_shrink(32, 3, 0, depadding=3))
            self.conv_ch_up_1.append(nn.Conv2d(32, 64, 5, padding=0, stride=2))#-4
            self.resi_block3.append(ResidualBlock_shrink(64, 5, 0, depadding=6))#-6
            self.conv_end_1.append(nn.Conv2d(64, 128, 7, padding=0, padding_mode='replicate', stride=1))

        # if this does not work... add one more 1x1 convolutional layer here
        half_height = int(image_height/2)
        self.conv_end_2 = nn.Conv2d(128 * half_height, 512 * half_height, 1, padding=0, groups=half_height)
        self.conv_end_3 = nn.Conv2d(512 * half_height, (classes*2+1)*half_height, 1, padding=0, groups=half_height)

    def forward(self, x):
        device = x.device
        input = x
        #print(x.shape)

        ### LAYER 0
        x = torch.zeros((x.shape[0], x.shape[1], x.shape[2] + self.r_top + self.r_bottom, x.shape[3] + self.r*2),
                        device=device)
        x[:, :, self.r_top:x.shape[2] - self.r_bottom, self.r:-self.r] = input
        latent_shape = (input.shape[0], 128, int((x.shape[2] - 2 * self.r) / 2), int(input.shape[3] / 2))
        #print(latent_shape)
        intermediate = torch.zeros(latent_shape, device=device)

        step = int(self.height / self.slices)
        half_step = int(step / 2)
        half_height = int(self.height/2)
        for i in range(0, self.slices):
            s = x[:, :, (i*step):(self.r*2 + (i+1)*step), :]
            s = F.leaky_relu(self.conv_start[i](s))
            s = self.resi_block1[i](s)
            s = self.resi_block2[i](s)
            s = F.leaky_relu(self.conv_ch_up_1[i](s))
            s = self.resi_block3[i](s)
            s = F.leaky_relu(self.conv_end_1[i](s))
            intermediate[:, :, i*half_step:(i+1)*half_step, :] = s

        #here do the linewise 1x1 convolutions
        #print(x.shape)
        x_latent = x = intermediate
        x = x.transpose(1, 2)
        x = x.reshape((x.shape[0], 128 * half_height, 1, x.shape[3]))
        x = F.leaky_relu(self.conv_end_2(x))#the separable linewise convolutions
        x = F.leaky_relu(self.conv_end_3(x))
        x = x.reshape((x.shape[0], half_height, (self.classes*2+1), x.shape[3]))
        x = x.transpose(1, 2)

        classes = F.softmax(x[:, 0:self.classes, :, :], dim=1)
        regressions = x[:, self.classes:(2 * self.classes), :, :]
        mask = x[:, [-1], :, :]
        return classes, regressions, mask, x_latent


