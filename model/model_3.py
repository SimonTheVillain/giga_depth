import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock


#Model2 is Model1 without concatenating the y position at every step.
class Model3(nn.Module):

    def __init__(self):
        super(Model3, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel

        # full
        self.conv1 = nn.Conv2d(2, 64, 3, padding=1, padding_mode='zeros')
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1, padding_mode='same')
        self.strided_convolution1 = nn.Conv2d(64, 128, 5, stride=2, padding=2, padding_mode='same')
        #self.conv1 = nn.Conv2d(66, 63, 3, padding=1, padding_mode='same')

        # 1/2
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1, padding_mode='same')
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, padding_mode='same')
        self.strided_convolution2 = nn.Conv2d(128, 256, 5, stride=2, padding=2, padding_mode='same')

        # 1/4
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1, padding_mode='same')
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1, padding_mode='same')
        self.strided_convolution3 = nn.Conv2d(256, 512, 5, stride=2, padding=2, padding_mode='same')

        # 1/8
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1, padding_mode='same')
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1, padding_mode='same')
        self.strided_convolution4 = nn.Conv2d(512, 1024, 5, stride=2, padding=2, padding_mode='same')

        # 1/16
        self.conv9 = nn.Conv2d(1024, 1024, 3, padding=1, padding_mode='same')
        self.conv10 = nn.Conv2d(1024, 1024, 3, padding=1, padding_mode='same')
        self.up_convolution1 = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=0, padding_mode='zeros')
        self.up_convolution1_correction = nn.Conv2d(512, 512, 2)

        # 1/8
        self.conv_decode_1 = nn.Conv2d(1024, 512, 3, padding=1, padding_mode='same')
        self.conv_decode_2 = nn.Conv2d(512, 512, 3, padding=1, padding_mode='same')
        self.up_convolution2 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=0, padding_mode='zeros')
        self.up_convolution2_correction = nn.Conv2d(256, 256, 2)

        # 1/4
        self.conv_decode_3 = nn.Conv2d(512, 256, 3, padding=1, padding_mode='same')
        self.conv_decode_4 = nn.Conv2d(256, 256, 3, padding=1, padding_mode='same')
        self.up_convolution3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=0, padding_mode='zeros')
        self.up_convolution3_correction = nn.Conv2d(128, 128, 2)

        # 1/2
        self.conv_decode_5 = nn.Conv2d(256, 128, 3, padding=1, padding_mode='same')
        self.conv_decode_6 = nn.Conv2d(128, 2, 3, padding=1, padding_mode='same')

    def forward(self, x):
        #full
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.strided_convolution1(x))

        # 1/2
        x = F.leaky_relu(self.conv3(x))
        x_2 = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.strided_convolution2(x_2))


        # 1/4
        x = F.leaky_relu(self.conv5(x))
        x_4 = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.strided_convolution3(x_4))

        # 1/8
        x = F.leaky_relu(self.conv7(x))
        x_8 = F.leaky_relu(self.conv8(x))
        x = F.leaky_relu(self.strided_convolution4(x_8))

        # 1/16
        x = F.leaky_relu(self.conv9(x))
        x = F.leaky_relu(self.conv10(x))
        x = F.leaky_relu(self.up_convolution1(x))

        # 1/8
        x = F.leaky_relu(self.up_convolution1_correction(x))
        x = torch.cat((x, x_8), 1)
        x = F.leaky_relu(self.conv_decode_1(x))
        x = F.leaky_relu(self.conv_decode_2(x))
        x = F.leaky_relu(self.up_convolution2(x))

        # 1/4
        x = F.relu(self.up_convolution2_correction(x))
        x = torch.cat((x, x_4), 1)
        x = F.leaky_relu(self.conv_decode_3(x))
        x = F.leaky_relu(self.conv_decode_4(x))
        x_latent = torch.cat((x, x_4))
        x = F.leaky_relu(self.up_convolution3(x))

        # 1/2
        x = F.relu(self.up_convolution3_correction(x))
        x = torch.cat((x, x_2), 1)
        x = F.leaky_relu(self.conv_decode_5(x))
        x = F.leaky_relu(self.conv_decode_6(x))

        return x, x_latent


