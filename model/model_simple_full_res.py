import torch
import torch.nn as nn
import torch.nn.functional as F



class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_dwn_0_1 = nn.Conv2d(2, 8, 3, padding=1, padding_mode='same')
        self.conv_dwn_1_1 = nn.Conv2d(8, 32, 3, padding=1, padding_mode='same')
        self.conv_dwn_2_1 = nn.Conv2d(32, 64, 3, padding=1, padding_mode='same')
        self.conv_dwn_3_1 = nn.Conv2d(64, 128, 3, padding=1, padding_mode='same')
        self.conv_dwn_3_2 = nn.Conv2d(128, 256, 3, padding=1, padding_mode='same')


        self.upconv_3_2 = nn.ConvTranspose2d(256, 192, 3, 2) # channels in/out , kernel size stride
        self.conv_up_2_1 = nn.Conv2d(192, 192, 2) # after the upconv each dimension grows for +1 which we need to reverse
        #at this point we concatenate 64 channels
        self.upconv_2_1 = nn.ConvTranspose2d(256, 96, 3, 2) # channels in/out , kernel size stride
        self.conv_up_1_1 = nn.Conv2d(96, 96, 2) # after the upconv each dimension grows for +1 which we need to reverse
        #at this point we concatenate 16 channels
        self.upconv_1_0 = nn.ConvTranspose2d(128, 56, 3, 2) # channels in/out , kernel size stride
        self.conv_up_0_1 = nn.Conv2d(56, 56, 2) # after the upconv each dimension grows for +1 which we need to reverse
        #at this point we concatenate 4 channels

        self.conv_up_0_2 = nn.Conv2d(64, 32, 3, padding=1, padding_mode='same')
        self.conv_up_0_3 = nn.Conv2d(32, 16, 3, padding=1, padding_mode='same')
        self.conv_up_0_4 = nn.Conv2d(16, 2, 3, padding=1, padding_mode='same')

    def forward(self, x):
        ### LAYER 0
        layer1 = F.leaky_relu(self.conv_dwn_0_1(x))

        ### LAYER 1
        x = F.max_pool2d(layer1, (2, 2))#2 by 2 window
        #8
        layer2 = F.leaky_relu(self.conv_dwn_1_1(x))
        x = F.max_pool2d(layer2, 2)# same 2 by 2 window
        #32
        layer3 = F.leaky_relu(self.conv_dwn_2_1(x))

        ### LAYER 2
        x = F.max_pool2d(layer3, 2)
        #64
        x = F.leaky_relu(self.conv_dwn_3_1(x))
        #128
        x = F.leaky_relu(self.conv_dwn_3_2(x))


        #256
        x = F.leaky_relu(self.upconv_3_2(x))
        #192
        x = F.leaky_relu(self.conv_up_2_1(x))
        #192
        # concatenate + 64
        x = torch.cat((x, layer3), 1)

        ### LAYER 1
        #256
        x = F.leaky_relu(self.upconv_2_1(x))
        #96
        x = F.leaky_relu(self.conv_up_1_1(x))
        #96
        #concatenate + 32
        x = torch.cat((x, layer2), 1)

        ### LAYER 0
        #128
        x = F.leaky_relu(self.upconv_1_0(x))
        #56
        x = F.leaky_relu(self.conv_up_0_1(x))
        #56
        #condatenate + 8
        x = torch.cat((x, layer1), 1)
        #64
        x = F.leaky_relu(self.conv_up_0_2(x))
        #32
        x = F.leaky_relu(self.conv_up_0_3(x))
        #16
        x = self.conv_up_0_4(x)
        #2

        return x


