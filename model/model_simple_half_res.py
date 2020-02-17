import torch
import torch.nn as nn
import torch.nn.functional as F



class SimpleNetHalf(nn.Module):

    def __init__(self):
        super(SimpleNetHalf, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_dwn_0_1 = nn.Conv2d(2, 8, 3, padding=1, padding_mode='same')
        self.conv_dwn_0_2 = nn.Conv2d(8, 32, 3, padding=1, padding_mode='same')
        self.conv_dwn_0_3 = nn.Conv2d(32, 64, 3, padding=1, padding_mode='same')
        #pool
        self.conv_dwn_1_1 = nn.Conv2d(64, 64, 3, padding=1, padding_mode='same')
        #pool
        self.conv_dwn_2_1 = nn.Conv2d(64, 128, 3, padding=1, padding_mode='same')
        #pool
        self.conv_dwn_3_1 = nn.Conv2d(128, 128, 3, padding=1, padding_mode='same')
        self.conv_dwn_3_2 = nn.Conv2d(128, 256, 1)
        self.conv_dwn_3_3 = nn.Conv2d(256, 256, 1)


        self.upconv_3_2 = nn.ConvTranspose2d(256, 128, 3, 2) # channels in/out , kernel size stride
        self.conv_up_2_1 = nn.Conv2d(128, 128, 2) # after the upconv each dimension grows for +1 which we need to reverse
        #concatenate here +128
        self.conv_up_2_2 = nn.Conv2d(256, 128, 3, padding=1, padding_mode='same')  #128

        #at this point we concatenate 64 channels
        self.upconv_2_1 = nn.ConvTranspose2d(128, 64, 3, 2) # channels in/out , kernel size stride
        self.conv_up_1_1 = nn.Conv2d(64, 64, 2) # after the upconv each dimension grows for +1 which we need to reverse
        #concatenate here +64

        self.conv_up_1_2 = nn.Conv2d(128, 64, 3, padding=1, padding_mode='same')
        self.conv_up_1_3 = nn.Conv2d(64, 32, 3, padding=1, padding_mode='same')
        self.conv_up_1_4 = nn.Conv2d(32, 2, 3, padding=1, padding_mode='same')


    def forward(self, x):

        ### RES_LAYER 0
        x = F.leaky_relu(self.conv_dwn_0_1(x))#8
        x = F.leaky_relu(self.conv_dwn_0_2(x))#32
        x = F.leaky_relu(self.conv_dwn_0_3(x))#64

        ### RES_LAYER 1
        x = F.max_pool2d(x, (2, 2))#64
        layer2 = F.leaky_relu(self.conv_dwn_1_1(x))#64
        ### RES_LAYER 2
        x = F.max_pool2d(layer2, 2)#64
        layer3 = F.leaky_relu(self.conv_dwn_2_1(x))#128
        ### RES_LAYER 3
        x = F.max_pool2d(layer3, 2)#128

        x = F.leaky_relu(self.conv_dwn_3_1(x))#128
        x = F.leaky_relu(self.conv_dwn_3_2(x))#256
        x = F.leaky_relu(self.conv_dwn_3_3(x))#256

        #########UPWARD#########
        ### RES_LAYER 2
        x = F.leaky_relu(self.upconv_3_2(x))#128
        x = F.leaky_relu(self.conv_up_2_1(x))#128
        x = torch.cat((x, layer3), 1)#128 + 128
        x = F.leaky_relu(self.conv_up_2_2(x))#128

        ### RES_LAYER 1
        x = F.leaky_relu(self.upconv_2_1(x))#64
        x = F.leaky_relu(self.conv_up_1_1(x))#64
        x = torch.cat((x, layer2), 1)# 64 +64
        x = F.leaky_relu(self.conv_up_1_2(x))#64
        x = F.leaky_relu(self.conv_up_1_3(x))#32
        x = F.leaky_relu(self.conv_up_1_4(x))#2

        return x


