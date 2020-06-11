import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_shrink


#in case we subsample rather later:
#640×480×4×(1024+513+ 128+4×(128*4 + 64×7 + 1)) would result in a bit below 7GB of video memory.
#in case we subsaple at the first pissible position:
#640×480×4×(1024+513+ 128*5+4×(64×7 + 1)) would result in a bit below 5GB of video memory.
# (conv_ch_up_1 would have a stride of 2)
class Model_CR10_hsn(nn.Module):
    @staticmethod
    def padding():
        return 15 #15 when striding at first opportunity 27 when at second

    def __init__(self, slices, classes, image_height, pad_top=True, pad_bottom=True):
        super(Model_CR10_hsn, self).__init__()
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
        self.conv_ds1 = nn.ModuleList()
        self.conv_1 = nn.ModuleList()
        self.conv_2 = nn.ModuleList()
        #self.conv_ds2 = nn.ModuleList()
        self.conv_3 = nn.ModuleList()
        self.conv_4 = nn.ModuleList()
        self.conv_5 = nn.ModuleList()
        self.conv_6 = nn.ModuleList()
        self.conv_7 = nn.ModuleList()
        #self.conv_8 = nn.moduleList()
        #self.conv_9 = nn.ModuleList()
        #self.conv_10 = nn.ModuleList()
        #self.conv_11 = nn.moduleList()
        #self.resi_block1 = nn.ModuleList()
        #self.resi_block2 = nn.ModuleList()
        #self.conv_ch_up_1 = nn.ModuleList()
        #self.resi_block3 = nn.ModuleList()
        #self.conv_end_1 = nn.ModuleList()
        for i in range(0, self.slices):
            self.conv_start.append(nn.Conv2d(1, 16, 3, padding=0)) #1
            self.conv_ds1.append(nn.Conv2d(16, 32, 5, padding=0, stride=2, groups=1+0*16)) # + 2 = 3
            self.conv_1.append(nn.Conv2d(32, 32, 3, padding=0, stride=1)) # + 2 x 1 = 5
            self.conv_2.append(nn.Conv2d(32, 32, 3, padding=0, stride=1, groups=1+0*32)) # + 2 x 1 = 7
            self.conv_3.append(nn.Conv2d(32, 64, 3, padding=0, stride=1)) # + 2 x 1 = 9
            self.conv_4.append(nn.Conv2d(64, 64, 3, padding=0, stride=1, groups=1+0*64))  # + 2 x 1 = 11
            self.conv_5.append(nn.Conv2d(64, 128, 3, padding=0, stride=1)) # + 2 x 1 = 13
            self.conv_6.append(nn.Conv2d(128, 128, 3, padding=0, stride=1, groups=1+0*128))
            self.conv_7.append(nn.Conv2d(128, 256, 1, padding=0, stride=1, groups=1+0*128)) # + 2 x 1 = 17

            #self.resi_block1.append(ResidualBlock_shrink(32, 3, 0, depadding=3))
            #self.resi_block2.append(ResidualBlock_shrink(32, 3, 0, depadding=3))
            #self.conv_ch_up_1.append(nn.Conv2d(32, 64, 5, padding=0, stride=2))#-4
            #self.resi_block3.append(ResidualBlock_shrink(64, 3, 0, depadding=3))#-6
            #self.conv_end_1.append(nn.Conv2d(64, 128, 7, padding=0, padding_mode='replicate', stride=1))

        # if this does not work... add one more 1x1 convolutional layer here
        half_height = int(image_height/2)
        #the line-wise version of the class prediction
        #self.conv_end_c = nn.Conv2d(128 * half_height, classes * half_height, 1,
        #                            padding=0, groups=half_height)
        self.conv_end_c = nn.Conv2d(128, self.classes, 1,padding=0,groups=1)
        self.conv_end_r = nn.Conv2d((128 + 256 + self.classes) * half_height, classes * half_height, 1,
                                    padding=0, groups=half_height)
        self.conv_end_m = nn.Conv2d(128, 1, 1,
                                    padding=0, groups=1)


    def forward(self, x):
        device = x.device
        input = x
        #print(x.shape)

        ### LAYER 0
        x = torch.zeros((x.shape[0], x.shape[1], x.shape[2] + self.r_top + self.r_bottom, x.shape[3] + self.r*2),
                        device=device)
        x[:, :, self.r_top:x.shape[2] - self.r_bottom, self.r:-self.r] = input
        latent_shape = (input.shape[0], 128+256+self.classes, int((x.shape[2] - 2 * self.r) / 2), int(input.shape[3] / 2))
        #print(latent_shape)
        intermediate = torch.zeros(latent_shape, device=device)

        step = int(self.height / self.slices)
        half_step = int(step / 2)
        half_height = int(self.height/2)
        for i in range(0, self.slices):
            s = x[:, :, (i*step):(self.r*2 + (i+1)*step), :]
            s = F.leaky_relu(self.conv_start[i](s))
            s = F.leaky_relu(self.conv_ds1[i](s))
            s = F.leaky_relu(self.conv_1[i](s))
            s = F.leaky_relu(self.conv_2[i](s))
            s = F.leaky_relu(self.conv_3[i](s))
            s = F.leaky_relu(self.conv_4[i](s))
            s = F.leaky_relu(self.conv_5[i](s))
            s1 = F.leaky_relu(self.conv_6[i](s))
            s2 = F.leaky_relu(self.conv_7[i](s1))
            intermediate[:, 0:128, i*half_step:(i+1)*half_step, :] = s1
            intermediate[:, 128:(128+256), i*half_step:(i+1)*half_step, :] = s2

        #here do the linewise 1x1 convolutions
        #print(x.shape)
        x_latent = intermediate[:, 0:(128+256), :, :]
        #the linewise version of the class prediction
        #x = intermediate[:, 0:128, :, :]
        #x = x.transpose(1, 2)
        #x = x.reshape((x.shape[0], 128 * half_height, 1, x.shape[3]))
        #x = F.leaky_relu(self.conv_end_c(x))
        #x = x.reshape((x.shape[0], half_height, self.classes, x.shape[3]))
        #x = x.transpose(1, 2)

        #lets not use linewise weights for classification and mask:
        x = F.leaky_relu(self.conv_end_c(intermediate[:, 0:128, :, :]))
        intermediate[:, (128 + 256):(128 + 256 + self.classes), :, :] = x
        mask = F.leaky_relu(self.conv_end_m(intermediate[:, 0:128, :, :]))

        classes = F.softmax(x, dim=1)

        #linewise weights for regression
        x = intermediate
        x = x.transpose(1, 2)
        x = x.reshape((x.shape[0], (128 + 256 + self.classes) * half_height, 1, x.shape[3]))
        regressions = F.leaky_relu(self.conv_end_r(x))
        regressions = regressions.reshape((x.shape[0], half_height, self.classes, x.shape[3]))
        regressions = regressions.transpose(1, 2)


        return classes, regressions, mask, x_latent


