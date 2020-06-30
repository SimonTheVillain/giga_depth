import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_shrink


# in case we subsample rather later:
# 640×480×4×(1024+513+ 128+4×(128*4 + 64×7 + 1)) would result in a bit below 7GB of video memory.
# in case we subsaple at the first pissible position:
# 640×480×4×(1024+513+ 128*5+4×(64×7 + 1)) would result in a bit below 5GB of video memory.
# (conv_ch_up_1 would have a stride of 2)
class Model_CR10_1_hsn(nn.Module):
    @staticmethod
    def padding():
        return 15  # 15 when striding at first opportunity 27 when at second

    def __init__(self, slices, classes, image_height):
        super(Model_CR10_1_hsn, self).__init__()
        self.slices = slices
        self.classes = classes
        self.r = self.padding()
        self.r_top = 0
        self.r_bottom = 0
        self.height = int(image_height)
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel

        self.short = True
        if self.short:
            self.conv_start = nn.Conv2d(1, 16, 3, padding=1)  # 1
            self.conv_ds1 = nn.Conv2d(16, 32, 5, padding=2, stride=2, groups=1 + 0 * 16)  # + 2 = 3
            self.conv_1 = nn.Conv2d(32, 32, 3, padding=1, stride=1)  # + 2 x 1 = 5
            self.conv_2 = nn.Conv2d(32, 64, 3, padding=1, stride=1, groups=1 + 0 * 32)  # + 2 x 1 = 7
            self.conv_3 = nn.Conv2d(64, 64, 5, padding=2, stride=1)  # + 2 x 2 = 11
            self.conv_4 = nn.Conv2d(64, 128, 5, padding=2, stride=1, groups=1 + 0 * 64)  # + 2 x 2 = 15

        else:
            self.conv_start = nn.Conv2d(1, 16, 3, padding=1)  # 1
            self.conv_ds1 = nn.Conv2d(16, 32, 5, padding=2, stride=2, groups=1 + 0 * 16)  # + 2 = 3
            self.conv_1 = nn.Conv2d(32, 32, 3, padding=1, stride=1)  # + 2 x 1 = 5
            self.conv_2 = nn.Conv2d(32, 32, 3, padding=1, stride=1, groups=1 + 0 * 32)  # + 2 x 1 = 7
            self.conv_3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)  # + 2 x 1 = 9
            self.conv_4 = nn.Conv2d(64, 64, 3, padding=1, stride=1, groups=1 + 0 * 64)  # + 2 x 1 = 11
            self.conv_5 = nn.Conv2d(64, 128, 3, padding=1, stride=1)  # + 2 x 1 = 13
            self.conv_6 = nn.Conv2d(128, 128, 3, padding=1, stride=1, groups=1 + 0 * 128)  # + 2 x 1 = 17


        self.conv_end_shared = nn.ModuleList()
        self.conv_end_c = nn.ModuleList()

        for i in range(0, self.slices):
            self.conv_end_shared.append(nn.Conv2d(128, 256, 1, padding=0, stride=1, groups=1 + 0 * 128))
            self.conv_end_c.append(nn.Conv2d(128, self.classes, 1, padding=0, groups=1))

        # if this does not work... add one more 1x1 convolutional layer here
        half_height = int(image_height / 2)
        # the line-wise version of the class prediction
        # self.conv_end_c = nn.Conv2d(128 * half_height, classes * half_height, 1,
        #                            padding=0, groups=half_height)
        self.conv_end_r = nn.Conv2d((128 + 256 + self.classes) * half_height, classes * half_height, 1,
                                    padding=0, groups=half_height)
        self.conv_end_m = nn.Conv2d(128, 1, 1,
                                    padding=0, groups=1)

    def forward(self, x):
        device = x.device
        input = x
        # print(x.shape)

        x = F.leaky_relu(self.conv_start(x))
        x = F.leaky_relu(self.conv_ds1(x))
        x = F.leaky_relu(self.conv_1(x))
        x = F.leaky_relu(self.conv_2(x))
        x = F.leaky_relu(self.conv_3(x))
        x = F.leaky_relu(self.conv_4(x))
        if not self.short:
            x = F.leaky_relu(self.conv_5(x))
            x = F.leaky_relu(self.conv_6(x))
        #return x
        ### LAYER 0
        latent_shape = (input.shape[0], 128 + 256 + self.classes, x.shape[2], x.shape[3])

        intermediate = torch.zeros(latent_shape, device=device, dtype=x.dtype)
        intermediate[:, 0:128, :, :] = x
        step = int(self.height / self.slices)
        half_step = int(step / 2)
        half_height = int(self.height / 2)
        for i in range(0, self.slices):
            s = x[:, :, (i * half_step):((i + 1) * half_step), :]

            s2 = F.leaky_relu(self.conv_end_shared[i](s))
            s3 = F.leaky_relu(self.conv_end_c[i](s))
            intermediate[:, 128:(128 + 256), i * half_step:(i + 1) * half_step, :] = s2
            intermediate[:, (128 + 256):(128 + 256 + self.classes), i * half_step:(i + 1) * half_step, :] = s3

        # here do the linewise 1x1 convolutions
        # print(x.shape)
        x_latent = intermediate[:, 0:(128 + 256), :, :]
        # the linewise version of the class prediction
        # x = intermediate[:, 0:128, :, :]
        # x = x.transpose(1, 2)
        # x = x.reshape((x.shape[0], 128 * half_height, 1, x.shape[3]))
        # x = F.leaky_relu(self.conv_end_c(x))
        # x = x.reshape((x.shape[0], half_height, self.classes, x.shape[3]))
        # x = x.transpose(1, 2)

        mask = F.leaky_relu(self.conv_end_m(intermediate[:, 0:128, :, :]))
        classes = F.softmax(intermediate[:, (128 + 256):(128 + 256 + self.classes), :, :], dim=1)

        # linewise weights for regression
        x = intermediate
        x = x.transpose(1, 2)
        x = x.reshape((x.shape[0], (128 + 256 + self.classes) * half_height, 1, x.shape[3]))
        regressions = F.leaky_relu(self.conv_end_r(x))
        regressions = regressions.reshape((x.shape[0], half_height, self.classes, x.shape[3]))
        regressions = regressions.transpose(1, 2)

        return classes, regressions, mask, x_latent


