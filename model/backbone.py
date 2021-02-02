import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_shrink


# same as the linewise cr8_no_residual_light. It seems like a good compromise
# no residuals are needed according to my experiment
# also it might be possible to go further down with computations but first lets check this out!

class Backbone(nn.Module):
    def radius(self):
        return 17

    def __init__(self):
        super(Backbone, self).__init__()
        # kernel
        self.conv_start = nn.Conv2d(1, 16, 3, padding=(0, 1))  # 1
        self.conv_1 = nn.Conv2d(16, 32, 3, padding=(0, 1))  # + 1 = 2
        self.conv_2 = nn.Conv2d(32, 32, 3, padding=(0, 1))  # + 1 = 3
        self.conv_3_down = nn.Conv2d(32, 64, 5, padding=(0, 2), stride=(2, 2))  # + 2 = 5
        # subsampled from here!
        self.conv_4 = nn.Conv2d(64, 64, 3, padding=(0, 1))  # + 1 * 2 = 7
        self.conv_5 = nn.Conv2d(64, 64, 3, padding=(0, 1))  # + 1 * 2 = 9
        self.conv_6 = nn.Conv2d(64, 64, 3, padding=(0, 1))  # + 1 * 2 = 11
        self.conv_7 = nn.Conv2d(64, 64, 3, padding=(0, 1))  # + 1 * 2 = 13
        self.conv_8 = nn.Conv2d(64, 64, 3, padding=(0, 1))  # + 1 * 2 = 15
        self.conv_9 = nn.Conv2d(64, 128, 3, padding=(0, 1))  # + 1 * 2 = 17

    def forward(self, x):
        x = F.leaky_relu(self.conv_start(x))
        x = F.leaky_relu(self.conv_1(x))
        x = F.leaky_relu(self.conv_2(x))
        x = F.leaky_relu(self.conv_3_down(x))
        x = F.leaky_relu(self.conv_4(x))
        x = F.leaky_relu(self.conv_5(x))
        x = F.leaky_relu(self.conv_6(x))
        x = F.leaky_relu(self.conv_7(x))
        x = F.leaky_relu(self.conv_8(x))
        x = F.leaky_relu(self.conv_9(x))
        return x


class Slice(nn.Module):
    def radius(self):
        return 6

    def __init__(self):
        super(Slice, self).__init__()
        self.conv_4 = nn.Conv2d(64, 64, 3, padding=(0, 1))  # + 1 * 2 = 7
        self.conv_5 = nn.Conv2d(64, 64, 3, padding=(0, 1))  # + 1 * 2 = 9
        self.conv_6 = nn.Conv2d(64, 64, 3, padding=(0, 1))  # + 1 * 2 = 11
        self.conv_7 = nn.Conv2d(64, 64, 3, padding=(0, 1))  # + 1 * 2 = 13
        self.conv_8 = nn.Conv2d(64, 64, 3, padding=(0, 1))  # + 1 * 2 = 15
        self.conv_9 = nn.Conv2d(64, 128, 3, padding=(0, 1))  # + 1 * 2 = 17

    def forward(self, x):
        x = F.leaky_relu(self.conv_4(x))
        x = F.leaky_relu(self.conv_5(x))
        x = F.leaky_relu(self.conv_6(x))
        x = F.leaky_relu(self.conv_7(x))
        x = F.leaky_relu(self.conv_8(x))
        x = F.leaky_relu(self.conv_9(x))
        return x

class BackboneSliced(nn.Module):
    def radius(self):
        return 12

    def __init__(self, slices=8, height=896):
        super(BackboneSliced, self).__init__()
        # the first 4 layers are properly padded:
        self.height = height
        self.conv_start = nn.Conv2d(1, 16, 3, padding=(1, 1))
        self.conv_1 = nn.Conv2d(16, 32, 3, padding=(1, 1))
        self.conv_2 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.conv_3_down = nn.Conv2d(32, 64, 5, padding=(2, 2), stride=(2, 2))
        #448
        #If it wouldn't be padded vertically, the image

        # subsampled from here!
        self.slices = nn.ModuleList()
        for i in range(0, slices):
            self.slices.append(Slice())

    def forward(self, x):
        r = self.slices[0].radius()
        device = x.device
        nr_slices = len(self.slices)

        x = F.leaky_relu(self.conv_start(x))
        x = F.leaky_relu(self.conv_1(x))
        x = F.leaky_relu(self.conv_2(x))
        x = F.leaky_relu(self.conv_3_down(x))

        height_half = int(self.height / 2)
        height_slice = int(height_half / nr_slices)
        assert self.height % 2 == 0, "expect height to be divisible by 2"
        assert height_half % nr_slices == 0, "expect height/2 to be divisible by nr of slices"
        #print(x.shape)
        result_accumulator = torch.zeros((x.shape[0], 128, x.shape[2] - r*2, x.shape[3]), device=device)
        t_out = 0
        for i in range(0, nr_slices):
            #calculate the source slice:
            t_in = max(0, i * height_slice - r)
            b_in = min((i + 1) * height_slice + r - 1, height_half - 1)

            #calculate the destionation (bottom edge) of this slice:
            b_out = t_out + b_in - t_in - 2 * r

            x_p = x[:, :, t_in:b_in, :]

            #print(f" take from {t_in} to {b_in}")
            x_p = self.slices[i](x_p)

            #print(f" store from {t_out} to {b_out}")
            #print(x_p.shape)
            result_accumulator[:, :, t_out:b_out, :] = x_p

            #calculate the destination top edge of the next slice:
            t_out = b_out + 1
        #print(result_accumulator.shape)
        x = F.pad(result_accumulator, (0, 0, r, r), "replicate")
        #print(x.shape)
        return x


class SliceShort(nn.Module):

    @staticmethod
    def radius():
        return 6

    def __init__(self, channels=[64, 64, 64, 64, 64]):
        super(SliceShort, self).__init__()
        self.conv_1 = nn.Conv2d(channels[0], channels[1], 5, padding=(0, 2))  # + 2 * 2 = 9
        self.conv_2 = nn.Conv2d(channels[1], channels[2], 5, padding=(0, 2))  # + 2 * 2 = 13
        self.conv_3 = nn.Conv2d(channels[2], channels[3], 3, padding=(0, 1))  # + 1 * 2 = 15
        self.conv_4 = nn.Conv2d(channels[3], channels[4], 3, padding=(0, 1))  # + 1 * 2 = 17

    def forward(self, x):
        x = F.leaky_relu(self.conv_1(x))
        x = F.leaky_relu(self.conv_2(x))
        x = F.leaky_relu(self.conv_3(x))
        x = F.leaky_relu(self.conv_4(x))
        return x


class BackboneSliced2(nn.Module):

    @staticmethod
    def radius():
        return 12

    def __init__(self, slices=8, height=896, channels=[16, 32, 64], channels_sub=[64, 64, 64, 64, 64]):
        super(BackboneSliced2, self).__init__()
        # the first 4 layers are properly padded:
        self.height = height
        self.features = channels_sub[4]
        self.conv_start = nn.Conv2d(1, channels[0], 3, padding=(1, 1))
        self.conv_1 = nn.Conv2d(channels[0], channels[1], 5, padding=(2, 2))
        self.conv_down = nn.Conv2d(channels[1], channels[2], 5, padding=(2, 2), stride=(2, 2))
        #448
        #If it wouldn't be padded vertically, the image

        # subsampled from here!
        self.slices = nn.ModuleList()
        for i in range(0, slices):
            self.slices.append(SliceShort(channels=channels_sub))

    def forward(self, x):
        r = self.slices[0].radius()
        device = x.device
        nr_slices = len(self.slices)
        #print(nr_slices)
        #print(x.shape)
        x = F.leaky_relu(self.conv_start(x))
        x = F.leaky_relu(self.conv_1(x))
        x = F.leaky_relu(self.conv_down(x))
        #print(x.shape)
        height_half = int(self.height / 2)
        height_slice = int(height_half / nr_slices)
        assert self.height % 2 == 0, "expect height to be divisible by 2"
        assert height_half % nr_slices == 0, "expect height/2 to be divisible by nr of slices"
        #print(x.shape)
        result_accumulator = torch.zeros((x.shape[0], self.features, x.shape[2] - r*2, x.shape[3]), device=device)
        t_out = 0
        for i in range(0, nr_slices):
            #calculate the source slice:
            t_in = max(0, i * height_slice - r)
            b_in = min((i + 1) * height_slice + r - 1, height_half - 1)

            #calculate the destionation (bottom edge) of this slice:
            b_out = t_out + b_in - t_in - 2 * r

            x_p = x[:, :, t_in:b_in, :]

            #print(f" take from {t_in} to {b_in}")
            x_p = self.slices[i](x_p)

            #print(f" store from {t_out} to {b_out}")
            #print(x_p.shape)
            result_accumulator[:, :, t_out:b_out, :] = x_p

            #calculate the destination top edge of the next slice:
            t_out = b_out + 1

        #print(result_accumulator.shape)
        x = F.pad(result_accumulator, (0, 0, r, r), "replicate")
        #print(x.shape)
        return x
