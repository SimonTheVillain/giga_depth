import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_shrink
from common.common import LCN_tensors
import cv2 #TODO: remove for debug measures


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


#same as before but one level shorter
class SliceShort2(nn.Module):

    @staticmethod
    def radius():
        return 6

    def __init__(self, channels=[64, 64, 64, 64], use_bn=False):
        super(SliceShort2, self).__init__()
        if use_bn:
            self.bn1 = nn.BatchNorm2d(channels[1])
            self.bn2 = nn.BatchNorm2d(channels[2])
            self.bn3 = nn.BatchNorm2d(channels[3])
        self.conv_1 = nn.Conv2d(channels[0], channels[1], 5, padding=(0, 2))  # + 2 * 2 = 9
        self.conv_2 = nn.Conv2d(channels[1], channels[2], 5, padding=(0, 2))  # + 2 * 2 = 13
        self.conv_3 = nn.Conv2d(channels[2], channels[3], 5, padding=(0, 2))  # + 2 * 2 = 17

    def forward(self, x, with_debug=False):
        if hasattr(self, 'bn1'):
            x = F.leaky_relu(self.bn1(self.conv_1(x)))
            x = F.leaky_relu(self.bn2(self.conv_2(x)))
            if with_debug:
                x = self.conv_3(x)

                debug = {"bb_mean_x_before_bn": x.abs().mean()}
                x = self.bn3(x)
                debug["bb_mean_x_after_bn"] = x.abs().mean()
                x = F.leaky_relu(x)
                return x, debug
            else:
                x = F.leaky_relu(self.bn3(self.conv_3(x)))
                return x
        else:
            x = F.leaky_relu(self.conv_1(x))
            x = F.leaky_relu(self.conv_2(x))
            x = F.leaky_relu(self.conv_3(x))
            return x


class BackboneSliced3(nn.Module):

    @staticmethod
    def radius():
        return 12

    def __init__(self, slices=8, height=896, channels=[16, 32, 64], channels_sub=[64, 64, 64, 64], use_bn=False):
        super(BackboneSliced3, self).__init__()
        # the first 4 layers are properly padded:
        self.height = height
        self.features = channels_sub[3]
        self.conv_start = nn.Conv2d(1, channels[0], 3, padding=(1, 1))
        self.conv_1 = nn.Conv2d(channels[0], channels[1], 5, padding=(2, 2))
        self.conv_down = nn.Conv2d(channels[1], channels[2], 5, padding=(2, 2), stride=(2, 2))

        if use_bn:
            self.bn_start = nn.BatchNorm2d(channels[0])
            self.bn_1 = nn.BatchNorm2d(channels[1])
            self.bn_down = nn.BatchNorm2d(channels[2])
        #448
        #If it wouldn't be padded vertically, the image

        # subsampled from here!
        self.slices = nn.ModuleList()
        for i in range(0, slices):
            self.slices.append(SliceShort2(channels=channels_sub, use_bn=use_bn))

    def forward(self, x, with_debug=False):
        r = self.slices[0].radius()
        device = x.device
        nr_slices = len(self.slices)
        #print(nr_slices)
        #print(x.shape)
        if hasattr(self, 'bn_start'):
            x = F.leaky_relu(self.bn_start(self.conv_start(x)))
            x = F.leaky_relu(self.bn_1(self.conv_1(x)))
            x = F.leaky_relu(self.bn_down(self.conv_down(x)))
        else:
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
        debugs = {}
        for i in range(0, nr_slices):
            #calculate the source slice:
            t_in = max(0, i * height_slice - r)
            b_in = min((i + 1) * height_slice + r - 1, height_half - 1)

            #calculate the destionation (bottom edge) of this slice:
            b_out = t_out + b_in - t_in - 2 * r

            x_p = x[:, :, t_in:b_in, :]

            #print(f" take from {t_in} to {b_in}")
            if with_debug:
                x_p, debugs_slice = self.slices[i](x_p, True)
                for key, val in debugs_slice.items():
                    if key in debugs:
                        debugs[key] += val
                    else:
                        debugs[key] = val

            else:
                x_p = self.slices[i](x_p)

            #print(f" store from {t_out} to {b_out}")
            #print(x_p.shape)
            result_accumulator[:, :, t_out:b_out, :] = x_p

            #calculate the destination top edge of the next slice:
            t_out = b_out + 1

        #print(result_accumulator.shape)
        x = F.pad(result_accumulator, (0, 0, r, r), "replicate")
        #print(x.shape)
        if with_debug:
            return x, debugs
        else:
            return x


class BackboneNoSlice3(nn.Module):

    @staticmethod
    def radius():
        return 12

    def __init__(self, height=896, channels=[16, 32, 64], channels_sub=[64, 64, 64, 64], use_bn=False):
        super(BackboneNoSlice3, self).__init__()
        self.height = height
        self.features = channels_sub[3]
        self.conv_start = nn.Conv2d(1, channels[0], 3, padding=(1, 1))
        self.conv_1 = nn.Conv2d(channels[0], channels[1], 5, padding=(2, 2))
        self.conv_down = nn.Conv2d(channels[1], channels[2], 5, padding=(2, 2), stride=(2, 2))

        self.conv_2 = nn.Conv2d(channels_sub[0], channels_sub[1], 5, padding=(2, 2))  # + 2 * 2 = 9
        self.conv_3 = nn.Conv2d(channels_sub[1], channels_sub[2], 5, padding=(2, 2))  # + 2 * 2 = 13
        self.conv_4 = nn.Conv2d(channels_sub[2], channels_sub[3], 5, padding=(2, 2))  # + 2 * 2 = 17

        if use_bn:
            self.bn_start = nn.BatchNorm2d(channels[0])
            self.bn_1 = nn.BatchNorm2d(channels[1])
            self.bn_down = nn.BatchNorm2d(channels[2])
            self.bn_2 = nn.BatchNorm2d(channels_sub[1])
            self.bn_3 = nn.BatchNorm2d(channels_sub[2])
            self.bn_4 = nn.BatchNorm2d(channels_sub[3])

    def forward(self, x, with_debug=False):
        device = x.device
        #print(nr_slices)
        #print(x.shape)
        if hasattr(self, 'bn_start'):
            x = F.leaky_relu(self.bn_start(self.conv_start(x)))
            x = F.leaky_relu(self.bn_1(self.conv_1(x)))
            x = F.leaky_relu(self.bn_down(self.conv_down(x)))
            x = F.leaky_relu(self.bn_2(self.conv_2(x)))
            x = F.leaky_relu(self.bn_3(self.conv_3(x)))
            if with_debug:
                x = self.conv_4(x)
                debugs = {"bb_mean_x_before_bn": x.abs().mean()}
                x = self.bn_4(x)
                debugs["bb_mean_x_after_bn"] = x.abs().mean()
                x = F.leaky_relu(x)
                return x, debugs
            else:
                x = F.leaky_relu(self.bn_4(self.conv_4(x)))
                return x
        else:
            x = F.leaky_relu(self.conv_start(x))
            x = F.leaky_relu(self.conv_1(x))
            x = F.leaky_relu(self.conv_down(x))
            x = F.leaky_relu(self.conv_2(x))
            x = F.leaky_relu(self.conv_3(x))
            x = F.leaky_relu(self.conv_4(x))
            return x


# backboneNoSlice4 is similar to BackboneNoSlice3 but when downsampling the channels are split into a near and a far
# receptive field near and far receptive field are concatenated at the output
class BackboneNoSlice4(nn.Module):

    @staticmethod
    def radius():
        return 12

    def __init__(self, height=896, channels=[16, 32, 64], channels_sub=[64, 64, 64, 64], use_bn=False):
        super(BackboneNoSlice4, self).__init__()
        self.height = height
        self.features = channels_sub[3]
        self.conv_start = nn.Conv2d(1, channels[0], 3, padding=(1, 1)) # receptive field (radius) = 1
        self.conv_1 = nn.Conv2d(channels[0], channels[1], 5, padding=(2, 2)) # + 2 = 3
        self.conv_down = nn.Conv2d(channels[1], channels[2] * 2, 5, padding=(2, 2), stride=(2, 2)) # +2 = 5

        self.conv_2 = nn.Conv2d(channels_sub[0]*2, channels_sub[1]*2, 5, padding=(2, 2), groups=2)  # + 2 * 2 = 9
        self.conv_3 = nn.Conv2d(channels_sub[1], channels_sub[2], 5, padding=(2, 2))  # + 2 * 2 = 13
        self.conv_4 = nn.Conv2d(channels_sub[2], channels_sub[3], 5, padding=(2, 2))  # + 2 * 2 = 17

        if use_bn:
            self.bn_start = nn.BatchNorm2d(channels[0])
            self.bn_1 = nn.BatchNorm2d(channels[1])
            self.bn_down = nn.BatchNorm2d(channels[2]*2)
            self.bn_2 = nn.BatchNorm2d(channels_sub[1]*2)
            self.bn_3 = nn.BatchNorm2d(channels_sub[2])
            self.bn_4 = nn.BatchNorm2d(channels_sub[3])

    def forward(self, x, with_debug=False):
        device = x.device
        #print(nr_slices)
        #print(x.shape)
        if hasattr(self, 'bn_start'):
            x = F.leaky_relu(self.bn_start(self.conv_start(x)))
            x = F.leaky_relu(self.bn_1(self.conv_1(x)))
            x = F.leaky_relu(self.bn_down(self.conv_down(x)))
            x = F.leaky_relu(self.bn_2(self.conv_2(x)))

            x, x_l = torch.split(x, int(x.shape[1]/2), 1)# split up the channels in two
            x = F.leaky_relu(self.bn_3(self.conv_3(x)))
            if with_debug:
                x = self.conv_4(x)
                debugs = {"bb_mean_x_before_bn": x.abs().mean()}
                x = self.bn_4(x)
                debugs["bb_mean_x_after_bn"] = x.abs().mean()
                x = F.leaky_relu(x)
                x = torch.cat((x, x_l), 1)
                return x, debugs
            else:
                x = F.leaky_relu(self.bn_4(self.conv_4(x)))
                x = torch.cat((x, x_l), 1)
                return x
        else:
            x = F.leaky_relu(self.conv_start(x))
            x = F.leaky_relu(self.conv_1(x))
            x = F.leaky_relu(self.conv_down(x))
            x = F.leaky_relu(self.conv_2(x))
            x, x_l = torch.split(x, int(x.shape[1]/2), 1)# split up the output in two
            x = F.leaky_relu(self.conv_3(x))
            x = F.leaky_relu(self.conv_4(x))
            x = torch.cat((x, x_l), 1)
            return x

# go a bit more u-shaped
class BackboneU1(nn.Module):

    def __init__(self):
        super(BackboneU1, self).__init__()
        self.start = nn.Conv2d(1, 8, 3, padding=1, stride=(2, 1))#putting this here gives us less than a millisecond
        self.conv1 = nn.Conv2d(8, 16, 5, padding=2)  # reduce lines early on (gets down from 51 to 47ms)
        self.convdown = nn.Conv2d(16, 32, 5, padding=2, stride=(1, 2))  # reduce colums right after

        self.conv_sub1 = nn.Conv2d(32, 64, 5, padding=2, stride=2)
        self.conv_sub2 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv_sub3 = nn.Conv2d(64, 64, 1, padding=0)
        self.conv_sub4 = nn.Conv2d(64, 64, 1, padding=0)
        self.conv_sub5 = nn.Conv2d(64, 32 * 4, 1, padding=0)  # todo: should this one have a kernel size bigger 1?

        self.convout = nn.Conv2d(64, 64, 3, padding=1)

        self.bn_start = nn.BatchNorm2d(8)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn_down = nn.BatchNorm2d(32)

        self.bnout = nn.BatchNorm2d(64)

        self.bnsub1 = nn.BatchNorm2d(64)
        self.bnsub2 = nn.BatchNorm2d(64)
        self.bnsub3 = nn.BatchNorm2d(64)
        self.bnsub4 = nn.BatchNorm2d(64)
        self.bnsub5 = nn.BatchNorm2d(32)

    def forward(self, x, with_debug=False):
        x = F.leaky_relu(self.bn_start(self.start(x)))
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn_down(self.convdown(x)))
        x_l1 = x

        x_l1 = F.leaky_relu(self.bnsub1(self.conv_sub1(x_l1)))
        x_l1 = F.leaky_relu(self.bnsub2(self.conv_sub2(x_l1)))
        x_l1 = F.leaky_relu(self.bnsub3(self.conv_sub3(x_l1)))  # these here are incredibly cheap
        x_l1 = F.leaky_relu(self.bnsub4(self.conv_sub4(x_l1)))
        x_l1 = self.conv_sub5(x_l1)
        x_l1 = x_l1.reshape((x_l1.shape[0], 32, 2, 2, x_l1.shape[2], x_l1.shape[3]))
        x_l1 = x_l1.permute((0, 1, 4, 2, 5, 3)).reshape((x_l1.shape[0], 32, x.shape[2], x.shape[3]))
        x_l1 = F.leaky_relu(self.bnsub5(x_l1))

        x = torch.cat((x, x_l1), dim=1)
        x = F.leaky_relu(self.bnout(self.convout(x)))
        if with_debug:
            debugs = {}
            return x, debugs
        return x

# go a bit more u-shaped
class BackboneU2(nn.Module):

    def __init__(self):
        super(BackboneU2, self).__init__()
        self.start = nn.Conv2d(1, 16, 5, padding=2, stride=2) # receptive field (radius) r = 2
        self.conv1 = nn.Conv2d(16, 16, 5, padding=2)  #  + 2 * 2 = 6

        self.conv_sub1 = nn.Conv2d(16, 32, 5, padding=2, stride=2) # + 2 * 2 = 10
        self.conv_sub2 = nn.Conv2d(32, 64, 5, padding=2) # + 2*2*2 = 18
        self.conv_sub3 = nn.Conv2d(64, 128, 1, padding=0)
        self.conv_sub4 = nn.Conv2d(128, 32 * 4, 1, padding=0)  # todo: should this one have a kernel size bigger 1?

        self.convout = nn.Conv2d(48, 64, 3, padding=1, groups=2) # 18 + 2*2 = 22 or 10 for the channels coming from the skip conn.

        self.bn_start = nn.BatchNorm2d(16)
        self.bn1 = nn.BatchNorm2d(16)

        self.bnout = nn.BatchNorm2d(64)

        self.bnsub1 = nn.BatchNorm2d(32)
        self.bnsub2 = nn.BatchNorm2d(64)
        self.bnsub3 = nn.BatchNorm2d(128)
        self.bnsub4 = nn.BatchNorm2d(32)

    def forward(self, x, with_debug=False):
        x = F.leaky_relu(self.bn_start(self.start(x)))
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x_skip = x

        x = F.leaky_relu(self.bnsub1(self.conv_sub1(x)))
        x = F.leaky_relu(self.bnsub2(self.conv_sub2(x)))
        x = F.leaky_relu(self.bnsub3(self.conv_sub3(x)))  # these here are incredibly cheap
        x = self.conv_sub4(x)
        x = x.reshape((x.shape[0], 32, 2, 2, x.shape[2], x.shape[3]))
        x = x.permute((0, 1, 4, 2, 5, 3)).reshape((x.shape[0], 32, x_skip.shape[2], x_skip.shape[3]))
        x = F.leaky_relu(self.bnsub4(x))

        x = torch.cat((x, x_skip), dim=1)
        x = F.leaky_relu(self.bnout(self.convout(x)))
        if with_debug:
            debugs = {}
            return x, debugs
        return x

# Forget the idea of saturating the tensor units! Try to be more minimalistic than U2
class BackboneU3(nn.Module):

    def __init__(self):
        super(BackboneU3, self).__init__()
        self.start = nn.Conv2d(1, 8, 5, padding=2, stride=2) # receptive field (radius) r = 2
        self.conv1 = nn.Conv2d(8, 16, 3, padding=1)  #  + 2 * 1 = 4

        self.conv_sub1 = nn.Conv2d(16, 32, 5, padding=2, stride=2) # + 2 * 2 = 8
        self.conv_sub2 = nn.Conv2d(32, 64, 3, padding=1) # + 2*2*1 = 12
        self.conv_sub3 = nn.Conv2d(64, 32 * 4, 3, padding=1) # + 2*2*1 = 18

        self.convout = nn.Conv2d(48, 64, 3, padding=1, groups=2) # 18 + 2*2 = 20 or 8 for the channels coming from the skip conn.

        self.bn_start = nn.BatchNorm2d(8)
        self.bn1 = nn.BatchNorm2d(16)


        self.bnsub1 = nn.BatchNorm2d(32)
        self.bnsub2 = nn.BatchNorm2d(64)
        self.bnsub3 = nn.BatchNorm2d(32)

        self.bnout = nn.BatchNorm2d(64)

    def forward(self, x, with_debug=False):
        x = F.leaky_relu(self.bn_start(self.start(x)))
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x_skip = x

        x = F.leaky_relu(self.bnsub1(self.conv_sub1(x)))
        x = F.leaky_relu(self.bnsub2(self.conv_sub2(x)))
        x = self.conv_sub3(x)
        x = x.reshape((x.shape[0], 32, 2, 2, x.shape[2], x.shape[3]))
        x = x.permute((0, 1, 4, 2, 5, 3)).reshape((x.shape[0], 32, x_skip.shape[2], x_skip.shape[3]))
        x = F.leaky_relu(self.bnsub3(x))

        x = torch.cat((x, x_skip), dim=1)
        x = F.leaky_relu(self.bnout(self.convout(x)))
        if with_debug:
            debugs = {}
            return x, debugs
        return x

# Skip the output network that once more operates at the target resolution
#better utilization of the Tensor-cores than U3
class BackboneU4(nn.Module):

    def __init__(self):
        super(BackboneU4, self).__init__()
        self.start = nn.Conv2d(1, 8, 5, padding=2, stride=2)  # receptive field (radius) r = 2
        self.conv1 = nn.Conv2d(8, 16, 3, padding=1)  # + 2 * 1 = 4
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)  # + 2 * 2 = 8
        self.conv_sub1 = nn.Conv2d(32, 64, 5, padding=2, stride=2)  # + 2 * 2 = 12
        self.conv_sub2 = nn.Conv2d(64, 64, 3, padding=1)  # + 2*2*1 = 18
        self.conv_sub3 = nn.Conv2d(64, 32 * 4, 3, padding=1)  # + 2*2*1 = 20

        self.bn_start = nn.BatchNorm2d(8)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)

        self.bnsub1 = nn.BatchNorm2d(64)
        self.bnsub2 = nn.BatchNorm2d(64)
        self.bnsub3 = nn.BatchNorm2d(32)

    def forward(self, x, with_debug=False):
        x = F.leaky_relu(self.bn_start(self.start(x)))
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x_skip = x

        x = F.leaky_relu(self.bnsub1(self.conv_sub1(x)))
        x = F.leaky_relu(self.bnsub2(self.conv_sub2(x)))
        x = self.conv_sub3(x)
        x = x.reshape((x.shape[0], 32, 2, 2, x.shape[2], x.shape[3]))
        x = x.permute((0, 1, 4, 2, 5, 3)).reshape((x.shape[0], 32, x_skip.shape[2], x_skip.shape[3]))
        x = F.leaky_relu(self.bnsub3(x))

        x = torch.cat((x, x_skip), dim=1)
        if with_debug:
            debugs = {}
            return x, debugs
        return x

# No final layer at target resolution and only 2 layers at the lowest resolution
#better utilization of the Tensor-cores than U3
class BackboneU5(nn.Module):

    def __init__(self, norm='batch', lcn=False):
        super(BackboneU5, self).__init__()
        in_channels = 1
        self.LCN = lcn
        if lcn:
            in_channels = 2
        self.start = nn.Conv2d(in_channels, 8, 5, padding=2, stride=2)  # receptive field (radius) r = 2
        self.conv1 = nn.Conv2d(8, 16, 5, padding=2)  # + 2 * 2 = 6
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)  # + 2 * 2 = 10
        self.conv_sub1 = nn.Conv2d(32, 64, 5, padding=2, stride=2)  # + 2 * 2 = 14
        self.conv_sub2 = nn.Conv2d(64, 32 * 4, 3, padding=1)  # + 2*2*1 = 20

        if norm == 'batch':
            self.n_start = nn.BatchNorm2d(8)
            self.n1 = nn.BatchNorm2d(16)
            self.n2 = nn.BatchNorm2d(32)

            self.nsub1 = nn.BatchNorm2d(64)
            self.nsub2 = nn.BatchNorm2d(32)
        if norm == 'group':
            self.n_start = nn.GroupNorm(2, 8)
            self.n1 = nn.GroupNorm(2, 16)
            self.n2 = nn.GroupNorm(4, 32)

            self.nsub1 = nn.GroupNorm(8, 64)
            self.nsub2 = nn.GroupNorm(4, 32)

    def forward(self, x, with_debug=False):
        if self.LCN:
            x = torch.cat((x, LCN_tensors(x)), 1)
        x = F.leaky_relu(self.n_start(self.start(x)))
        x = F.leaky_relu(self.n1(self.conv1(x)))
        x = F.leaky_relu(self.n2(self.conv2(x)))
        x_skip = x

        x = F.leaky_relu(self.nsub1(self.conv_sub1(x)))
        x = self.conv_sub2(x)
        x = x.reshape((x.shape[0], 32, 2, 2, x.shape[2], x.shape[3]))
        x = x.permute((0, 1, 4, 2, 5, 3)).reshape((x.shape[0], 32, x_skip.shape[2], x_skip.shape[3]))
        x = F.leaky_relu(self.nsub2(x))

        x = torch.cat((x, x_skip), dim=1)
        if with_debug:
            debugs = {}
            return x, debugs
        return x