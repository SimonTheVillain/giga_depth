import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_shrink
from common.common import LCN_tensors, LCN


# No final layer at target resolution and only 2 layers at the lowest resolution
# better utilization of the Tensor-cores than U3
class BackboneU5Slice(nn.Module):

    def __init__(self, pad='', in_channels=1):
        super(BackboneU5Slice, self).__init__()
        self.pad = pad
        self.start = nn.Conv2d(in_channels, 8, 5, padding=(0, 2), stride=2)  # receptive field (radius) r = 2
        self.conv1 = nn.Conv2d(8, 16, 5, padding=(0, 2))  # + 2 * 2 = 6
        self.conv2 = nn.Conv2d(16, 32, 5, padding=(0, 2))  # + 2 * 2 = 10
        self.conv_sub1 = nn.Conv2d(32, 64, 5, padding=(0, 2), stride=2)  # + 2 * 2 = 14
        self.conv_sub2 = nn.Conv2d(64, 32 * 4, 3, padding=(0, 1))  # + 2*2*1 = 18

        self.n_start = nn.BatchNorm2d(8)
        self.n1 = nn.BatchNorm2d(16)
        self.n2 = nn.BatchNorm2d(32)

        self.nsub1 = nn.BatchNorm2d(64)
        self.nsub2 = nn.BatchNorm2d(32)
        self.p1 = (0, 0, 0, 0)
        self.p2 = (0, 0, 0, 0)
        if self.pad == 'top':
            self.p1 = (0, 0, 2, 0)
            self.p2 = (0, 0, 1, 0)
        if self.pad == 'bottom':
            self.p1 = (0, 0, 0, 2)
            self.p2 = (0, 0, 0, 1)
        if self.pad == "both":
            self.p1 = (0, 0, 2, 2)
            self.p2 = (0, 0, 1, 1)

    @staticmethod
    def get_required_padding(downsample=False):
        return 18

    def forward(self, x):

        x = F.pad(x, self.p1)
        x = F.leaky_relu(self.n_start(self.start(x)))
        x = F.pad(x, self.p1)
        x = F.leaky_relu(self.n1(self.conv1(x)))
        x = F.pad(x, self.p1)
        x = F.leaky_relu(self.n2(self.conv2(x)))

        # for the skip connection we need to cut out any row that we lose during
        # convolution:
        # if we pad at the top, this means only the bottom pixel get cut off
        # and vice versa
        if self.pad == 'top':
            x_skip = x[:, :, 0:-4, :]
        if self.pad == 'bottom':
            x_skip = x[:, :, 4:, :]
        if self.pad == '':
            x_skip = x[:, :, 4:-4, :]
        if self.pad == 'both':
            x_skip = x[:, :, :, :]

        x = F.pad(x, self.p1)
        x = F.leaky_relu(self.nsub1(self.conv_sub1(x)))

        x = F.pad(x, self.p2)
        x = self.conv_sub2(x)

        x = x.reshape((x.shape[0], 32, 2, 2, x.shape[2], x.shape[3]))
        x = x.permute((0, 1, 4, 2, 5, 3)).reshape((x.shape[0], 32, x.shape[4] * 2, x.shape[5] * 2))

        x = F.leaky_relu(self.nsub2(x))

        x = torch.cat((x, x_skip), dim=1)
        return x


class Backbone3Slice(nn.Module):

    @staticmethod
    def get_required_padding(downsample=True):
        if downsample:
            return 17
        return 17

    def __init__(self, channels=[16, 32], channels_sub=[64, 64, 64, 64],
                 use_bn=False, pad='', channels_in=2, downsample=True):
        super(Backbone3Slice, self).__init__()
        self.downsample = downsample
        assert len(channels) == 2, "Only 2 stages are supported before the downsampling"
        #assert downsample, "It is assumed that the output resolution is half the input."
        p1 = (0, 0, 0, 0)
        p2 = (0, 0, 0, 0)
        if pad == 'top':
            p1 = (0, 0, 1, 0)
            p2 = (0, 0, 2, 0)
        if pad == 'bottom':
            p1 = (0, 0, 0, 1)
            p2 = (0, 0, 0, 2)
        if pad == 'both':
            p1 = (0, 0, 1, 1)
            p2 = (0, 0, 2, 2)

        if use_bn:
            self.block1 = nn.Sequential(nn.ConstantPad2d(p1, 0),
                                        nn.Conv2d(channels_in, channels[0], 3, padding=(0, 1)),  # 1
                                        nn.BatchNorm2d(channels[0]),
                                        nn.LeakyReLU(),
                                        nn.ConstantPad2d(p2, 0),
                                        nn.Conv2d(channels[0], channels[1], 5, padding=(0, 2)),  # + 2 = 3
                                        nn.BatchNorm2d(channels[1]),
                                        nn.LeakyReLU())
        else:
            self.block1 = nn.Sequential(nn.ConstantPad2d(p1, 0),
                                        nn.Conv2d(channels_in, channels[0], 3, padding=(0, 1)),
                                        nn.LeakyReLU(),
                                        nn.ConstantPad2d(p2, 0),
                                        nn.Conv2d(channels[0], channels[1], 5, padding=(0, 2)),
                                        nn.LeakyReLU())


        if downsample:
            assert len(channels_sub) == 4, "The downsampled CNN needs to have 4 stages"
            modules = [nn.ConstantPad2d(p2, 0),
                       nn.Conv2d(channels[1], channels_sub[0], 5, padding=(0, 2), stride=(2, 2)),  # + 2 = 5
                       nn.BatchNorm2d(channels_sub[0]),
                       nn.LeakyReLU()]
            for i in range(0, 3):
                modules.append(nn.ConstantPad2d(p2, 0))
                modules.append(nn.Conv2d(channels_sub[i], channels_sub[i + 1], 5, padding=(0, 2)))  # + 2*2
                if use_bn:
                    modules.append(nn.BatchNorm2d(channels_sub[i + 1]))
                modules.append(nn.LeakyReLU())
        else:
            modules = [nn.ConstantPad2d(p2, 0),
                       nn.Conv2d(channels[1], channels_sub[0], 3, padding=(0, 1), stride=(1, 1)),  # + 1 = 3
                       nn.BatchNorm2d(channels_sub[0]),
                       nn.LeakyReLU()]
            assert len(channels_sub) == 8, "The non-downsampled CNN needs to have 8 stages"
            for i in range(0, 7):
                modules.append(nn.ConstantPad2d(p2, 0))
                modules.append(nn.Conv2d(channels_sub[i], channels_sub[i + 1], 5, padding=(0, 2)))  # + 2
                if use_bn:
                    modules.append(nn.BatchNorm2d(channels_sub[i + 1]))
                modules.append(nn.LeakyReLU())

        self.block2 = nn.Sequential(*modules)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class BackboneSlice(nn.Module):

    def get_required_padding(self, downsample=True):
        return self.receptive_field

    @staticmethod
    def get_pad(mode, value):
        k = value
        p = (0, 0, 0, 0)
        if mode == 'top':
            p = (0, 0, k, 0)
        if mode == 'bottom':
            p = (0, 0, 0, k)
        if mode == 'both':
            p = (0, 0, k, k)
        return p

    def __init__(self, channels=[16, 32],
                 kernel_sizes=[5, 5],
                 channels_sub=[64, 64, 64, 64],
                 kernel_sizes_sub=[5, 5, 5, 5],
                 use_bn=False, pad='', channels_in=2):
        super(BackboneSlice, self).__init__()
        self.downsample = len(channels_sub) > 0

        self.base_res = False
        self.receptive_field = 0
        if len(channels) != 0:
            self.base_res = True
            p = kernel_sizes[0] // 2
            self.receptive_field += p

            modules = []

            if pad == "both":
                modules.append(nn.Conv2d(channels_in, channels[0], kernel_sizes[0], padding=(p, p)))  # + p
            else:
                modules.append(nn.ConstantPad2d(self.get_pad(pad, p), 0))
                modules.append(nn.Conv2d(channels_in, channels[0], kernel_sizes[0], padding=(0, p)))# + p
            modules.append(nn.BatchNorm2d(channels[0]))
            modules.append(nn.LeakyReLU())
            for ch_in, ch_out, ks in zip(channels[:-1], channels[1:], kernel_sizes[1:]):
                p = ks // 2
                self.receptive_field += p
                if pad == "both":
                    modules.append(nn.Conv2d(ch_in, ch_out, ks, padding=(p, p)))  # + p
                else:
                    modules.append(nn.ConstantPad2d(self.get_pad(pad, p), 0))
                    modules.append(nn.Conv2d(ch_in, ch_out, ks, padding=(0, p)))  # + p
                if use_bn:
                    modules.append(nn.BatchNorm2d(ch_out))
                modules.append(nn.LeakyReLU())
            self.block1 = nn.Sequential(*modules)
        else:
            channels = [channels_in]

        if self.downsample:
            p = kernel_sizes_sub[0] // 2
            self.receptive_field += p
            modules = []
            if pad == "both":
                modules.append(
                    nn.Conv2d(channels[-1], channels_sub[0], kernel_sizes_sub[0], padding=(p, p), stride=(2, 2)))  # + 2
            else:
                modules.append(nn.ConstantPad2d(self.get_pad(pad, p), 0))
                modules.append(
                    nn.Conv2d(channels[-1], channels_sub[0], kernel_sizes_sub[0], padding=(0, p), stride=(2, 2)))  # + 2

            modules.append(nn.BatchNorm2d(channels_sub[0]))
            modules.append(nn.LeakyReLU())

            for ch_in, ch_out, ks in zip(channels_sub[:-1], channels_sub[1:], kernel_sizes_sub[1:]):
                p = ks // 2
                self.receptive_field += 2 * p
                if pad == "both":
                    modules.append(nn.Conv2d(ch_in, ch_out, ks, padding=(p, p)))  # + 2*p
                else:
                    modules.append(nn.ConstantPad2d(self.get_pad(pad, p), 0))
                    modules.append(nn.Conv2d(ch_in, ch_out, ks, padding=(0, p)))  # + 2*p
                if use_bn:
                    modules.append(nn.BatchNorm2d(ch_out))
                modules.append(nn.LeakyReLU())
            self.block2 = nn.Sequential(*modules)

    def forward(self, x, with_debug=False):
        if self.base_res:
            x = self.block1(x)
        if self.downsample:
            x = self.block2(x)
        if with_debug:
            return x, {}
        return x

class BackboneULight(nn.Module):
    def __init__(self, in_channels):
        super(BackboneULight, self).__init__()

        self.preconv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True))
        self.conv1 = self.downsample_conv(32, 64, 5) # receptive field = 3 + 2*3 -> 9
        #self.conv2 = self.downsample_conv(32, 64, 5) # += 2*2 + 4*2 -> 21
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=5, padding=2), # += 2*2 -> 13
            nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1), # += 2*1 ->15
            nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True))

        self.upconv1 = self.upconv(128, 64)

        self.outconv = torch.nn.Sequential(
            torch.nn.Conv2d(96, 96, kernel_size=3, padding=1), #  += 2*3 -> 9
            nn.BatchNorm2d(96),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(96, 96, kernel_size=3, padding=1), #  += 2*3 -> 9
            nn.BatchNorm2d(96),
            torch.nn.ReLU(inplace=True))


    def get_required_padding(self, downsample=True):
        return 15

    def downsample_conv(self, in_planes, out_planes, kernel_size=3):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(out_planes),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(out_planes),
            torch.nn.ReLU(inplace=True)
        )

    def upconv(self, in_planes, out_planes):
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_planes),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x, with_debug=False):
        x1 = self.preconv1(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.upconv1(x3)
        x4 = torch.cat((x1, x4), dim=1)
        xout = self.outconv(x4)
        if with_debug:
            return xout, {}
        return xout


class BackboneSlicer(nn.Module):

    def __init__(self, BackboneSlice, constructor, slices, in_channels, downsample_output=True):#, required_padding=-1):
        super(BackboneSlicer, self).__init__()
        #if required_padding == -1:
        #    self.required_padding = BackboneSlice.get_required_padding(downsample_output)# TODO: move this to the parameters!
        #else:
        #    self.required_padding = required_padding
        # assert slices > 1, "This model requires more than 1 slice to operate correctly"
        self.slices = nn.ModuleList()
        if slices == 1:
            self.slices = nn.ModuleList([constructor('both', in_channels, downsample_output)])
            self.required_padding = self.slices[0].get_required_padding(downsample_output)
            return
        for i in range(slices):
            if i == 0:
                self.slices.append(constructor('top', in_channels, downsample_output))
                self.required_padding = self.slices[0].get_required_padding(downsample_output)
            else:
                if i == (slices - 1):
                    self.slices.append(constructor('bottom', in_channels, downsample_output))
                else:
                    self.slices.append(constructor('', in_channels, downsample_output))

    def forward(self, x, with_debug=False):

        outputs = []
        pad = self.required_padding
        split = int(x.shape[2] / len(self.slices))
        for i in range(len(self.slices)):
            line_from = max(0, i * split - pad)
            line_to = min((i + 1) * split + pad, x.shape[2])
            #print(f"from {line_from} to {line_to}")
            x_sub = x[:, :, line_from:line_to, :]
            #print(x_sub.shape)
            out = self.slices[i](x_sub)
            #print(out.shape)
            outputs.append(out)

        x = torch.cat(outputs, dim=2)

        if with_debug:
            debugs = {}
            return x, debugs
        return x