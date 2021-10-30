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
        return 17

    def __init__(self, channels=[16, 32, 64], channels_sub=[64, 64, 64, 64],
                 use_bn=False, pad='', channels_in=2, downsample=True):
        super(Backbone3Slice, self).__init__()
        assert downsample, "It is assumed that the output resolution is half the input."
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
                                        nn.Conv2d(channels[0], channels[1], 5, padding=(0, 2)),  # + 2 = 5
                                        nn.BatchNorm2d(channels[1]),
                                        nn.LeakyReLU())
        else:
            self.block1 = nn.Sequential(nn.ConstantPad2d(p1, 0),
                                        nn.Conv2d(channels_in, channels[0], 3, padding=(0, 1)),
                                        nn.LeakyReLU(),
                                        nn.ConstantPad2d(p2, 0),
                                        nn.Conv2d(channels[0], channels[1], 5, padding=(0, 2)),
                                        nn.LeakyReLU())

        modules = [nn.ConstantPad2d(p2, 0),
                   nn.Conv2d(channels[1], channels[2], 5, padding=(0, 2), stride=(2, 2)),
                   nn.BatchNorm2d(channels[2]),
                   nn.LeakyReLU()]
        for i in range(0, 3):
            modules.append(nn.ConstantPad2d(p2, 0))
            modules.append(nn.Conv2d(channels_sub[i], channels_sub[i + 1], 5, padding=(0, 2)))  # + 2*2
            if use_bn:
                modules.append(nn.BatchNorm2d(channels_sub[i + 1]))
            modules.append(nn.LeakyReLU())

        self.block2 = nn.Sequential(*modules)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class BackboneSlicer(nn.Module):

    def __init__(self, BackboneSlice, constructor, slices, lcn=True, downsample_output=True):
        super(BackboneSlicer, self).__init__()
        in_channels = 1
        self.required_padding = BackboneSlice.get_required_padding(downsample_output)
        self.LCN = lcn
        if lcn:
            in_channels = 2
        # assert slices > 1, "This model requires more than 1 slice to operate correctly"
        self.slices = nn.ModuleList()
        if slices == 1:
            nn.ModuleList(constructor('both', in_channels, downsample_output))
            return
        for i in range(slices):
            if i == 0:
                self.slices.append(constructor('top', in_channels, downsample_output))
            else:
                if i == (slices - 1):
                    self.slices.append(constructor('bottom', in_channels, downsample_output))
                else:
                    self.slices.append(constructor('', in_channels, downsample_output))

    def forward(self, x, with_debug=False):
        if self.LCN:
            lcn, _, _ = LCN_tensors(x)
            x = torch.cat((x, lcn), 1)
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