import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_shrink



# No final layer at target resolution and only 2 layers at the lowest resolution
#better utilization of the Tensor-cores than U3
class BackboneU5Slice(nn.Module):

    def __init__(self, pad=''):
        super(BackboneU5Slice, self).__init__()
        self.pad = pad
        self.start = nn.Conv2d(1, 8, 5, padding=(0, 2), stride=2)  # receptive field (radius) r = 2
        self.conv1 = nn.Conv2d(8, 16, 5, padding=(0, 2))  # + 2 * 2 = 6
        self.conv2 = nn.Conv2d(16, 32, 5, padding=(0, 2))  # + 2 * 2 = 10
        self.conv_sub1 = nn.Conv2d(32, 64, 5, padding=(0, 2), stride=2)  # + 2 * 2 = 14
        self.conv_sub2 = nn.Conv2d(64, 32 * 4, 3, padding=(0, 1))  # + 2*2*1 = 18

        self.n_start = nn.BatchNorm2d(8)
        self.n1 = nn.BatchNorm2d(16)
        self.n2 = nn.BatchNorm2d(32)

        self.nsub1 = nn.BatchNorm2d(64)
        self.nsub2 = nn.BatchNorm2d(32)

    def forward(self, x):
        p = (0, 0, 0, 0)
        if self.pad == 'top':
            p = (0, 0, 2, 0)
        if self.pad == 'bottom':
            p = (0, 0, 0, 2)
        x = F.pad(x, p)
        x = F.leaky_relu(self.n_start(self.start(x)))
        x = F.pad(x, p)
        x = F.leaky_relu(self.n1(self.conv1(x)))
        x = F.pad(x, p)
        x = F.leaky_relu(self.n2(self.conv2(x)))

        # if we pad at the top, this means only the bottom pixel get cut off
        # and vice versa
        if self.pad == 'top':
            x_skip = x[:, :, 0:-4, :]
        if self.pad == 'bottom':
            x_skip = x[:, :, 4:, :]
        if self.pad == '':
            x_skip = x[:, :, 4:-4, :]

        x = F.pad(x, p)
        x = F.leaky_relu(self.nsub1(self.conv_sub1(x)))

        if self.pad == 'top':
            p = (0, 0, 1, 0)
        if self.pad == 'bottom':
            p = (0, 0, 0, 1)
        x = F.pad(x, p)
        x = self.conv_sub2(x)

        x = x.reshape((x.shape[0], 32, 2, 2, x.shape[2], x.shape[3]))
        x = x.permute((0, 1, 4, 2, 5, 3)).reshape((x.shape[0], 32, x.shape[4] * 2, x.shape[5] * 2))

        x = F.leaky_relu(self.nsub2(x))

        x = torch.cat((x, x_skip), dim=1)
        return x

# No final layer at target resolution and only 2 layers at the lowest resolution
#better utilization of the Tensor-cores than U3
class BackboneU5Sliced(nn.Module):

    def __init__(self, slices):
        super(BackboneU5Sliced, self).__init__()
        assert slices > 1, "This model requires more than 1 slice to operate correctly"
        self.slices = nn.ModuleList()
        for i in range(slices):
            if i == 0:
                self.slices.append(BackboneU5Slice('top'))
            else:
                if i == (slices - 1):
                    self.slices.append(BackboneU5Slice('bottom'))
                else:
                    self.slices.append(BackboneU5Slice())


    def forward(self, x, with_debug=False):
        outputs = []
        pad = 18
        split = int(x.shape[2] / len(self.slices))
        for i in range(len(self.slices)):
            line_from = max(0, i * split - pad)
            line_to = min((i + 1) * split + pad, x.shape[2] - 1)
            x_sub = x[:, :, line_from:line_to, :]
            out = self.slices[i](x_sub)
            outputs.append(out)

        x = torch.cat(outputs, dim=2)

        if with_debug:
            debugs = {}
            return x, debugs
        return x