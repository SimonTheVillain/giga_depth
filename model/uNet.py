import torch
import torch.nn as nn

#from the u-net architecture
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, cin=1, cout=1, half_out=False, channel_size_scale=1):
        super().__init__()
        self.half_out = half_out
        css = channel_size_scale
        # ENCODER
        self.e1 = EncoderBlock(cin, int(64*css))
        self.e2 = EncoderBlock(int(64 * css), int(128 * css))
        self.e3 = EncoderBlock(int(128 * css), int(256 * css))
        self.e4 = EncoderBlock(int(256 * css), int(512 * css))
        # BOTTLENECK
        self.b = ConvBlock(int(512 * css), int(1024 * css))
        # DECODER
        self.d1 = DecoderBlock(int(1024 * css), int(512 * css))
        self.d2 = DecoderBlock(int(512 * css), int(256 * css))
        self.d3 = DecoderBlock(int(256 * css), int(128 * css))
        if not half_out:
            self.d4 = DecoderBlock(int(128 * css), int(64 * css))
            # CLASSIFIER
            self.outputs = nn.Conv2d(int(64 * css), cout, kernel_size=1, padding=0)
        else:
            # CLASSIFIER
            self.outputs = nn.Conv2d(int(128 * css), cout, kernel_size=1, padding=0)

    def forward(self, inputs, debugs=False):
        # ENCODER
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        # BOTTLENECK
        b = self.b(p4)
        # DECODER
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        if self.half_out:
            outputs = self.outputs(d3)
        else:
            d4 = self.d4(d3, s1)
            # CLASSIFIER
            outputs = self.outputs(d4)
        if debugs:
            return outputs, []
        return outputs