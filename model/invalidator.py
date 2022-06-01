import torch
import torch.nn as nn
from uNet import ConvBlock, EncoderBlock, DecoderBlock

class InvalidationModel(nn.Module):

    def __init__(self, channels=[4, 8, 32, 32, 1]):
        super(InvalidationModel, self).__init__()
        self.seq = nn.Sequential()
        for i in range(len(channels)-1):
            self.seq.add_module(f"conv{i}", nn.Conv2d(channels[i], channels[i+1], 5, 1, 2))
            self.seq.add_module(f"BN{i}", nn.BatchNorm2d(channels[i+1]))
            self.seq.add_module(f"ReLU{i}", nn.LeakyReLU())

    def forward(self, x):
        return self.seq(x)


class InvalidationModelU(nn.Module):
    def __init__(self):
        super().__init__()
        # ENCODER
        self.e1 = EncoderBlock(4, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        self.e4 = EncoderBlock(128, 256)
        # BOTTLENECK
        self.b = ConvBlock(256, 512)
        # DECODER
        self.d1 = DecoderBlock(512, 256)
        self.d2 = DecoderBlock(256, 128)
        self.d3 = DecoderBlock(128, 64)
        self.d4 = DecoderBlock(64, 32)
        # CLASSIFIER
        self.outputs = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
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
        d4 = self.d4(d3, s1)
        # CLASSIFIER
        outputs = self.outputs(d4)
        return outputs