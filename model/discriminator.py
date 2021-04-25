import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock




class DiscriminatorSlice(nn.Module):

    def __init__(self, ch_in, ch_internal):
        super(DiscriminatorSlice, self).__init__()
        #what the hell!!?
        self.relublock1 = ResidualBlock(ch_in, 3, 1, 4)
        #maxpool /2
        self.conv1 = nn.Conv2d(ch_in, ch_internal, 5, padding=2, padding_mode='same')

        self.relublock2 = ResidualBlock(ch_internal, 3, 1)
        self.relublock3 = ResidualBlock(ch_internal, 3, 1)
        #maxpool /4

        self.relublock4 = ResidualBlock(ch_internal, 3, 1)
        self.relublock5 = ResidualBlock(256, 3, 1)
        #maxpool /8

        self.relublock6 = ResidualBlock(ch_internal, 3, 1)
        self.relublock7 = ResidualBlock(ch_internal, 3, 1)
        #maxpool /16
        self.conv2 = nn.Conv2d(ch_internal, 32, 5, padding=2, padding_mode='same')


        #w = w/16
        #h = h/16
        #self.fc1 = nn.Linear(128 * w * h,4096)
        #self.fc2 = nn.Linear(4096,1)

        #fully connected layers?

    def forward(self, x):
        # x has 2 channels

        x = self.relublock1(x)
        x = F.max_pool2d(x, 2) # /2
        x = self.leaky_relu(self.conv1(x))
        x = self.relublock2(x)
        x = self.relublock3(x)
        x = F.max_pool2d(x, 2) # /4
        x = self.relublock4(x)
        x = self.relublock5(x)
        x = F.max_pool2d(x, 2) # /8
        x = self.relublock6(x)
        x = self.relublock7(x)
        x = F.max_pool2d(x, 2) # /16
        x = F.leaky_relu(self.conv2(x))
        #x = torch.flatten(x,1)
        #x = F.leaky_relu(self.fc1(x))
        #x = F.sigmoid(self.fc2(x))
        return x



class DiscriminatorSliced(nn.Module):

    def __init__(self, ch_in, width, height, slices):
        super(DiscriminatorSliced, self).__init__()
        self.width = width
        self.height = height
        self.ch_internal = 256

        self.slices = nn.ModuleList()
        for i in range(slices):
            self.descriminators.append(DiscriminatorSlice(ch_in, self.ch_internal))

        # todo: how many channels do we have for the output
        ch_out = (width * height) / (16*16)

        self.fc1 = nn.Linear(ch_out, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        slice_height = int(x.shape[2] / len(self.slices))
        output = []
        for i, slice in enumerate(self.slices):
            x_slice = x[:, :, i * slice_height:(i+1)*slice_height, :]
            x_slice = slice(x_slice)
            output.append(torch.flatten(x_slice, start_dim=1))
        x = torch.cat(output, dim=1)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))

        #TODO: find out how the output is supposed to look like
        # https://github.com/corenel/pytorch-adda/blob/master/models/discriminator.py
        # they use 2 outputs and do a softmax... Let's try a sigmoid instead

        return x




