import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock




class Discriminator1(nn.Module):

    def __init__(self, w, h):
        super(Discriminator1, self).__init__()
        #input dimensions: 67 + 128 + 2
        size = 67+128+2
        self.relublock1 = ResidualBlock(size, 3, 1, 4)
        #maxpool /2
        self.conv1 = nn.Conv2d(size, 128, 5, padding=2, padding_mode='same')

        self.relublock2 = ResidualBlock(128, 3, 1)
        self.relublock3 = ResidualBlock(128, 3, 1)
        #maxpool /4

        self.relublock4 = ResidualBlock(128, 3, 1)
        self.relublock5 = ResidualBlock(128, 3, 1)
        #maxpool /8

        self.relublock6 = ResidualBlock(128, 3, 1)
        self.relublock7 = ResidualBlock(128, 3, 1)
        #maxpool /16
        self.conv2 = nn.Conv2d(128, 32, 5, padding=2, padding_mode='same')

        w = w/16
        h = h/16
        self.fc1 = nn.Linear(128 * w * h,4096)
        self.fc2 = nn.Linear(4096,1)

        #fully connected layers?

    def forward(self, x, x_latent):
        # x has 2 channels
        # x_latent has # 67+128
        x = torch.cat(x, x_latent, 1)
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
        x = torch.flatten(x,1)
        x = F.leaky_relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x


