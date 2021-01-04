import torch
import torch.nn as nn
import torch.nn.functional as F
from model.cuda_cond_mul.cond_mul import CondMul
from model.residual_block import ResidualBlock_shrink


#Regressor
class RegressorBranchless(nn.Module):

    def __init__(self, input_channels=64, height=448, width=608):
        super(RegressorBranchless, self).__init__()
        #per line weights for classes
        self.height = int(height)
        self.width = int(width)
        self.input_channels = input_channels
        self.depth = [128, 128, 1024]
        #the first stage is supposed to output 32 classes (+ in future 32 variables that might help in the next steps)

        self.stage_1 = nn.Conv2d(self.input_channels, self.depth[0], 3, padding=1)
        self.stage_2 = nn.Conv2d(self.depth[0] * self.height,
                                 self.depth[1] * self.height,
                                 1, padding=0, groups=self.height)
        self.stage_3 = nn.Conv2d(self.depth[1] * self.height,
                                 self.depth[2] * self.height,
                                 1, padding=0, groups=self.height)
        self.stage_4 = nn.Conv2d(self.depth[2] * self.height,
                                 2 * self.height,
                                 1, padding=0, groups=self.height)

    def forward(self, x, x_gt=None, mask_gt=None):
        # prevent indices in the groundtruth, that's out of bounds

        x = F.leaky_relu(self.stage_1(x))


        # go from (b, c, h, w) to (b, h, c, w)
        x = x.transpose(1, 2)

        # go from (b, h, c, w) to (b, h * c, 1, w)
        x = x.reshape((x.shape[0], self.depth[0] * self.height, 1, x.shape[3]))
        x = F.leaky_relu(self.stage_2(x))
        x = F.leaky_relu(self.stage_3(x))
        x = self.stage_4(x)

        # go from (b, h * c, 1, w) to (b, h, c, w)
        x = x.reshape((x.shape[0], self.height, 2, self.width))

        # go from (b, h, c, w) to (b, c, h, w)
        x = x.transpose(1, 2)

        mask = F.leaky_relu(x[:, 1, :, :])

        x = x[:, 0, :, :]
        if x_gt is not None:
            class_losses = []

        if x_gt is None:
            return x, mask
        else:
            return x, mask, class_losses


