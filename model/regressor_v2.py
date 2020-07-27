import torch
import torch.nn as nn
import torch.nn.functional as F
from model.cuda_cond_mul.cond_mul import CondMul
from model.residual_block import ResidualBlock_shrink


#TODO: this regressor
class Regressor2(nn.Module):

    def __init__(self, classes=128, height=448, width=608):
        super(Regressor2, self).__init__()
        #per line weights for classes
        self.half_height = int(height)
        self.half_width = width
        self.classes = classes
        self.conv_c = nn.Conv2d(128 * self.half_height, self.classes * self.half_height, 1, padding=0, groups=self.half_height)
        self.conv_1 = CondMul(self.classes * self.half_height, 128, 32)
        self.conv_2 = CondMul(self.classes * self.half_height, 128, 32)
        self.conv_rc = CondMul(self.classes * self.half_height, 32, 2)



    def calc_x_pos(self, class_inds, regression, class_count):
        regression = regression * (1.0 / class_count)
        x = class_inds * (1.0 / class_count) + regression
        return x

    def forward(self, x, x_gt):
        batches = x.shape[0]
        device = x.device
        x_1 = x.transpose(1, 2)
        x_1 = x_1.reshape((x_1.shape[0], 128 * self.half_height, 1, x_1.shape[3]))

        classes = F.leaky_relu(self.conv_c(x_1))

        classes = classes.reshape((classes.shape[0], self.half_height, self.classes, classes.shape[3]))
        #classes = classes.transpose(1, 2)
        #classes = F.softmax(classes, dim=1)

        loss_class = None
        if x_gt is None:
            inds = classes.argmax(dim=2).unsqueeze(2)
            inds = inds.transpose(1, 2) #first finding argmax and only then transposing saves us some memory bandwidth
        else:
            classes = classes.transpose(1, 2)
            classes = F.softmax(classes, dim=1)
            inds = torch.clamp((x_gt * self.classes).type(torch.int64), 0, self.classes - 1)
            gt_class_label = inds.squeeze(1)
            loss_class = F.cross_entropy(classes, gt_class_label, reduction='none')

        offset = torch.arange(0, self.half_height, device=device)
        offset = offset.unsqueeze(0).unsqueeze(0).unsqueeze(3)
        ind_shape = inds.shape
        inds_original = inds
        inds = inds + offset * self.classes
        inds = inds.reshape(-1).type(torch.int32)


        x_2 = x.permute([0, 2, 3, 1])
        x_2 = x_2.reshape((x_2.shape[0] * x_2.shape[1] * x_2.shape[2], x_2.shape[3]))
        x_2 = F.leaky_relu(self.conv_1(x_2.contiguous(), inds))
        x_2 = F.leaky_relu(self.conv_2(x_2, inds))
        x_2 = F.leaky_relu(self.conv_rc(x_2, inds))
        x_2 = x_2.reshape((batches, self.half_height, self.half_width, 2))
        x = x_2.permute([0, 3, 1, 2])

        mask = x[:, 1, :, :]
        x = x[:, 0, :, :]

        x = self.calc_x_pos(inds_original, x, self.classes)

        #TODO: find out what else we need here!
        return x, mask, loss_class


