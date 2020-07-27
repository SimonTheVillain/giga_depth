import torch
import torch.nn as nn
import torch.nn.functional as F


# basic regressor... hopefully not anything you would
class Regressor1(nn.Module):
    def __init__(self, classes=128, height=448, width=608):
        super(Regressor1, self).__init__()
        #per line weights for classes
        half_height = int(height)
        self.classes = classes
        self.conv_c = nn.Conv2d(128 * half_height, self.classes * half_height, 1, padding=0, groups=half_height)
        self.conv_r = nn.Conv2d(128 * half_height, self.classes * half_height, 1, padding=0, groups=half_height)

        self.conv_m = nn.Conv2d(128, 1, 1)

    def calc_x_pos(self, class_inds, regression, class_count):
        regression = regression * (1.0 / class_count)
        x = class_inds * (1.0 / class_count) + regression
        return x

    def forward(self, x, x_gt=None):
        mask = F.leaky_relu(self.conv_m(x))

        half_height = int(x.shape[2])
        x_1 = x.transpose(1, 2)
        x_1 = x_1.reshape((x_1.shape[0], 128 * half_height, 1, x_1.shape[3]))
        classes = F.leaky_relu(self.conv_c(x_1))
        classes = classes.reshape((classes.shape[0], half_height, self.classes, classes.shape[3]))
        classes = classes.transpose(1, 2)
        classes = F.softmax(classes, dim=1)

        regressions = self.conv_r(x_1)
        regressions = regressions.reshape((regressions.shape[0], half_height, self.classes, regressions.shape[3]))
        regressions = regressions.transpose(1, 2)

        loss_class = None
        if x_gt is None:
            inds = classes.argmax(dim=1).unsqueeze(1)
        else:

            inds = torch.clamp((x_gt * self.classes).type(torch.int64), 0, self.classes - 1)
            gt_class_label = inds.squeeze(1)
            loss_class = F.cross_entropy(classes, gt_class_label, reduction='none')

        #regressions = regressions.gather(1, inds)
        regression = torch.gather(regressions, 1, inds)
        x = self.calc_x_pos(inds, regression, self.classes)
        return x, mask, loss_class


