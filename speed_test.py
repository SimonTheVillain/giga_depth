import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from model.model_CR9hs import Model_CR9_hsn
from model.model_CR10hs import Model_CR10_hsn
from model.model_CR10_1hs import Model_CR10_1_hsn
from model.model_CR10_2hs import Model_CR10_2_hsn
from model.model_CR10_3hs import Model_CR10_3_hsn
from model.model_CR10_4hs import Model_CR10_4_hsn
from model.model_CR10_5hs import Model_CR10_5_hsn
from model.model_CR11 import Model_CR11_hn
from model.model_CR10hs_half import Model_CR10_hsn_half

from model.backbone_v1 import Backbone1
from model.backbone_v2 import Backbone2
from model.backbone_6_64 import Backbone6_64
from model.regressor_v1 import Regressor1
from model.regressor_v2 import Regressor2
from model.regressor_2Stage import Regressor2Stage
from torch.utils.data import DataLoader
import time

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class Model_test(nn.Module):

    def __init__(self):
        super(Model_test, self).__init__()
        self.start = nn.Conv2d(1, 16, 3, padding=1)
        self.conv1 = nn.Conv2d(16, 32, 5, padding=2)
        self.convdown = nn.Conv2d(32, 64, 5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv4 = nn.Conv2d(64, 64, 5, padding=2)

    def forward(self, x):

        x = F.leaky_relu(self.start(x))
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.convdown(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        return x

#the model/backbone used in many if not most experiments
class Model_test2(nn.Module):

    def __init__(self):
        super(Model_test2, self).__init__()
        self.start = nn.Conv2d(1, 8, 3, padding=1)
        self.conv1 = nn.Conv2d(8, 16, 5, padding=2)
        self.convdown = nn.Conv2d(16, 32, 5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv4 = nn.Conv2d(64, 64, 5, padding=2)
        self.bn_start = nn.BatchNorm2d(8)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn_down = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)


    def forward(self, x):

        x = F.leaky_relu(self.bn_start(self.start(x)))
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn_down(self.convdown(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        return x

# lets reduce the horizontal resolution in the second to first step!
class Model_test3(nn.Module):

    def __init__(self):
        super(Model_test3, self).__init__()
        self.start = nn.Conv2d(1, 8, 3, padding=1)
        self.conv1 = nn.Conv2d(8, 16, 5, padding=2, stride=(2, 1))# reduce lines early on (gets down from 51 to 47ms)
        self.convdown = nn.Conv2d(16, 32, 5, padding=2, stride=(1, 2))# reduce colums right after
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv4 = nn.Conv2d(64, 64, 5, padding=2)
        self.bn_start = nn.BatchNorm2d(8)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn_down = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.leaky_relu(self.bn_start(self.start(x)))
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn_down(self.convdown(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        return x

#go a bit more u-shaped
class Model_test4(nn.Module):

    def __init__(self):
        super(Model_test4, self).__init__()
        self.start = nn.Conv2d(1, 8, 3, padding=1)
        self.conv1 = nn.Conv2d(8, 16, 5, padding=2, stride=(2, 1))# reduce lines early on (gets down from 51 to 47ms)
        self.convdown = nn.Conv2d(16, 32, 5, padding=2, stride=(1, 2))# reduce colums right after

        self.conv_sub1 = nn.Conv2d(32, 64, 5, padding=2, stride=2)
        self.conv_sub2 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv_sub3 = nn.Conv2d(64, 64, 1, padding=0)
        self.conv_sub4 = nn.Conv2d(64, 64, 1, padding=0)
        self.conv_sub5 = nn.Conv2d(64, 32*4, 1, padding=0)#todo: should this one have a kernel size bigger 1?

        self.conv2 = nn.Conv2d(32, 16, 5, padding=2)
        self.conv3 = nn.Conv2d(16, 16, 5, padding=2)
        self.conv4 = nn.Conv2d(16, 32, 5, padding=2)


        self.bn_start = nn.BatchNorm2d(8)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn_down = nn.BatchNorm2d(32)

        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(32)

        self.bnsub1 = nn.BatchNorm2d(64)
        self.bnsub2 = nn.BatchNorm2d(64)
        self.bnsub3 = nn.BatchNorm2d(64)
        self.bnsub4 = nn.BatchNorm2d(64)
        self.bnsub5 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = F.leaky_relu(self.bn_start(self.start(x)))
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn_down(self.convdown(x)))
        x_l1 = x
        x = F.leaky_relu(self.bn2(self.conv2(x_l1)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))

        x_l1 = F.leaky_relu(self.bnsub1(self.conv_sub1(x_l1)))
        x_l1 = F.leaky_relu(self.bnsub2(self.conv_sub2(x_l1)))
        x_l1 = F.leaky_relu(self.bnsub3(self.conv_sub3(x_l1)))#these here are incredibly cheap
        x_l1 = F.leaky_relu(self.bnsub4(self.conv_sub4(x_l1)))
        x_l1 = F.leaky_relu(self.bnsub5(self.conv_sub5(x_l1)))
        x_l1 = x_l1.reshape((x_l1.shape[0], 32, 2, 2, x_l1.shape[2], x_l1.shape[3]))
        x_l1 = x_l1.permute((0, 1, 4, 2, 5, 3)).reshape((x_l1.shape[0], 32, x.shape[2], x.shape[3]))

        x = torch.cat((x, x_l1), dim=1)
        return x

# go a bit more u-shaped
class Model_test5(nn.Module):

    def __init__(self):
        super(Model_test5, self).__init__()
        self.start = nn.Conv2d(1, 8, 3, padding=1, stride=(2, 1))#putting this here gives us less than a millisecond
        self.conv1 = nn.Conv2d(8, 16, 5, padding=2)  # reduce lines early on (gets down from 51 to 47ms)
        self.convdown = nn.Conv2d(16, 32, 5, padding=2, stride=(1, 2))  # reduce colums right after

        self.conv_sub1 = nn.Conv2d(32, 64, 5, padding=2, stride=2)
        self.conv_sub2 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv_sub3 = nn.Conv2d(64, 64, 1, padding=0)
        self.conv_sub4 = nn.Conv2d(64, 64, 1, padding=0)
        self.conv_sub5 = nn.Conv2d(64, 32 * 4, 1, padding=0)  # todo: should this one have a kernel size bigger 1?

        self.convout = nn.Conv2d(64, 64, 3, padding=1)

        self.bn_start = nn.BatchNorm2d(8)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn_down = nn.BatchNorm2d(32)

        self.bnout = nn.BatchNorm2d(64)

        self.bnsub1 = nn.BatchNorm2d(64)
        self.bnsub2 = nn.BatchNorm2d(64)
        self.bnsub3 = nn.BatchNorm2d(64)
        self.bnsub4 = nn.BatchNorm2d(64)
        self.bnsub5 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = F.leaky_relu(self.bn_start(self.start(x)))
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn_down(self.convdown(x)))
        x_l1 = x

        x_l1 = F.leaky_relu(self.bnsub1(self.conv_sub1(x_l1)))
        x_l1 = F.leaky_relu(self.bnsub2(self.conv_sub2(x_l1)))
        x_l1 = F.leaky_relu(self.bnsub3(self.conv_sub3(x_l1)))  # these here are incredibly cheap
        x_l1 = F.leaky_relu(self.bnsub4(self.conv_sub4(x_l1)))
        x_l1 = F.leaky_relu(self.bnsub5(self.conv_sub5(x_l1)))
        x_l1 = x_l1.reshape((x_l1.shape[0], 32, 2, 2, x_l1.shape[2], x_l1.shape[3]))
        x_l1 = x_l1.permute((0, 1, 4, 2, 5, 3)).reshape((x_l1.shape[0], 32, x.shape[2], x.shape[3]))

        x = torch.cat((x, x_l1), dim=1)
        x = F.leaky_relu(self.bnout(self.convout(x)))

        return x

#same but more 3x3 convolutions and only 2 5x5 ones
# go a bit more u-shaped
class Model_test6(nn.Module):

    def __init__(self):
        super(Model_test6, self).__init__()
        self.start = nn.Conv2d(1, 8, 3, padding=1,
                               stride=(2, 1))  # putting this here gives us less than a millisecond
        self.conv1 = nn.Conv2d(8, 16, 3, padding=1)  # reduce lines early on (gets down from 51 to 47ms)
        self.convdown = nn.Conv2d(16, 32, 5, padding=2, stride=(1, 2))  # reduce colums right after

        self.conv_sub1 = nn.Conv2d(32, 64, 5, padding=2, stride=2)
        self.conv_sub2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_sub3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_sub4 = nn.Conv2d(64, 64, 1, padding=0)
        self.conv_sub5 = nn.Conv2d(64, 32 * 4, 1, padding=0)  # todo: should this one have a kernel size bigger 1?

        self.convout = nn.Conv2d(64, 64, 3, padding=1)

        self.bn_start = nn.BatchNorm2d(8)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn_down = nn.BatchNorm2d(32)

        self.bnout = nn.BatchNorm2d(64)

        self.bnsub1 = nn.BatchNorm2d(64)
        self.bnsub2 = nn.BatchNorm2d(64)
        self.bnsub3 = nn.BatchNorm2d(64)
        self.bnsub4 = nn.BatchNorm2d(64)
        self.bnsub5 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = F.leaky_relu(self.bn_start(self.start(x)))
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn_down(self.convdown(x)))
        x_l1 = x

        x_l1 = F.leaky_relu(self.bnsub1(self.conv_sub1(x_l1)))
        x_l1 = F.leaky_relu(self.bnsub2(self.conv_sub2(x_l1)))
        x_l1 = F.leaky_relu(self.bnsub3(self.conv_sub3(x_l1)))  # these here are incredibly cheap
        x_l1 = F.leaky_relu(self.bnsub4(self.conv_sub4(x_l1)))
        x_l1 = F.leaky_relu(self.bnsub5(self.conv_sub5(x_l1)))
        x_l1 = x_l1.reshape((x_l1.shape[0], 32, 2, 2, x_l1.shape[2], x_l1.shape[3]))
        x_l1 = x_l1.permute((0, 1, 4, 2, 5, 3)).reshape((x_l1.shape[0], 32, x.shape[2], x.shape[3]))

        x = torch.cat((x, x_l1), dim=1)
        x = F.leaky_relu(self.bnout(self.convout(x)))

        return x

class CompositeModel(nn.Module):
    def __init__(self, backbone, regressor):
        super(CompositeModel, self).__init__()
        self.backbone = backbone
        self.regressor = regressor



    def forward(self, x, x_gt=None):
        x = self.backbone(x)
        x, mask, class_loss = self.regressor(x, x_gt)
        return x, mask, class_loss

slices = 8
class_count = 128
core_image_height = 112
padding = Model_CR10_3_hsn.padding()
pad_top = False
pad_bottom = False
crop_div = 1
crop_res = (core_image_height + 30, 1216/crop_div)

core_image_height = 896
crop_res = (core_image_height + 30 -30, 1216/crop_div)

runs = 10
#model = Model_CR10_hsn_half(slices, class_count, core_image_height, pad_top, pad_bottom)
#model = Model_CR10_2_hsn(slices, class_count, core_image_height)
#model = Model_CR10_3_hsn(class_count, core_image_height)
model = Model_CR10_5_hsn(class_count, core_image_height)

backbone = Backbone6_64()
regressor = Regressor2Stage(64, 448, 608)
model = CompositeModel(backbone, regressor)
#model = Model_CR11_hn(core_image_height, class_count)
model = Model_test5()
input_channels = 1
batches = 1
warm_up = True
torch.backends.cudnn.benchmark = True


model.cuda()
print(f"nr model params: {get_n_params(model)}")

for half_precision in [False, True]:#, True]:

    if half_precision:
        model.half()
    else:
        model.float()
    model.eval()

    with torch.no_grad():
        test = torch.rand((batches, input_channels, int(crop_res[0]), int(crop_res[1])), dtype=torch.float32).cuda()
        if half_precision:
            test = test.half()
        else:
            test = test.float()
        print(test.device)
        print(torch.cuda.get_device_name(test.device))
        if warm_up:
            model(test)
        torch.cuda.synchronize()
        tsince = int(round(time.time() * 1000))
        for i in range(0, runs):
            model(test)
            torch.cuda.synchronize()
        ttime_elapsed = int(round(time.time() * 1000)) - tsince
        print('test time elapsed {}ms'.format(ttime_elapsed / runs))
    model.train()