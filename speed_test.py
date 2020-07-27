import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import os
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
from model.regressor_v1 import Regressor1
from model.regressor_v2 import Regressor2
from torch.utils.data import DataLoader
import time


class Model_test(nn.Module):

    def __init__(self):
        super(Model_test, self).__init__()
        self.start = nn.Conv2d(8, 64, 3, padding=1)
        self.conv1 = nn.Conv2d(64, 64, 7, padding=3)
        self.conv2 = nn.Conv2d(64, 64, 7, padding=3)
        self.conv3 = nn.Conv2d(64, 128, 7, padding=3)
        self.conv4 = nn.Conv2d(128, 256, 7, padding=3)
        self.conv5 = nn.Conv2d(256, 8, 7, padding=3)

    def forward(self, x):

        x = F.leaky_relu(self.start(x))
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
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

half_precision = True
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

backbone = Backbone2()
regressor = Regressor1(128, 448, 608)
regressor = Regressor2(128, 448, 608)
model = CompositeModel(backbone, regressor)
#model = Model_CR11_hn(core_image_height, class_count)
#model = Model_test()
input_channels = 1
warm_up = True
torch.backends.cudnn.benchmark = True


model.cuda()

for half_precision in [False]:#, True]:

    if half_precision:
        model.half()
    else:
        model.float()
    model.eval()

    with torch.no_grad():
        test = torch.rand((1, input_channels, int(crop_res[0]), int(crop_res[1])), dtype=torch.float32).cuda()
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