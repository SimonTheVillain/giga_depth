import matplotlib
matplotlib.use('Agg')
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
from model.composite_model import CompositeModel
from model.backboneSliced import *
from model.uNet import UNet

import re

warm_up = True
runs = 10
src_res = (1401, 1001)
src_cxy = (700, 500)
tgt_res = (1216, 896)
tgt_cxy = (604, 457)
# the focal length is shared between src and target frame
focal = 1.1154399414062500e+03

rr = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1], tgt_res[0], tgt_res[1])

regressor_model_pth = "trained_models/full_66_lcn_j4_regressor_chk.pt"
backbone_model_pth = "trained_models/full_66_lcn_j4_backbone_chk.pt"

#regressor_model_pth = "trained_models/full_65_nolcn_jitter4_regressor_chk.pt"
#backbone_model_pth = "trained_models/full_65_nolcn_jitter4_backbone_chk.pt"

device = "cuda:0"
backbone = torch.load(backbone_model_pth, map_location=device)
backbone.eval()
regressor = torch.load(regressor_model_pth, map_location=device)
regressor.eval()
model = CompositeModel(backbone, regressor)

if False:
    constructor = lambda pad, channels, downsample: BackboneU5Slice(pad=pad, in_channels=channels)
    backboneType = BackboneU5Slice
    backbone = BackboneSlicer(backboneType, constructor,
                              4,
                              lcn=False,
                              downsample_output=True)
    backbone.cuda()
    model = backbone

if True:
    backboneType = BackboneSlice
    constructor = lambda pad, channels, downsample: BackboneSlice(
        channels=[],#[8, 16],
        kernel_sizes=[],
        channels_sub=[16, 24, 32, 40, 64, 64, 96, 160],#[16, 32, 32, 64, 64+32+32],#[32, 32, 64, 64],
        kernel_sizes_sub=[5, 3, 3, 3, 3, 5, 3, 3],
        use_bn=True,
        pad=pad, channels_in=channels)#,downsample=True)
    if False:
        constructor = lambda pad, channels, downsample: BackboneSlice(
            channels=[],#[8, 16],
            kernel_sizes=[],
            channels_sub=[16, 32, 32, 64, 128],#[16, 32, 32, 64, 64+32+32],#[32, 32, 64, 64],
            kernel_sizes_sub=[5, 5, 5, 5, 5],
            use_bn=True,
            pad=pad, channels_in=channels)#,downsample=True)
    backbone = constructor('both', 2, True)
    backbone = BackboneSlicer(backboneType, constructor,
                              1,
                              in_channels=1,
                              downsample_output=True)
    #backbone = UNet(1, 64, True, 0.5)
    backbone.cuda()
    model = backbone
path = "/home/simon/datasets/structure_core_unity_test"
inds = os.listdir(path)
inds = [re.search(r'\d+', s).group() for s in inds]
inds = set(inds)
inds = list(inds)
inds.sort()
paths = []
for ind in inds:
    paths.append((path + f"/{ind}_left.jpg"))

for half_precision in [False, True]:#, True]:

    p = paths[0]
    irl = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if len(irl.shape) == 3:
        # the rendered images are 3 channel bgr
        irl = cv2.cvtColor(irl, cv2.COLOR_BGR2GRAY)
    else:
        # the rendered images are 16
        irl = irl / 255.0
    irl = irl[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]]
    irl = irl.astype(np.float32) * (1.0 / 255.0)
    irl = torch.tensor(irl).cuda()
    test = irl.unsqueeze(0).unsqueeze(0)
    if half_precision:
        model.half()
    else:
        model.float()
    model.eval()

    with torch.no_grad():
        #half_precision = True
        if half_precision:
            model.half()
            model.half_precision = True
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
            #output, _ = model(test)
            torch.cuda.synchronize()
        ttime_elapsed = int(round(time.time() * 1000)) - tsince
        print(f"test time elapsed {ttime_elapsed / runs} ms")

        #output = output[0,0,:,:].cpu().detach().numpy()
        #cv2.imshow("input", irl.cpu().numpy())
        #cv2.imshow("output", output)
        #cv2.waitKey()
    model.train()