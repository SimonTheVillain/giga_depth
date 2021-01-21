import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import matplotlib
from dataset.dataset_rendered_2 import DatasetRendered2
from experiments.lines.model_lines_CR8_n import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



class CompositeModel(nn.Module):
    def __init__(self, backbone, regressor):
        super(CompositeModel, self).__init__()
        self.backbone = backbone
        self.regressor = regressor

    def forward(self, x, x_gt=None, mask_gt=None):
        x = self.backbone(x)
        return self.regressor(x, x_gt, mask_gt)

dataset_path = "/media/simon/ssd_data/data/datasets/structure_core_unity_slice_100_35"
tgt_res = (1216, 896)#(1216, 896)
principal = (604, 457)
focal = 1.1154399414062500e+03
focal *= 0.5
principal = (principal[0] * 0.5, principal[1] * 0.5)
# according to the simulation in unity & the dotpattern extractor (check if this still holds true)
focal_projector = 850
res_projector = 1024

baselines = [0.0634 - 0.07501, 0.0634 - 0.0] # left, right
lines_only = True
is_npy = True

regressor = "trained_models/cr8_2021_32_std_4_regressor_chk.pt"
backbone = "trained_models/cr8_2021_32_std_4_backbone_chk.pt"

#regressor = "trained_models/cr8_2021_regressor_chk.pt"
#backbone = "trained_models/cr8_2021_backbone_chk.pt"

backbone = torch.load(backbone)
backbone.eval()

regressor = torch.load(regressor)
regressor.eval()

device = torch.cuda.current_device()

model = CompositeModel(backbone, regressor)
model.to(device)
model.eval()
dataset = DatasetRendered2(dataset_path, 0, 40000, tgt_res=tgt_res, is_npy=is_npy)

fig, axs = plt.subplots(4, 1)
for i, data in enumerate(dataset):
    bl = baselines[(i + 0) % 2]
    ir, x_gt, mask_gt = data
    with torch.no_grad():
        ir = torch.tensor(ir, device=device).unsqueeze(0)
        x, sigma_sq = model(ir)
        #print(x)
        ir = ir.cpu().numpy()
        x = x.cpu().numpy()
        sigma_sq =sigma_sq.cpu().numpy()

        x_range = np.arange(0, tgt_res[0] / 2).astype(np.float32)
        den = focal_projector * (x_range[np.newaxis, ...] - principal[0]) - focal * (x_gt[0, :, :] * res_projector - 511.5)
        d_gt = -np.divide(bl * (focal_projector * focal), den)
        d_gt = d_gt.clip(-20, 20)
        den = focal_projector * (x_range[np.newaxis, ...] - principal[0]) - focal * (x[0, :, :] * res_projector - 511.5)
        d = -np.divide(bl * (focal_projector * focal), den)
        d = d.clip(-20, 20)

        axs[0].imshow(ir[0, 0, :, :])
        axs[1].cla()
        axs[1].plot(x_gt.squeeze())
        axs[1].plot(x.squeeze())

        axs[2].cla()
        axs[2].plot(d.squeeze())
        axs[2].plot(d_gt.squeeze())

        #todo: the depth estimaion!
        axs[3].cla()
        axs[3].plot(np.sqrt(np.clip(sigma_sq.squeeze(), 0, 100)))
        # plt.ylabel('')
        plt.show(block=False)
        plt.waitforbuttonpress()
        plt.pause(5.)





