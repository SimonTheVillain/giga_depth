import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
dataset_path_results = "/media/simon/ssd_data/data/datasets/structure_core_unity_slice_100_35_results"
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

regressor = "trained_models/cr8_2021_32_scaled_sigma_regressor_chk.pt"
backbone = "trained_models/cr8_2021_32_scaled_sigma_backbone_chk.pt"
regressor = "trained_models/cr8_2021_128_2_lr02_alpha1_regressor_chk.pt"
backbone = "trained_models/cr8_2021_128_2_lr02_alpha1_backbone_chk.pt"

regressor = "trained_models/cr8_short_128c_superclasses_regressor_chk.pt"
backbone = "trained_models/cr8_short_128c_superclasses_backbone_chk.pt"

regressor = "trained_models/cr8_short_bb32_512c_32sc_384_8_latent_nocat_regressor_chk.pt"
backbone = "trained_models/cr8_short_bb32_512c_32sc_384_8_latent_nocat_backbone_chk.pt"


#regressor = "trained_models/line_bb128_32c1_32c2_32sc_128_32_lr01_alpha100_regressor_chk.pt"
#backbone = "trained_models/line_bb128_32c1_32c2_32sc_128_32_lr01_alpha100_backbone_chk.pt"


#regressor = "trained_models/line_bb128_16_16_16_32sc_128_32_lr01_alpha200_regressor_chk.pt"
#backbone = "trained_models/line_bb128_16_16_16_32sc_128_32_lr01_alpha200_backbone_chk.pt"

regressor = "trained_models/line_bb64_16_16_16c123_128_128bb_64sc_128_32reg_lr01_alpha200_regressor_chk.pt"
backbone = "trained_models/line_bb64_16_16_16c123_128_128bb_64sc_128_32reg_lr01_alpha200_backbone_chk.pt"


regressor = "trained_models/line_bb64_16_16_8c123_32_32_32_256bb_128sc_128_32reg_lr01_alpha200_regressor_chk.pt"
backbone = "trained_models/line_bb64_16_16_8c123_32_32_32_256bb_128sc_128_32reg_lr01_alpha200_backbone_chk.pt"

regressor = "trained_models/line_bb64_16_16_8c123_32_32_32_256bb_2048sc_128_32reg_lr01_alpha200_regressor_chk.pt"
backbone = "trained_models/line_bb64_16_16_8c123_32_32_32_256bb_2048sc_128_32reg_lr01_alpha200_backbone_chk.pt"

regressor = "trained_models/line_bb64_16_16_8c123_32_32_32_256bb_2048sc_128_32reg_lr01_alpha200_1nn_regressor_chk.pt"
backbone = "trained_models/line_bb64_16_16_8c123_32_32_32_256bb_2048sc_128_32reg_lr01_alpha200_1nn_backbone_chk.pt"

regressor = "trained_models/line_bb64_16_14_12c123_32_32_32_128bb_672sc_128_32reg_lr01_alpha200_1nn_regressor_chk.pt"
backbone = "trained_models/line_bb64_16_14_12c123_32_32_32_128bb_672sc_128_32reg_lr01_alpha200_1nn_backbone_chk.pt"

regressor = "trained_models/line_bb64_16_14_12c123_32_32_32_64bb_168sc_64_16reg_lr01_alpha100_1nn_regressor_chk.pt"
backbone = "trained_models/line_bb64_16_14_12c123_32_32_32_64bb_168sc_64_16reg_lr01_alpha100_1nn_backbone_chk.pt"

sigma_estimator = "trained_models/sigma_mask_scaled_chk.pt"
sigma_estimator = torch.load(sigma_estimator)
sigma_estimator.eval()

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
dataset = DatasetRendered2(dataset_path, 0, 40000, tgt_res=tgt_res, is_npy=is_npy, result_dir=dataset_path_results)

fig, axs = plt.subplots(5, 1)
for i, data in enumerate(dataset):
    bl = baselines[(i + 0) % 2]
    ir, x_gt, mask_gt, x_results = data
    with torch.no_grad():
        ir = torch.tensor(ir, device=device).unsqueeze(0)
        x, sigma = model(ir.contiguous())
        mask, sigma_est = sigma_estimator(ir)

        #print(x)
        ir = ir.cpu().numpy()
        x = x.cpu().numpy()
        sigma = sigma.cpu().numpy()
        sigma_est = sigma_est.cpu().numpy()
        mask = mask.cpu().numpy()

        x_range = np.arange(0, tgt_res[0] / 2).astype(np.float32)
        den = focal_projector * (x_range[np.newaxis, ...] - principal[0]) - focal * (x_gt[0, :, :] * res_projector - 511.5)
        d_gt = -np.divide(bl * (focal_projector * focal), den)
        d_gt = d_gt.clip(-20, 20)
        den = focal_projector * (x_range[np.newaxis, ...] - principal[0]) - focal * (x[0, :, :] * res_projector - 511.5)
        d = -np.divide(bl * (focal_projector * focal), den)
        d = d.clip(-20, 20)

        x_n = x_gt + (np.random.random(x_gt.shape) - 0.5) * 0.1/1024.0
        den = focal_projector * (x_range[np.newaxis, ...] - principal[0]) - focal * (x_n[0, :, :] * res_projector - 511.5)
        d_n = -np.divide(bl * (focal_projector * focal), den)
        d_n = d_n.clip(-20, 20)


        axs[0].set_title('IR Image')
        axs[0].imshow(ir[0, 0, :, :])


        axs[1].cla()
        axs[1].set_title('X single line')
        axs[1].plot(x_gt.squeeze()*1024)
        axs[1].plot(x.squeeze()*1024)
        #axs[1].plot(x_results.squeeze()*1024)

        axs[2].cla()
        axs[2].set_title('Depth single line')
        axs[2].plot(d_gt.squeeze())
        axs[2].plot(d.squeeze())

        #todo: the depth estimaion!
        axs[3].cla()
        axs[3].set_title('delta depth')
        axs[3].plot((d_gt.squeeze() - d.squeeze()).clip(-1, 1))
        #axs[3].plot((d_gt.squeeze() - d_n.squeeze()).clip(-1, 1))
        #axs[3].plot(sigma.squeeze().clip(0, 10))
        #axs[3].plot(sigma_est.squeeze())

        axs[4].cla()
        axs[4].set_title('delta x')
        axs[4].plot(((x_gt.squeeze()-x.squeeze())*1024).clip(-1, 1))
        #axs[4].plot((np.abs(x_gt.squeeze() - x_results.squeeze()) < 20/1024))
        #axs[4].plot(mask.squeeze())
        # plt.ylabel('')
        plt.show(block=False)
        plt.waitforbuttonpress()
        plt.pause(5.)





