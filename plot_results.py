import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import matplotlib
from dataset.datasets import GetDataset
from experiments.lines.model_lines_CR8_n import *

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg


def display_pcl(z, offset, half_res=True):
    fx = 1115.44
    cxr = 604.0
    cyr = 896.0 * 0.5
    if half_res:
        fx = fx * 0.5
        cxr = cxr * 0.5
        cyr = cyr * 0.5
    print(z.shape)
    pts = []
    for i in range(0, z.shape[1]):
        for j in range(0, z.shape[2]):
            y = i + offset - cyr
            x = j - cxr
            depth = z[0, i, j]
            if 0 < depth < 20:
                pts.append([x*depth/fx, y*depth/fx, depth])
    xyz = np.array(pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])


class CompositeModel(nn.Module):
    def __init__(self, backbone, regressor):
        super(CompositeModel, self).__init__()
        self.backbone = backbone
        self.regressor = regressor

    def forward(self, x, x_gt=None):
        x = self.backbone(x)
        return self.regressor(x, x_gt)

dataset_path = "/media/simon/ssd_data/data/datasets/structure_core"
dataset_version = "structure_core_unity_4"

tgt_res = (1216, 896)#(1216, 896)
principal = (604, 457)
focal = 1.1154399414062500e+03
focal *= 0.5
principal = (principal[0] * 0.5, principal[1] * 0.5)
# according to the simulation in unity & the dotpattern extractor (check if this still holds true)
focal_projector = 850
res_projector = 1024

#baselines = [0.0634 - 0.07501, 0.0634 - 0.0] # right, left. the left camera has the higher baseline
#if dataset_version >= 3:
baselines = [0.0634, 0.0634 - 0.07501] # TODO: are the baselines really switched in the new dataset? is this right?
    # it really seems like the left camera has a higher baseline

lines_only = True
is_npy = True

show_pcl = False
show_full = True
rendered = True


regressor = "trained_models/cr8_2021_32_scaled_sigma_regressor_chk.pt"
backbone = "trained_models/cr8_2021_32_scaled_sigma_backbone_chk.pt"

regressor = "trained_models/cr8_2021_128_2_lr02_alpha1_regressor_chk.pt"
backbone = "trained_models/cr8_2021_128_2_lr02_alpha1_backbone_chk.pt"

regressor = "trained_models/cr8_short_128c_superclasses_regressor_chk.pt"
backbone = "trained_models/cr8_short_128c_superclasses_backbone_chk.pt"
input_height = 2*17+1


regressor = "trained_models/bb64_256c_16sc_256_8_lr02_regressor_chk.pt"
backbone = "trained_models/bb64_256c_16sc_256_8_lr02_backbone_chk.pt"

regressor = "trained_models/slice_2stage_class_43_regressor_chk.pt"
backbone = "trained_models/slice_2stage_class_43_backbone_chk.pt"

regressor = "trained_models/slice_2stage_class_50_regressor_chk.pt"
backbone = "trained_models/slice_2stage_class_50_backbone_chk.pt"

regressor = "trained_models/2stage_class_43_2_regressor_chk.pt"
backbone = "trained_models/2stage_class_43_2_backbone_chk.pt"


regressor = "trained_models/slice128_2stage_class_43_old_dataset_regressor_chk.pt"
backbone = "trained_models/slice128_2stage_class_43_old_dataset_backbone_chk.pt"


regressor = "trained_models/slice256_2stage_class_56_regressor.pt"
backbone = "trained_models/slice256_2stage_class_56_backbone.pt"

regressor = "trained_models/2stage_class_52_regressor.pt"
backbone = "trained_models/2stage_class_52_backbone.pt"

regressor = "trained_models/slice256_2stage_class_58_regressor_chk.pt"
backbone = "trained_models/slice256_2stage_class_58_backbone_chk.pt"


regressor = "trained_models/slice256_2stage_class_59_regressor_chk.pt"
backbone = "trained_models/slice256_2stage_class_59_backbone_chk.pt"


regressor = "trained_models/slice256_2stage_class_61_regressor_chk.pt"
backbone = "trained_models/slice256_2stage_class_61_backbone_chk.pt"


regressor = "trained_models/slice256_2stage_class_64_regressor_chk.pt"
backbone = "trained_models/slice256_2stage_class_64_backbone_chk.pt"


regressor = "trained_models/slice256_2stage_class_64_regressor_chk.pt"
backbone = "trained_models/slice256_2stage_class_64_backbone_chk.pt"


regressor = "trained_models/full_64_nolcn_regressor_chk.pt"
backbone = "trained_models/full_64_nolcn_backbone_chk.pt"




backbone = torch.load(backbone)
backbone.eval()

regressor = torch.load(regressor)
regressor.eval()

clip_from = 100
input_height = 128
if regressor.height == 448:
    input_height = 2*448
    clip_from = 0
    if not show_pcl:
        if show_full:
            fig, axs = plt.subplots(1, 2)
        else:
            fig, axs = plt.subplots(6, 1)
else:
    if not show_pcl:
        fig, axs = plt.subplots(6, 1)
device = torch.cuda.current_device()

model = CompositeModel(backbone, regressor)
model.to(device)
model.eval()

if rendered:
    dataset_path = "/media/simon/ssd_datasets/datasets/structure_core_unity_3"
    datasets, baselines, has_lr, focal, principal, src_res = GetDataset(dataset_path, is_npy=False, tgt_res=(1216, 896), version=dataset_version)
    dataset = datasets["val"]
else:
    dataset_path = "/media/simon/ssd_datasets/datasets/structure_core/sequences_combined"

    assert 0, "shit, this needs to be redone"
    #datasets, baselines, has_lr, focal, principal, src_res = DatasetCaptured(root_dir=dataset_path, from_ind=0, to_ind=800)


for i, ir in enumerate(dataset):
    side = "left"
    if has_lr:
        side = ["left", "right"][i % 2]

    bl = baselines[side]

    with torch.no_grad():
        if rendered:
            ir = ir[0] # for the rendered dataset ir is a tuple of ir, mask and gt
        ir = torch.tensor(ir, device=device).unsqueeze(0)
        print(ir.shape)

        #TODO: remove this debug!!!!!
        print("TODO: remove!!!!!!! Load one specific frame")
        ir_cv = cv2.imread("/media/simon/ssd_datasets/datasets/structure_core/sequences_ambient/sequences_home_2_ambient/004/ir3.png")
        ir = torch.tensor(ir_cv.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
        ir = ir[:, :, :, 1216:, 0] * (1.0 / 255.0)
        print(ir.shape)

        #ir = ir[:, :, 100+1:(100+17*2+1)+1, :]
        ir = ir[:, :, clip_from:(clip_from + input_height), :]

        x, sigma = model(ir.contiguous())
        #mask, sigma_est = sigma_estimator(ir)
        sigma = torch.sigmoid(sigma)

        #print(x)
        ir = ir.cpu().numpy()
        x = x.cpu().numpy()
        sigma = sigma.cpu().numpy()
        #sigma_est = sigma_est.cpu().numpy()
        #mask = mask.cpu().numpy()

        pad = 0.1#todo: make this a parameter that is stored within the network!
        #x = (x - pad) / (1.0 - 2.0 * pad)

        x_range = np.arange(0, x.shape[3]).astype(np.float32)
        x_range = x_range.reshape((1,) * 3 + (-1,))
        disp = x * x.shape[3] - x_range
        disp = disp * src_res[0] / disp.shape[3]
        d = focal[0] * bl / disp

        d = d.clip(-0.1, 5)
        if show_pcl:
            display_pcl(d, clip_from)
        else:
            if regressor.height == 448 and show_full:
                axs[0].cla()
                axs[0].set_title('IR Image')
                axs[0].imshow(ir[0, 0, :, :])

                axs[1].cla()
                axs[1].set_title('Depth')
                axs[1].imshow(d[0, 0, :, :])
            else:
                axs[0].set_title('IR Image')
                axs[0].imshow(ir[0, 0, :, :])


                axs[1].cla()
                axs[1].set_title('X single line')
                axs[1].plot(x[0, 0, 32, :].squeeze()*1024)

                axs[2].cla()
                axs[2].set_title('Depth single line')
                axs[2].plot(d[0, 0, 32, :].squeeze())

                axs[3].cla()
                axs[3].set_title('sigma')
                axs[3].plot(sigma[0, 0, 32, :].squeeze())

                #todo: the depth estimaion!
                axs[4].cla()
                axs[4].set_title('Depth')
                axs[4].imshow(d[0, 0, :, :])

                axs[5].cla()
                axs[5].set_title('confidence')
                axs[5].imshow(np.clip(sigma[0, 0, :, :], 0, 1))


            plt.show(block=False)
            plt.waitforbuttonpress()
            plt.pause(5.)





