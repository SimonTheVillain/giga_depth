import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
import cv2
import numpy as np

#Work in the parent directory
import os
import model
from model.composite_model import CompositeModel
import torch
import os
import cv2
import numpy as np
import re
from pathlib import Path

def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]

    return color_range.reshape(256, 1, 3)

def color_code(im, start, stop):
    msk = np.zeros_like(im)
    msk[np.logical_and(start < im, im < stop)] = 1.0
    im = np.clip(im, start, stop)
    im = (im-start) / float(stop-start)
    im = im * 255.0
    im = im.astype(np.uint8)
    im = cv2.applyColorMap(im, get_mpl_colormap("viridis"))
    im[msk != 1.0] = 0
    return im


regressor_model_pth = "trained_models/full_66_j1_regressor_chk.pt"
backbone_model_pth = "trained_models/full_66_j1_backbone_chk.pt"


device = "cuda:0"
backbone = torch.load(backbone_model_pth, map_location=device)
backbone.eval()
regressor = torch.load(regressor_model_pth, map_location=device)
regressor.eval()
model = CompositeModel(backbone, regressor)



regressor_model_pth = "trained_models/full_66_lcn_j4_regressor_chk.pt"
backbone_model_pth = "trained_models/full_66_lcn_j4_backbone_chk.pt"
backbone = torch.load(backbone_model_pth, map_location=device)
backbone.eval()
regressor = torch.load(regressor_model_pth, map_location=device)
regressor.eval()
model_jitter = CompositeModel(backbone, regressor)


mode = "captured"#"rendered" rendered_shapenet or captured
half_res = False


if mode == "captured":
    tgt_res = (1216, 896)
    tgt_cxy = (604, 457)
    # the focal length is shared between src and target frame
    focal = 1.1154399414062500e+03

    rr = (tgt_res[0], 0, tgt_res[0], tgt_res[1])
    path = "/media/simon/ssd_datasets/datasets/structure_core_photoneo_test"
    path_out = "/media/simon/ssd_datasets/datasets/structure_core_photoneo_test_results/GigaDepth66"
    path_out = "/media/simon/ssd_datasets/datasets/structure_core_photoneo_test_results/GigaDepth66_domain_transfer"


    path = "/home/simon/datasets/structure_core_photoneo_test"
    path = "/home/simon/datasets/structure_core/sequences_combined_all"
    path_out = "/home/simon/Pictures/images_paper/supplemental/jitter"

    #path = "/media/simon/ssd_datasets/datasets/structure_core/sequences_combined_all"
    #path_out = "/media/simon/ssd_datasets/datasets/structure_core/sequences_combined_all_GigaDepth66LCN"

    folders = os.listdir(path)
    scenes = [x for x in folders if os.path.isdir(Path(path) / x)]

    scenes.sort()
    paths = []
    for scene in scenes:
        tgt_path = Path(path_out) / scene
        for i in range(4):
            src_pth = Path(path) / scene / f"ir{i}.png"
            tgt_pth = Path(tgt_path) / f"{i}"
            paths.append((src_pth, tgt_pth))


paths = [paths[i] for i in [40*4, 60*4, 61*4, 62*4, 63*4, 64*4, 65*4, 66*4, 67*4, 68*4]]
ind = 0
with torch.no_grad():
    for p, p_tgt in paths:
        ind +=1
        print(p)
        irl = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if len(irl.shape) == 3:
            # the rendered images are 3 channel bgr
            irl = cv2.cvtColor(irl, cv2.COLOR_BGR2GRAY)
        else:
            # the rendered images are 16
            irl = irl / 255.0
        irl = irl[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]]
        irl = irl.astype(np.float32) * (1.0 / 255.0)

        if half_res:
            irl = cv2.resize(irl, (int(irl.shape[1] / 2), int(irl.shape[0] / 2)))
            irl = irl[:448, :608]
        cv2.imshow("irleft", irl)
        p = f"{path_out}/{ind}_ir.png"
        cv2.imwrite(p, irl*255)
        irl = torch.tensor(irl).cuda().unsqueeze(0).unsqueeze(0)

        #run local contrast normalization (LCN)
        x, msk = model(irl)
        x = x[0, 0, :, :]
        x = x * x.shape[1]
        x_0 = torch.arange(0, x.shape[1]).unsqueeze(0).to(device)
        x -= x_0
        x *= -1.0

        result = x.cpu()[:, :].numpy()
        msk = np.clip(msk.cpu()[0, 0, :, :].numpy()*255, 0, 255).astype(np.uint8)

        #result = coresup_pred.cpu()[0, 0, :, :].numpy()


        result = color_code(result, 10.0, 100.0)
        cv2.imshow("result", result)
        p = f"{path_out}/{ind}_jitter.png"
        cv2.imwrite(p, result)


        x, msk = model_jitter(irl)
        x = x[0, 0, :, :]
        x = x * x.shape[1]
        x_0 = torch.arange(0, x.shape[1]).unsqueeze(0).to(device)
        x -= x_0
        x *= -1.0

        result = x.cpu()[:, :].numpy()

        result = color_code(result, 10.0, 100.0)
        cv2.imshow("result2", result)
        p = f"{path_out}/{ind}_full.png"
        cv2.imwrite(p, result)


        cv2.waitKey(1)