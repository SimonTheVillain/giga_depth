import matplotlib
matplotlib.use('Agg')
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

path_src = "/media/simon/ssd_datasets/datasets/structure_core_unity_test"
path_results = "/media/simon/ssd_datasets/datasets/structure_core_unity_test_results/GigaDpeth"

regressor_model_pth = "trained_models/full_64_nolcn_jitter5_regressor.pt"
backbone_model_pth = "trained_models/full_64_nolcn_jitter5_backbone.pt"

regressor_model_pth = "trained_models/full_65_nolcn_jitter4_regressor.pt"
backbone_model_pth = "trained_models/full_65_nolcn_jitter4_backbone.pt"


regressor_model_pth = "trained_models/full_66_lcn_j4_regressor_chk.pt"
backbone_model_pth = "trained_models/full_66_lcn_j4_backbone_chk.pt"

regressor_model_pth = "trained_models/full_66_j4_regressor_chk.pt"
backbone_model_pth = "trained_models/full_66_j4_backbone_chk.pt"

regressor_model_pth = "trained_models/full_67_regressor_chk.pt"
regressor_conv_model_pth = "trained_models/full_67_regressor_conv_chk.pt"
backbone_model_pth = "trained_models/full_67_backbone_chk.pt"

regressor_model_pth = "trained_models/full_66_lcn_j4_regressor_chk.pt"
backbone_model_pth = "trained_models/full_66_lcn_j4_backbone_chk.pt"

device = "cuda:0"
backbone = torch.load(backbone_model_pth, map_location=device)
backbone.eval()
regressor = torch.load(regressor_model_pth, map_location=device)
regressor.eval()
model = CompositeModel(backbone, regressor)
regressor_conv = False

use_conv = False
mode = "rendered_shapenet"#"rendered" rendered_shapenet or captured
half_res = False
if use_conv:
    regressor_conv = torch.load(regressor_conv_model_pth, map_location=device)
    regressor_conv.eval()

    model = CompositeModel(backbone, regressor, regressor_conv)


if mode == "rendered":
    src_res = (1401, 1001)
    src_cxy = (700, 500)
    tgt_res = (1216, 896)
    tgt_cxy = (604, 457)
    # the focal length is shared between src and target frame
    focal = 1.1154399414062500e+03

    rr = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1], tgt_res[0], tgt_res[1])
    path = "/media/simon/ssd_datasets/datasets/structure_core_unity_test"

    path = "/media/simon/ssd_datasets/datasets/structure_core_unity_test"
    path_out = "/media/simon/ssd_datasets/datasets/structure_core_unity_test_results/GigaDepth67"
    inds = os.listdir(path)
    inds  = [re.search(r'\d+', s).group() for s in inds]
    inds = set(inds)
    inds = list(inds)
    inds.sort()
    paths = []
    for ind in inds:
        paths.append((path + f"/{ind}_left.jpg", path_out + f"/{int(ind):05d}"))

if mode == "rendered_shapenet":

    tgt_res = (640, 480)
    rr = (0, 0, tgt_res[0], tgt_res[1])
    regressor_model_pth = "trained_models/full_66_shapenet_regressor_chk.pt"
    backbone_model_pth = "trained_models/full_66_shapenet_backbone_chk.pt"
    device = "cuda:0"
    backbone = torch.load(backbone_model_pth, map_location=device)
    backbone.eval()
    regressor = torch.load(regressor_model_pth, map_location=device)
    regressor.eval()
    model = CompositeModel(backbone, regressor)

    path = "/media/simon/ssd_datasets/datasets/shapenet_rendered_compressed_test/syn"
    path_out = "/media/simon/ssd_datasets/datasets/shapenet_rendered_compressed_test_results/GigaDepth"

    folders = os.listdir(path)
    scenes = [x for x in folders if os.path.isdir(Path(path) / x)]

    scenes.sort()
    paths = []
    for scene in scenes:
        tgt_path = Path(path_out) / scene
        if not os.path.exists(tgt_path):
            os.mkdir(tgt_path)
        for i in range(4):
            src_pth = Path(path) / scene / f"im0_{i}.png"
            tgt_pth = Path(tgt_path) / f"{i}"
            paths.append((src_pth, tgt_pth))

if mode == "captured":
    tgt_res = (1216, 896)
    tgt_cxy = (604, 457)
    # the focal length is shared between src and target frame
    focal = 1.1154399414062500e+03

    rr = (tgt_res[0], 0, tgt_res[0], tgt_res[1])
    path = "/media/simon/ssd_datasets/datasets/structure_core_photoneo_test"
    path_out = "/media/simon/ssd_datasets/datasets/structure_core_photoneo_test_results/GigaDepth66"

    path = "/media/simon/ssd_datasets/datasets/structure_core/sequences_combined_all"
    path_out = "/media/simon/ssd_datasets/datasets/structure_core/sequences_combined_all_GigaDepth66LCN"

    folders = os.listdir(path)
    scenes = [x for x in folders if os.path.isdir(Path(path) / x)]

    scenes.sort()
    paths = []
    for scene in scenes:
        tgt_path = Path(path_out) / scene
        if not os.path.exists(tgt_path):
            os.mkdir(tgt_path)
        for i in range(4):
            src_pth = Path(path) / scene / f"ir{i}.png"
            tgt_pth = Path(tgt_path) / f"{i}"
            paths.append((src_pth, tgt_pth))

with torch.no_grad():
    for p, p_tgt in paths:
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
        irl = torch.tensor(irl).cuda().unsqueeze(0).unsqueeze(0)

        #run local contrast normalization (LCN)
        # TODO: is this the way the x-positions are encoded?
        if regressor_conv:
            _, _, x, msk = model(irl)
        else:
            x, msk = model(irl)
        x = x[0, 0, :, :]
        x = x * x.shape[1]
        x_0 = torch.arange(0, x.shape[1]).unsqueeze(0).to(device)
        x -= x_0
        x *= -1.0

        result = x.cpu()[:, :].numpy()
        msk = np.clip(msk.cpu()[0, 0, :, :].numpy()*255, 0, 255).astype(np.uint8)

        #result = coresup_pred.cpu()[0, 0, :, :].numpy()

        p = str(p_tgt) + ".exr" # = path_out + f"/{int(ind):05d}.exr"
        cv2.imshow("result", result * (1.0 / 50.0))
        cv2.imwrite(p, result)


        p = str(p_tgt) + "_msk.png"# path_out + f"/{int(ind):05d}_msk.png"
        cv2.imshow("mask", msk)
        cv2.imwrite(p, result)
        cv2.waitKey(1)
        print(p)