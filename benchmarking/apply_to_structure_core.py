import matplotlib

matplotlib.use('Agg')
import torch
import cv2
import numpy as np

# Work in the parent directory
import os
import model
from model.composite_model import CompositeModel
import torch
import os
import cv2
import numpy as np
import re
from pathlib import Path
import yaml
import argparse


def apply_to_dataset(combined_model_pth="", output_folder_name="",
                     backbone_model_pth="", regressor_model_pth="",
                     mode="rendered"):
    with open("configs/paths-local.yaml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    for key in config.keys():
        config[key] = Path(config[key])

    device = "cuda:0"
    if combined_model_pth == "":
        backbone = torch.load(backbone_model_pth, map_location=device)
        backbone.eval()
        regressor = torch.load(regressor_model_pth, map_location=device)
        regressor.eval()
        model = CompositeModel(backbone, regressor)
    else:
        model = torch.load(combined_model_pth, map_location=device)
        model.eval()

    half_res = False

    if mode == "rendered":
        src_res = (1401, 1001)
        src_cxy = (700, 500)
        tgt_res = (1216, 896)
        tgt_cxy = (604, 457)
        # the focal length is shared between src and target frame
        focal = 1.1154399414062500e+03

        rr = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1], tgt_res[0], tgt_res[1])

        path = config["dataset_unity_test_path"]
        path_out_root = config["dataset_unity_test_results_path"]
        path_out = f"{path_out_root}/{output_folder_name}"

        if not os.path.exists(path_out):
            os.mkdir(path_out)

        inds = os.listdir(path)
        inds = [re.search(r'\d+', s).group() for s in inds]
        inds = set(inds)
        inds = list(inds)
        inds.sort()
        paths = []
        for ind in inds:
            paths.append((str(path) + f"/{ind}_left.jpg", path_out + f"/{int(ind):05d}"))

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
        path_out = "/media/simon/ssd_datasets/datasets/shapenet_rendered_compressed_test_results/GigaDepth66LCN"

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

    if mode == "captured_photoneo" or mode == "captured_plane":
        tgt_res = (1216, 896)
        tgt_cxy = (604, 457)
        # the focal length is shared between src and target frame
        focal = 1.1154399414062500e+03

        rr = (tgt_res[0], 0, tgt_res[0], tgt_res[1])

        if mode == "captured_plane":
            path = config["dataset_structure_plane_test_path"]
            path_out_root = config["dataset_structure_plane_test_results_path"]
            path_out = f"{path_out_root}/{output_folder_name}"
        else:  # experiment is photoneo
            path = config["dataset_structure_photoneo_test_path"]
            path_out_root = config["dataset_structure_photoneo_test_results_path"]
            path_out = f"{path_out_root}/{output_folder_name}"

        if not os.path.exists(path_out):
            os.mkdir(path_out)

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

            # run local contrast normalization (LCN)
            # TODO: is this the way the x-positions are encoded?

            x = model(irl)
            x = x[0, 0, :, :]
            x = x * x.shape[1]
            x_0 = torch.arange(0, x.shape[1]).unsqueeze(0).to(device)
            x -= x_0
            x *= -1.0

            result = x.cpu()[:, :].numpy()
            # msk = np.clip(msk.cpu()[0, 0, :, :].numpy()*255, 0, 255).astype(np.uint8)

            # result = coresup_pred.cpu()[0, 0, :, :].numpy()

            p = str(p_tgt) + ".exr"  # = path_out + f"/{int(ind):05d}.exr"
            cv2.imshow("result", result * (1.0 / 50.0))
            cv2.imwrite(p, result)

            p = str(p_tgt) + "_msk.png"  # path_out + f"/{int(ind):05d}_msk.png"
            # cv2.imshow("mask", msk)
            # cv2.imwrite(p, result)
            cv2.waitKey(1)
            #print(p)


experiments = [#("GigaDepth76c1920LCN", "full_76_lcn_j2_c1920.pt"),
               #("GigaDepth76c1920", "full_76_j2_c1920.pt"),
               #("GigaDepth76c1280LCN", "full_76_lcn_j2_c1280.pt"),
               #("GigaDepth78Uc1920", "full_78_unet_j2_c1920.pt")
               #("GigaDepth76j4c1280LCN", "full_76_lcn_j4_c1280.pt"),
                #("GigaDepth73LineLCN", "full_73_lcn_line.pt"),
                #("GigaDepth72UNetLCN", "full_72_lcn_unet.pt"),
                ("GigaDepth76J4C1280_8GB_DELETE", "full_76_j4_c1280_8GB.pt"),
                ("GigaDepth76J4C1280_8GBPNG_DELETE", "full_76_j4_c1280_8GB_png.pt"),
               ]
for output_folder_name, model_name in experiments:
    combined_model_pth = f"trained_models/{model_name}"
    for mode in ["captured_plane", "captured_photoneo", "rendered"]:  # "rendered",
        apply_to_dataset(combined_model_pth=combined_model_pth,
                         mode=mode,
                         output_folder_name=output_folder_name)
