import os
os.chdir("../")
import torch
from model.composite_model import CompositeModel
import cv2
import numpy as np
import timeit
import re
from pathlib import Path

#Work in the parent directory

device = "cuda:0"
def apply_recursively(model, input_root, output_root, measure_time = False ):
    src_res = (1401, 1001)
    src_cxy = (700, 500)
    tgt_res = (1216, 896)
    tgt_cxy = (604, 457)
    # the focal length is shared between src and target frame
    focal = 1.1154399414062500e+03

    rr = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1], tgt_res[0], tgt_res[1])
    half_res = False
    regressor_conv = False
    inds = os.listdir(input_root)
    inds  = [re.search(r'\d+', s).group() for s in inds]
    inds = set(inds)
    inds = list(inds)
    inds.sort()
    paths = []
    for ind in inds:
        paths.append((input_root + f"/{ind}_left.jpg", output_root + f"/{int(ind):05d}"))

    if measure_time:
        paths = paths[:100]
        time = 0
        count = 0
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

            if measure_time:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                x = model.backbone(irl)
                start.record()
                x = model.regressor(x)
                end.record()
                torch.cuda.synchronize()
                time += start.elapsed_time(end)
                count += 1
                continue

            # run local contrast normalization (LCN)
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
            msk = np.clip(msk.cpu()[0, 0, :, :].numpy() * 255, 0, 255).astype(np.uint8)

            # result = coresup_pred.cpu()[0, 0, :, :].numpy()

            p = str(p_tgt) + ".exr"  # = path_out + f"/{int(ind):05d}.exr"
            cv2.imshow("result", result * (1.0 / 50.0))
            cv2.imwrite(p, result)

            p = str(p_tgt) + "_msk.png"  # path_out + f"/{int(ind):05d}_msk.png"
            cv2.imshow("mask", msk)
            cv2.imwrite(p, result)
            cv2.waitKey(1)
            print(p)
    print(f"time: {time / count}")



out_folder = "/home/simon/datasets/structure_core_unity_test_results/class_mlp_experiments"
out_folder = "/media/simon/T7/datasets/structure_core_unity_test_results/class_mlp_experiments"
experiments = [#("c288", "class_288_r2"),
               #("c384", "class_384_r2"),
               ("c640_r1", "class_640_r1"),
               ("c640", "class_640_r2"),
               #("c640_r3", "class_640_r3"),
               ("c1280", "class_1280_r2"),
               ("c1920", "class_1920_r2"),
               ("c2688", "class_2688_r2")]
experiments = [("c640_r3_v3", "class_640_r3_v3")]
experiments = [("c288_v2", "class_288_r2_v2"),
               ("c384_v2", "class_384_r2_v2"),
               ("c640_r1_v2", "class_640_r1_v2"),
               ("c640_r2_v2", "class_640_r2_v2"),
               ("c640_r3_v2", "class_640_r3_v2"),
               ("c1280_v2", "class_1280_r2_v2"),
               ("c1536_v2", "class_1536_r2_v2"),
               ("c1920_v2", "class_1920_r2_v2"),
               ("c2688_v2", "class_2688_r2_v2")]

measure_time = True
for net, folder_out in experiments:
    print(folder_out)
    regressor_model_pth = f"trained_models/full_68_lcn_j2_{net}_regressor_chk.pt"
    backbone_model_pth = f"trained_models/full_68_lcn_j2_{net}_backbone_chk.pt"

    path_src = "/home/simon/datasets/structure_core_unity_test"
    path_results = f"{out_folder}/{folder_out}"  # GigaDepthNoLCN"
    backbone = torch.load(backbone_model_pth, map_location=device)
    backbone.eval()

    regressor = torch.load(regressor_model_pth, map_location=device)
    regressor.eval()

    model = CompositeModel(backbone, regressor)

    model.to(device)
    apply_recursively(model, path_src, path_results, measure_time=measure_time)
