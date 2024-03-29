import numpy as np
import torch
from pathlib import Path
import h5py
import cv2
import time
import os

#todo: get this from the config file
data_path = str(Path("/media/simon/sandisk/datasets/DepthInSpace/"))
data_output_path = str(Path("/media/simon/sandisk/datasets/DepthInSpace_results/GigaDepth/"))
data_inds_captured = np.arange(4, 147, 8)
data_inds_rendered = np.arange(2 ** 9, 2 ** 10)
thresholds = [0.1, 0.5, 1, 2, 5]

algorithms = [  ("syn_default", "dis_def_lcn_c1920.pt", "rendered_default", data_inds_rendered),
                ("syn_kinect", "dis_kin_lcn_c1920_chk.pt", "rendered_kinect", data_inds_rendered),
                ("syn_real", "dis_real_lcn_c1920_chk.pt", "rendered_real", data_inds_rendered),
                ("real", "dis_real_lcn_j2_c1920_chk.pt", "captured", data_inds_captured)
                ]

for name, model_name, subpath, inds in algorithms:
    if not os.path.exists(f"{data_output_path}/{subpath}"):
        os.mkdir(f"{data_output_path}/{subpath}")
    print("-" * 80)
    print(name)

    print("-" * 80)
    print()
    model = torch.load(f"trained_models/{model_name}")
    model.eval()
    model.cuda()

    data = {}
    for th in thresholds:
        data[f"outliers_{th}"] = {
            "outliers": 0,
            "valid": 0
        }
    for ind in inds:
        if not os.path.exists(f"{data_output_path}/{subpath}/{ind:08d}"):
            os.mkdir(f"{data_output_path}/{subpath}/{ind:08d}")
        #todo: load data
        f = h5py.File(f'{data_path}/{subpath}/{ind:08d}/frames.hdf5', 'r')
        im = torch.tensor(np.array(f["im"])).cuda()
        disp = torch.tensor(np.array(f["disp"])).cuda()
        #im = np.array(f["im"])
        #disp = np.array(f["disp"])
        #todo: run_network
        with torch.no_grad():

            tsince = int(round(time.time() * 1000))
            result = model(im)
            torch.cuda.synchronize()
            ttime_elapsed = int(round(time.time() * 1000)) - tsince
            result *= result.shape[3]
            result -= torch.arange(0, result.shape[3]).reshape([1, 1, 1, -1]).cuda()
            result = -result
            msk = disp != 0
            valid_count = torch.count_nonzero(msk).item()
            delta = torch.abs(result - disp)

            for i in range(4):
                cv2.imwrite(f"{data_output_path}/{subpath}/{ind:08d}/{i}.exr",
                            result[i, 0, :, :].detach().cpu().numpy())

            if False:
                print(f"test time elapsed {ttime_elapsed } ms")
                cv2.imshow("im", im[0,0,:,:].detach().cpu().numpy())
                cv2.imshow("disp", disp[0,0,:,:].detach().cpu().numpy()/10)
                cv2.imshow("result", result[0,0,:,:].detach().cpu().numpy()/10)
                cv2.imshow("delta", delta[0,0,:,:].detach().cpu().numpy()/10)
                cv2.waitKey()

            for th in thresholds:
                outlier_count = torch.count_nonzero(torch.logical_and(delta > th, msk))
                data[f"outliers_{th}"]["outliers"] += outlier_count.item()
                data[f"outliers_{th}"]["valid"] += valid_count

    for th in thresholds:
        outlier_ratio = data[f"outliers_{th}"]["outliers"] / data[f"outliers_{th}"]["valid"]
        print(f"o({th}) = {outlier_ratio}")
