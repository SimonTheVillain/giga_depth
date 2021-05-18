import scipy.io
import os
import numpy as np
import cv2
from pathlib import Path

baseline = 0.075# is it?
focal = 578
# the depth is in millimeter: if we want it in meter.... divide by 1000

path_mrf = Path("/media/simon/LaCie/datasets/mrf")

if not os.path.exists(path_mrf / "raw_py"):
    os.mkdir(path_mrf / "raw_py")

for folder in ["angel", "arch", "fox", "gargoyle", "lion"]:
    if not os.path.exists(path_mrf / "raw_py" / folder):
        os.mkdir(path_mrf / "raw_py" / folder)
    for i in range(1, 13):
        depth = scipy.io.loadmat(path_mrf / "raw" / folder / f"depth{i:02d}.mat")
        depth = depth["D"].astype(np.float32) / 1000.0
        depth_2 = depth
        depth_2[depth == 0] = 0.01
        disp = focal * baseline / depth_2
        disp[depth == 0] = 0
        p = path_mrf / "raw_py" / folder / f"disp{i:02d}.exr"
        cv2.imwrite(str(path_mrf / "raw_py" / folder / f"disp{i:02d}.exr"), disp)

        rgb = scipy.io.loadmat(path_mrf / "raw" / folder / f"rgb{i:02d}.mat")
        rgb = rgb["I"]
        cv2.imwrite(str(path_mrf / "raw_py" / folder / f"rgb{i:02d}.png"), rgb)

        ir = scipy.io.loadmat(path_mrf / "raw" / folder / f"ir{i:02d}.mat")
        ir = ir["J"] * 64
        cv2.imwrite(str(path_mrf / "raw_py" / folder / f"rgb{i:02d}.png"), ir)



i = 0
if not os.path.exists(path_mrf / "train_py"):
    os.mkdir(path_mrf / "train_py")

path_out = path_mrf / "train_py"
for folder in ["extra", "1", "2", "3", "4", "5"]:
    path = path_mrf / "additional_2019/more_data" / folder
    l = os.listdir(path)
    for file in l:
        if file[-3:] == "mat" and file[:5] == "depth":
            key = file[5:-4]
            print(key)
            depth = scipy.io.loadmat(path / f"depth{key}.mat")
            depth = depth["D"].astype(np.float32) / 1000.0
            disp = focal * baseline / depth
            cv2.imwrite(str(path_out / f"disp{i:03d}.exr"), disp)

            rgb = scipy.io.loadmat(path / f"rgb{key}.mat")
            rgb = rgb["I"]
            cv2.imwrite(str(path_out / f"rgb{i:03d}.png"), rgb)

            ir = scipy.io.loadmat(path / f"ir{key}.mat")
            ir = ir["J"] * 64
            cv2.imwrite(str(path_out / f"rgb{i:03d}.png"), ir)

            i = i+1
