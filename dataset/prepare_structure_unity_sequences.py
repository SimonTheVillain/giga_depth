import os

import cv2
import numpy as np
from pathlib import Path
from distutils.dir_util import copy_tree

datasets = ["/media/simon/WD/datasets_raw/structure_core_unity_1",
            "/media/simon/WD/datasets_raw/structure_core_unity_2",
            "/media/simon/WD/datasets_raw/structure_core_unity_3",
            "/media/simon/LaCie/datasets_raw/structure_core_unity_4",
            "/media/simon/LaCie/datasets_raw/structure_core_unity_5",
            "/media/simon/LaCie/datasets_raw/structure_core_unity_6",
            "/media/simon/WD/datasets_raw/structure_core_unity_7",
            "/media/simon/ext_ssd/datasets_raw/structure_core_unity_8",
            "/media/simon/ext_ssd/datasets_raw/structure_core_unity_9",
            "/media/simon/ext_ssd/datasets_raw/structure_core_unity_10"]
count = 0

output_path = "/home/simon/datasets/structure_core_unity_sequences"

for dataset in datasets:
    dirs = os.listdir(dataset)
    for dir in dirs:
        sequence = dataset + "/" + dir
        if os.path.isdir(sequence):
            print(sequence)
            destination = output_path + "/" + f"{count:05d}"
            os.mkdir(destination)

            for i in range(4):
                os.system(f"cp -r {sequence}/{i}.json {destination}/{i}.json")
                for side in ["left", "right"]:
                    os.system(f"cp -r {sequence}/{i}_{side}.png {destination}/{i}_{side}.png")
                    #ir = cv2.imread(f"{sequence}/{i}_{side}.png", cv2.IMREAD_UNCHANGED)
                    #cv2.imwrite(f"{destination}/{i}_{side}.png", ir, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    os.system(f"cp -r {sequence}/{i}_{side}_gt.exr {destination}/{i}_{side}_gt.exr")
                    # create a mask
                    gt = cv2.imread(f"{sequence}/{i}_{side}_gt.exr", cv2.IMREAD_UNCHANGED)
                    ir_no = cv2.imread(f"{sequence}/{i}_{side}_noproj.exr", cv2.IMREAD_UNCHANGED)
                    ir_msk = cv2.imread(f"{sequence}/{i}_{side}_msk.exr", cv2.IMREAD_UNCHANGED)
                    msk = np.zeros((ir_no.shape[0], ir_no.shape[1]), dtype=np.ubyte)
                    th = 0.0001
                    delta = np.abs(ir_no - ir_msk)
                    msk[(delta[:, :, 0] + delta[:, :, 1] + delta[:, :, 2]) > th] = 255
                    msk[gt[:, :, 1] > 0] = 0

                    cv2.imwrite(f"{destination}/{i}_{side}_msk.png", msk)

            count = count + 1
