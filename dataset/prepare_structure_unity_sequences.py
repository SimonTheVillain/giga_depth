import os

import cv2
import numpy as np
from pathlib import Path
from distutils.dir_util import copy_tree

datasets = ["/media/simon/LaCie/datasets_raw/structure_core_unity_1",
            "/media/simon/LaCie/datasets_raw/structure_core_unity_2",
            "/media/simon/LaCie/datasets_raw/structure_core_unity_3",
            "/media/simon/LaCie/datasets_raw/structure_core_unity_4",
            "/media/simon/LaCie/datasets_raw/structure_core_unity_5",
            "/media/simon/LaCie/datasets_raw/structure_core_unity_6",
            "/media/simon/LaCie/datasets_raw/structure_core_unity_7",
            "/media/simon/LaCie/datasets_raw/structure_core_unity_8",
            "/media/simon/LaCie/datasets_raw/structure_core_unity_9",
            "/media/simon/LaCie/datasets_raw/structure_core_unity_10"]
count = 0

output_path = "/home/simon/datasets/structure_core_unity_sequences_2"

for dataset in datasets:
    dirs = os.listdir(dataset)
    for dir in dirs:
        sequence = dataset + "/" + dir
        if os.path.isdir(sequence):
            print(sequence)
            destination = output_path + "/" + f"{count:05d}"
            if not os.path.exists(destination):
                os.mkdir(destination)

            for i in range(4):
                os.system(f"cp -r {sequence}/{i}.json {destination}/{i}.json")
                for side in ["left", "right"]:
                    # create a mask
                    gt = cv2.imread(f"{sequence}/{i}_{side}_gt.exr", cv2.IMREAD_UNCHANGED)
                    ir_no = cv2.imread(f"{sequence}/{i}_{side}_noproj.exr", cv2.IMREAD_UNCHANGED)
                    ir_msk = cv2.imread(f"{sequence}/{i}_{side}_msk.exr", cv2.IMREAD_UNCHANGED)
                    msk = np.zeros((ir_no.shape[0], ir_no.shape[1]), dtype=np.ubyte)
                    th = 0.0001
                    delta = np.abs(ir_no - ir_msk)
                    msk[(delta[:, :, 0] + delta[:, :, 1] + delta[:, :, 2]) > th] = 255
                    msk[gt[:, :, 1] > 0] = 0

                    os.system(f"cp -r {sequence}/{i}_{side}.png {destination}/{i}_{side}.png")
                    os.system(f"cp -r {sequence}/{i}_{side}_gt.exr {destination}/{i}_{side}_gt.exr")
                    cv2.imwrite(f"{destination}/{i}_{side}_msk.png", msk)

                    ir_msk = ir_msk[:, :, :3]
                    filter = ir_msk
                    filter[ir_msk==1.0] = 0.0
                    filter[ir_msk==0.5] = 0.0
                    m = np.mean(filter) * 4.0
                    pth_out = "/home/simon/Pictures/images_paper/supplemental"
                    cv2.imwrite(f"{pth_out}/ir_no.png", ir_no/m * 255)
                    cv2.imwrite(f"{pth_out}/ir_msk.png", ir_msk/m * 255)
                    cv2.imwrite(f"{pth_out}/msk.png", msk)
                    cv2.imshow("ir_no", ir_no/m)
                    cv2.imshow("ir_msk", ir_msk/m)
                    cv2.imshow("msk", msk)
                    cv2.waitKey()


            count = count + 1
