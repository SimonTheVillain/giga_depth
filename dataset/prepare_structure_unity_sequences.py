import os

import cv2
import numpy as np
from pathlib import Path
from distutils.dir_util import copy_tree
import multiprocessing

datasets = ["/media/simon/WD/datasets_raw/structure_core_unity_1",
            "/media/simon/WD/datasets_raw/structure_core_unity_2",
            "/media/simon/WD/datasets_raw/structure_core_unity_3",
            "/media/simon/WD/datasets_raw/structure_core_unity_4",
            "/media/simon/WD/datasets_raw/structure_core_unity_5",
            "/media/simon/WD/datasets_raw/structure_core_unity_6",
            "/media/simon/WD/datasets_raw/structure_core_unity_7",
            "/media/simon/WD/datasets_raw/structure_core_unity_8",
            "/media/simon/WD/datasets_raw/structure_core_unity_9",
            "/media/simon/WD/datasets_raw/structure_core_unity_10"]
count = 0

output_path = "/media/simon/WD/datasets/structure_core_unity_sequences_2"
output_path = "/media/simon/sandisk/datasets/structure_core_unity_sequences_3"

threaded = True


def process(dir, count):
    sequence = dataset + "/" + dir
    if os.path.isdir(sequence):
        print(sequence)
        destination = output_path + "/" + f"{count:05d}"
        if not os.path.exists(destination):
            os.mkdir(destination)

        amb_scale = 1

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

                #TODO: one could combine the ambient images of all 4 (8) and do the exposure control in one batch
                # this would get same exposure for all images and thus better results!
                amb = ir_no[:, :, :-1]
                if amb_scale == 1.0:
                    amb_scale = amb_scale / np.median(amb)
                    while np.count_nonzero(amb*amb_scale > 1.0) > amb.flatten().shape[0] * 0.05:
                        # print(f"{np.count_nonzero(amb > 1.0) } of {amb.flatten().shape[0] * 0.05}")
                        amb_scale *= 0.8

                amb *= amb_scale
                #cv2.imshow("amb", amb)
                #cv2.waitKey()
                os.system(f"cp -r {sequence}/{i}_{side}.png {destination}/{i}_{side}.png")
                os.system(f"cp -r {sequence}/{i}_{side}_gt.exr {destination}/{i}_{side}_gt.exr")
                cv2.imwrite(f"{destination}/{i}_{side}_msk.png", msk)
                cv2.imwrite(f"{destination}/{i}_{side}_amb.png", amb*255)

                if False:
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


for dataset in datasets:
    dirs = os.listdir(dataset)

    if threaded:
        pool = multiprocessing.Pool(12)
        pool.starmap(process, zip(dirs, range(count, count+len(dirs))))
        count += len(dirs)
    else:
        for directory in dirs:
            process(directory, count)
            count += 1


