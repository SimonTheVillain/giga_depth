import numpy as np
import cv2
import os
from shutil import copyfile

path_src = "/media/simon/DATA/datasets/structure_core_unity"
path_dst = os.path.expanduser("~/datasets/structure_core_unity")

counter = 1
counter_result = 0
depth_threshold = 10

path_img = path_src + f"/{counter}_left_gt.exr"
while os.path.exists(path_img):
    gt = cv2.imread(path_img, cv2.IMREAD_UNCHANGED)

    valid_pixel = np.logical_and(gt[:, :, 0] < 10, gt[:, :, 1] == 0)
    ratio_valid = float(np.count_nonzero(valid_pixel)) / float(valid_pixel.shape[0] * valid_pixel.shape[1])
    print(f"file nr {counter}")
    #todo. insert query for if the
    if ratio_valid > 0.6:
        copyfile(path_src + f"/{counter}_left.png", path_dst + f"/{counter_result}_left.png")
        copyfile(path_src + f"/{counter}_right.png", path_dst + f"/{counter_result}_right.png")
        copyfile(path_src + f"/{counter}_left_n.png", path_dst + f"/{counter_result}_left_n.png")
        copyfile(path_src + f"/{counter}_right_n.png", path_dst + f"/{counter_result}_right_n.png")
        copyfile(path_src + f"/{counter}_left_gt.exr", path_dst + f"/{counter_result}_left_gt.exr")
        copyfile(path_src + f"/{counter}_right_gt.exr", path_dst + f"/{counter_result}_right_gt.exr")
        counter_result += 1

    counter += 1
    path_img = path_src + f"/{counter}_left_gt.exr"


