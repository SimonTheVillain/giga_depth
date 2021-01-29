import cv2
import numpy as np
import os
path = os.path.expanduser("~/datasets/structure_core_unity")
path = os.path.expanduser("/media/simon/ssd_data/data/datasets/structure_core_unity")

nr_files = 21000

for i in range(0, nr_files):
    print(i)
    ir_l = cv2.imread(f"{path}/{i}_left.png")
    ir_r = cv2.imread(f"{path}/{i}_right.png")
    gt_l = cv2.imread(f"{path}/{i}_left_gt.exr")
    gt_r = cv2.imread(f"{path}/{i}_right_gt.exr")
    n_l = cv2.imread(f"{path}/{i}_left_n.png")
    n_r = cv2.imread(f"{path}/{i}_left_n.png")

    if ir_l is None:
        print("ir_l is bad")
    if ir_l is None:
        print("ir_r is bad")

    if gt_l is None:
        print("gt_l is bad")

    if gt_r is None:
        print("gt_r is bad")

    if n_l is None:
        print("n_l is bad")

    if n_r is None:
        print("n_r is bad")



