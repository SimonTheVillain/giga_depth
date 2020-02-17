import cv2
import numpy as np

source_dir = "/media/simon/TOSHIBA EXT/unity_rendered_2/unity_rendered/"
target_dir = "/media/simon/TOSHIBA EXT/unity_rendered_2/reduced/"
count = 11000


for i in range(2, count):
    gt_l = cv2.imread(source_dir + str(i) + "_gt_l.exr", cv2.IMREAD_UNCHANGED)
    gt_l = np.uint16(gt_l * 65535)
    cv2.imwrite(target_dir + str(i) + "_gt_l.png", gt_l)

    gt_r = cv2.imread(source_dir + str(i) + "_gt_r.exr", cv2.IMREAD_UNCHANGED)
    gt_r = np.uint16(gt_r * 65535)
    cv2.imwrite(target_dir + str(i) + "_gt_r.png", gt_r)

    r = cv2.imread(source_dir + str(i) + "_r.exr", cv2.IMREAD_UNCHANGED)
    r = np.uint16(r * 65535)
    cv2.imwrite(target_dir + str(i) + "_r.png", r)

    r = cv2.imread(source_dir + str(i) + "_r.exr", cv2.IMREAD_UNCHANGED)
    r_w = cv2.imread(source_dir + str(i) + "_r_w.exr", cv2.IMREAD_UNCHANGED)
    r_wo = cv2.imread(source_dir + str(i) + "_r_wo.exr", cv2.IMREAD_UNCHANGED)
    r_w = cv2.cvtColor(r_w, cv2.COLOR_RGB2GRAY)
    r_wo = cv2.cvtColor(r_wo, cv2.COLOR_RGB2GRAY)
    print(r_w.shape)
    print(r_wo.shape)
    cv2.imshow("r", r)
    mask = np.greater(r_w - r_wo, 0.1)
    mask = mask.astype(np.float)
    cv2.imshow("r_w - r_wo", mask)
    cv2.waitKey()


