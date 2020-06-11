import cv2
import os
import numpy as np

source_path = '/media/simon/ssd_data/data/dataset_reduced_0_08'
tgt_path = '/media/simon/ssd_data/data/dataset_filtered_slice_146'
if os.name == 'nt':
    source_path = "D:/dataset_filtered"
    tgt_path = "D:/dataset_filtered_strip_100_146"

image_count = 10000
start_line = 100
slice_height = 146

for i in range(0, image_count):
    image_r = cv2.imread(source_path + "/" + str(i) + "_r.exr", cv2.IMREAD_UNCHANGED)
    gt_r = cv2.imread(source_path + "/" + str(i) + "_gt_r.exr", cv2.IMREAD_UNCHANGED)

    cv2.imshow("image", image_r)
    cv2.imshow("gt", gt_r)
    cv2.waitKey(1)

    print(i)
    image_r = cv2.cvtColor(image_r[start_line: start_line + slice_height, :, :], cv2.COLOR_BGR2GRAY)
    gt_r = gt_r[start_line:start_line+slice_height, :, :]
    np.save(tgt_path + "/" + str(i) + "_r.npy", image_r)
    np.save(tgt_path + "/" + str(i) + "_gt_r.npy", gt_r)





    #todo: load image and take image from that piece