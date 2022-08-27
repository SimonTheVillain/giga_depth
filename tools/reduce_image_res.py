import cv2
import numpy as np
import os

folder_src = "/home/simon/Pictures/images_paper2/supplemental/results_structure"
folder_dst = "/home/simon/Pictures/images_paper2/supplemental/results_structure_downsampled_jpg"
folder_src = "/home/simon/Pictures/images_paper2/supplemental/results_unity"
folder_dst = "/home/simon/Pictures/images_paper2/supplemental/results_unity_downsampled_jpg"
min_tgt_res = 300
export_jpg = True

def apply_recursively(input_root, output_root, current_sub, io_pairs):
    current_src = os.path.join(input_root, current_sub)
    current_tgt = os.path.join(output_root, current_sub)
    if os.path.isdir(current_src) and not os.path.exists(current_tgt):
        os.mkdir(current_tgt)
    for path in os.listdir(current_src):
        current = os.path.join(current_src, path)
        if os.path.isdir(current):
            apply_recursively(input_root, output_root, os.path.join(current_sub, path), io_pairs)
        else:
            if current[-3:] == "png":
                current_dst = os.path.join(current_tgt, path)
                io_pairs.append((current, current_dst))

io_pairs = []

apply_recursively(folder_src, folder_dst, "", io_pairs)

for src, dst in io_pairs:
    image = cv2.imread(src, cv2.IMREAD_UNCHANGED)

    while image.shape[0] > min_tgt_res * 2 and image.shape[1] > min_tgt_res * 2:
        image = cv2.pyrDown(image)
    if export_jpg:
        cv2.imwrite(dst[:-3] + "jpg", image)
    else:
        cv2.imwrite(dst, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])


print(io_pairs)