import cv2
import numpy as np
import os
import multiprocessing
import itertools

folder_src = "/media/simon/sandisk2/dataset_collection/synthetic_train"
folder_dst = "/media/simon/sandisk2/dataset_collection/synthetic_train_reduced"
min_tgt_res = 1000000
export_jpg = False

def apply_recursively(input_root, output_root, current_sub, io_pairs):
    current_src = os.path.join(input_root, current_sub)
    current_tgt = os.path.join(output_root, current_sub)
    #print(os.listdir(current_tgt))
    if 14 < len(os.listdir(current_tgt)) < 100:
        return
    if os.path.isdir(current_src) and not os.path.exists(current_tgt):
        os.mkdir(current_tgt)
    for path in os.listdir(current_src):
        current = os.path.join(current_src, path)
        #print(current)
        if os.path.isdir(current):
            apply_recursively(input_root, output_root, os.path.join(current_sub, path), io_pairs)
        else:
            if current[-3:] == "png":
                current_dst = os.path.join(current_tgt, path)
                io_pairs.append((current, current_dst))
            #else:
            #    current_dst = os.path.join(current_tgt, path)
            #    os.system(f"cp {current} {current_dst}")

io_pairs = []

apply_recursively(folder_src, folder_dst, "", io_pairs)
print(len(io_pairs))
pool = multiprocessing.Pool(processes=12)

def process_pair(src, dst):
    print(f"{src} to {dst}")
    image = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            image = image[:, :, :3]

    while image.shape[0] > min_tgt_res * 2 and image.shape[1] > min_tgt_res * 2:
        image = cv2.pyrDown(image)
    if export_jpg:
        cv2.imwrite(dst[:-3] + "jpg", image)
    else:
        cv2.imwrite(dst, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

if False:
    pool.starmap(process_pair, io_pairs)
else:
    for src, dst in io_pairs:
        print(f"{src} to {dst}")
        image = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                image = image[:, :, :3]

        while image.shape[0] > min_tgt_res * 2 and image.shape[1] > min_tgt_res * 2:
            image = cv2.pyrDown(image)
        if export_jpg:
            cv2.imwrite(dst[:-3] + "jpg", image)
        else:
            cv2.imwrite(dst, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])


print(io_pairs)