import numpy as np
import cv2
import os
from shutil import copyfile


in_paths = ["/media/simon/datasets/structure_core_unity_2"]
out_path = "/media/simon/ssd_data/data/datasets/structure_core_unity_2"
count = 0
for path in in_paths:

    files = os.listdir(path)
    names = []
    for file in files:
        if os.path.isfile(f"{path}/{file}"):
            names.append(file.split("_")[0])
    #make the names unitque
    names = list(set(names))
    names.sort()

    for name in names:
        to_check = ["_left.png", "_right.png",
                    "_left_msk.exr", "_right_msk.exr",
                    "_left_noproj.exr", "_right_noproj.exr",
                    "_left_gt.exr", "_right_gt.exr"]
        for chk in to_check:
            if not os.path.isfile(f"{path}/{name}chk"):
                skip = True
                continue
        skip = False

        if skip:
            continue
        for side in ["left", "right"]:
            print(name)
            ir = cv2.imread(f"{path}/{name}_{side}.png")
            cv2.imshow("ir", ir)
            #
            #cv2.imwrite(f"{out_path}/{count}_{side}.png", ir, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            cv2.imwrite(f"{out_path}/{count}_{side}.jpg", ir, [cv2.IMWRITE_JPEG_QUALITY, 95])

            ir_no = cv2.imread(f"{path}/{name}_{side}_noproj.exr", cv2.IMREAD_UNCHANGED)
            ir_msk = cv2.imread(f"{path}/{name}_{side}_msk.exr", cv2.IMREAD_UNCHANGED)
            gt = cv2.imread(f"{path}/{name}_{side}_gt.exr", cv2.IMREAD_UNCHANGED)
            #cv2.imshow("ir_no", ir_no)
            #cv2.imshow("ir_msk", ir_msk)
            cv2.imshow("depth", gt[:, :, 0] * 0.1)
            cv2.imshow("diff", np.abs(ir_no - ir_msk) * 100)
            #cv2.imshow("gt", gt)
            #cv2.imshow("gt_1", gt[:, :, 0])#depth
            #cv2.imshow("gt_2", gt[:, :, 1])#mask
            #cv2.imshow("gt_3", gt[:, :, 2])#disparity

            cv2.imwrite(f"{out_path}/{count}_{side}_d.exr", gt[:, :, 0],
                        [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])# write out depth as float16


            #todo: maybe scale this
            msk = np.zeros((ir_no.shape[0], ir_no.shape[1]), dtype=np.ubyte)
            th = 0.0001
            delta = np.abs(ir_no - ir_msk)
            msk[(delta[:, :, 0] + delta[:, :, 1] + delta[:, :, 2]) > th] = 255
            msk[gt[:, :, 1] > 0] = 0

            # pack bits of mask!
            # msk_pad = np.pad(msk, ((0, 0), (0, 7)), 'edge')
            # msk_bits = np.packbits(msk_pad)
            # msk_bits = np.reshape(msk_bits, (msk_pad.shape[0], int(msk_pad.shape[1] / 8)))
            # cv2.imwrite(f"{out_path}/{count}_{side}_msk_bits.png", msk_bits)

            cv2.imshow("msk", msk)

            #most optimal storage of the png file
            cv2.imwrite(f"{out_path}/{count}_{side}_msk.png", msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            cv2.waitKey(1)

        count += 1