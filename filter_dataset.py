import cv2
import numpy as np
import shutil

source_dir = "/media/simon/SSD/unity_rendered_2/dataset_scaled_x/"
target_dir = "/media/simon/ssd_data/data/reduced_0_08_2/"
count = 25000

ind = 0
# just a piece of code used to fix misnamed files
#for i in range(0, 28922):
#    print(i)
#    shutil.move(source_dir + str(i) + "_p_wo .png", source_dir + str(i) + "_p_wo.png")
with open(source_dir + 'valid_percentile.txt', 'a') as valid_ratio:
    for i in range(2, count):
        if i%10 == 0:
            print(i)
        #gt_l = cv2.imread(source_dir + str(i) + "_gt_l.exr", cv2.IMREAD_UNCHANGED)
        #gt_l = np.uint16(gt_l * 65535)
        #cv2.imwrite(target_dir + str(i) + "_gt_l.png", gt_l)

        r_w = cv2.imread(source_dir + str(i) + "_r_w.exr", cv2.IMREAD_UNCHANGED)
        r_wo = cv2.imread(source_dir + str(i) + "_r_wo.exr", cv2.IMREAD_UNCHANGED)
        r_w = cv2.cvtColor(r_w, cv2.COLOR_RGB2GRAY)
        r_wo = cv2.cvtColor(r_wo, cv2.COLOR_RGB2GRAY)
        mask = np.greater(r_w - r_wo, 0.1)
        mask = mask.astype(np.float)
        mean = np.mean(mask)
        valid_ratio.write("{} {}".format(i, mean))
        if mean > 0.8:
            for c in ['l', 'r']:
                shutil.copy(source_dir + str(i) + "_" + c + "_w.exr", target_dir + str(ind) + "_" + c + "_w.exr")
                shutil.copy(source_dir + str(i) + "_" + c + "_wo.exr", target_dir + str(ind) + "_" + c + "_wo.exr")
                shutil.copy(source_dir + str(i) + "_" + c + ".exr", target_dir + str(ind) + "_" + c + ".exr")
                shutil.copy(source_dir + str(i) + "_gt_" + c + ".exr", target_dir + str(ind) + "_gt_" + c + ".exr")

            shutil.copy(source_dir + str(i) + "_p.png", target_dir + str(ind) + "_p.png")
            shutil.copy(source_dir + str(i) + "_p_w.png", target_dir + str(ind) + "_p_w.png")
            shutil.copy(source_dir + str(i) + "_p_wo.png", target_dir + str(ind) + "_p_wo.png")

            p = cv2.imread(source_dir + str(i) + "_p.png", cv2.IMREAD_UNCHANGED)
            cv2.imshow("p", p)
            cv2.waitKey(1)
            ind = ind+1

        #todo copy over all images

        if False:
            gt_r = cv2.imread(source_dir + str(i) + "_gt_r.exr", cv2.IMREAD_UNCHANGED)
            gt_r = np.uint16(gt_r * 65535)
            cv2.imwrite(target_dir + str(i) + "_gt_r.exr", gt_r)

            r = cv2.imread(source_dir + str(i) + "_r.exr", cv2.IMREAD_UNCHANGED)
            r = np.uint16(r * 65535)
            cv2.imwrite(target_dir + str(i) + "_r.exr", r)

            r = cv2.imread(source_dir + str(i) + "_r.exr", cv2.IMREAD_UNCHANGED)
            r_w = cv2.cvtColor(r_w, cv2.COLOR_RGB2GRAY)
            r_wo = cv2.cvtColor(r_wo, cv2.COLOR_RGB2GRAY)
            print(r_w.shape)
            print(r_wo.shape)
            cv2.imshow("r", r)
            mask = np.greater(r_w - r_wo, 0.1)
            mask = mask.astype(np.float)
            cv2.imshow("r_w - r_wo", mask)
            cv2.waitKey()


