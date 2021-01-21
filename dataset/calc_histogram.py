import cv2
import numpy as np

dataset_path = "/media/simon/SSD/unity_rendered_2/reduced_0_08"
start_ind = 0
end_ind = 2500
height = 896
hist_width = 2000
hist = np.zeros([896, hist_width]).astype(np.int)

for i in range(start_ind, end_ind):
    image_path = dataset_path + "/" + str(i) + "_gt_r.exr"
    gt = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    cv2.imshow("gt", gt[:, :, 2])
    for j in range(0, height):
        hist_tmp = np.histogram(gt[j, :, 2], hist_width, [0, 1.0])
        hist[j, :] = hist[j, :] + np.asarray(hist_tmp[0])

    hist_p = hist[:, 1:]
    hist_p = hist_p.astype(np.float)
    hist_p = hist_p * (1.0 / np.max(hist_p))
    #hist_p = cv2.resize(hist_p, (1000, hist_p.shape[0]), cv2.INTER_LANCZOS4)
    cv2.imshow("hist", hist_p)
    cv2.waitKey(1)

cv2.waitKey()

