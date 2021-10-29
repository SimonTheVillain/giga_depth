import numpy as np
import cv2
import os
import re
from common.common import downsampleDepth
import pickle
import matplotlib.pyplot as plt
from multiprocessing import Pool


def process_results(algorithm):
    path_src = "/media/simon/ssd_datasets/datasets/structure_core_unity_test"
    path_results = "/media/simon/ssd_datasets/datasets/structure_core_unity_test_results"

    inds = os.listdir(path_src)
    inds = [re.search(r'\d+', s).group() for s in inds]
    inds = set(inds)
    inds = list(inds)
    inds.sort()
    #inds = inds[:10]

    src_res = (1401, 1001)
    src_cxy = (700, 500)
    tgt_res = (1216, 896)
    tgt_cxy = (604, 457)
    # the focal length is shared between src and target frame
    focal = 1.1154399414062500e+03
    baseline = 0.0634
    rr = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1], tgt_res[0], tgt_res[1])

    cutoff_dist = 20.0


    thresholds = np.arange(0.05, 20, 0.05)

    relative_th_base = 0.5
    relative_ths = np.arange(0.05, relative_th_base, 0.05)

    distances = np.arange(0.05, 10 - 0.05, 0.1)
    distances_ths = [0.1, 1, 5]

    data = {"inliers": {"ths": thresholds,
                        "data": [0] * thresholds.shape[0],
                        "pix_count": 0},
            "conditional_inliers": {"th": relative_th_base,
                                    "ths": relative_ths,
                                    "data": [0] * relative_ths.shape[0],
                                    "pix_count": 0}
            }
    for ind in inds:
        path_gt = path_src + f"/{ind}_left_d.exr"
        d_gt = cv2.imread(path_gt, cv2.IMREAD_UNCHANGED)
        d_gt = d_gt[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]]

        path_dst = path_results + "/" + algorithm + f"/{int(ind):05d}.exr"
        if not os.path.exists(path_dst):
            path_dst = path_results + "/" + algorithm + f"/{int(ind)}.exr"
        estimate = cv2.imread(path_dst, cv2.IMREAD_UNCHANGED)

        if d_gt.shape[0] // 2 == estimate.shape[0]:
            # downsample by a factor of 2
            d_gt = downsampleDepth(d_gt)
            disp_gt = focal * baseline * 0.5 / d_gt
            delta = np.abs(disp_gt - estimate) * 2.0 # in case

        else:
            disp_gt = focal * baseline / d_gt
            delta = np.abs(disp_gt - estimate)

        msk = d_gt < cutoff_dist
        msk_count = np.sum(msk)
        data["inliers"]["pix_count"] += msk_count
        for i, threshold in enumerate(thresholds):

            valid_count = np.sum(np.logical_and(delta < threshold, msk))
            data["inliers"]["data"][i] += valid_count

        # todo: simon! do you really need this? I THINK IT IS NOT NECESSARY
        msk = np.logical_and(d_gt < cutoff_dist, delta < relative_th_base)
        msk_count = np.sum(msk)
        data["conditional_inliers"]["pix_count"] += msk_count
        for i, threshold in enumerate(relative_ths):
            valid_count = np.sum(np.logical_and(delta < threshold, msk))
            data["conditional_inliers"]["data"][i] += valid_count

        for th in distances_ths:
            if f"inliers_{th}" not in data:
                data[f"inliers_{th}"] = {"distances": (distances[:-1] + distances[1:]) * 0.5,
                                         "data": [0] * (distances.shape[0] - 1),
                                         "pix_count": [0] * (distances.shape[0] - 1)}
            for i in range(len(distances) - 1):
                dist_low = distances[i]
                dist_high = distances[i + 1]
                msk = np.logical_and(d_gt > dist_low, d_gt < dist_high)
                msk_count = np.sum(msk)
                valid_count = np.sum(np.logical_and(delta < threshold, msk))
                data[f"inliers_{th}"]["data"][i] += valid_count
                data[f"inliers_{th}"]["pix_count"][i] += msk_count

        #cv2.imshow("gt", disp_gt * 0.02)
        #cv2.imshow("estimate", estimate * 0.02)
        #cv2.waitKey()
    f = open(path_results + f"/{algorithm}.pkl", "wb")
    pickle.dump(data, f)
    f.close()

def create_data():
    algorithms = ["GigaDepth", "ActiveStereoNet", "connecting_the_dots", "HyperDepth"]#, "GigaDepthLCN"]
    algorithms = ["HyperDepth"]
    threading = False

    if threading:
        with Pool(5) as p:
            p.map(process_results, algorithms)
    else:
        for algorithm in algorithms:
            process_results(algorithm)


def create_plot():
    path_results = "/media/simon/ssd_datasets/datasets/structure_core_unity_test_results"
    algorithms = ["GigaDepth", "ActiveStereoNet", "connecting_the_dots", "HyperDepth"] # "HyperDepth", "GigaDepthLCN"]

    legend_names = {"GigaDepth": "GigaDepth",
                    "ActiveStereoNet": "ActiveStereoNet",
                    "connecting_the_dots": "Connecting The Dots",
                    "HyperDepth": "HyperDepth"}

    plt.figure(1)
    for algorithm in algorithms:
        with open(path_results + f"/{algorithm}.pkl", "rb") as f:
            data = pickle.load(f)
        plt.plot(data["inliers"]["ths"], data["inliers"]["data"] / data["inliers"]["pix_count"])
    plt.legend(algorithms)

    th = 5
    plt.figure(2)
    for algorithm in algorithms:
        with open(path_results + f"/{algorithm}.pkl", "rb") as f:
            data = pickle.load(f)
        plt.plot(data[f"inliers_{th}"]["distances"],
                 np.array(data[f"inliers_{th}"]["data"]) / np.array(data[f"inliers_{th}"]["pix_count"]))
    plt.legend(algorithms)
    plt.show()

create_data()
create_plot()