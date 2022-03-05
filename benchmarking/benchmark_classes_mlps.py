import numpy as np
import cv2
import os
import re
from common.common import downsampleDepth
import pickle
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from multiprocessing import Pool
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


def process_results(algorithm):
    path_src = "/home/simon/datasets/structure_core_unity_test"
    path_results = "/home/simon/datasets/structure_core_unity_test_results/class_mlp_experiments"
    path_results = "/media/simon/T7/datasets/structure_core_unity_test_results/class_mlp_experiments"

    inds = os.listdir(path_src)
    inds = [re.search(r'\d+', s).group() for s in inds]
    inds = set(inds)
    inds = list(inds)
    inds.sort()
    inds = inds[:100]

    src_res = (1401, 1001)
    src_cxy = (700, 500)
    tgt_res = (1216, 896)
    tgt_cxy = (604, 457)
    # the focal length is shared between src and target frame
    focal = 1.1154399414062500e+03
    baseline = 0.0634
    rr = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1], tgt_res[0], tgt_res[1])

    cutoff_dist = 20.0

    thresholds = np.arange(0.05, 20, 0.01)

    relative_th_base = 0.5
    relative_ths = np.arange(0.05, relative_th_base, 0.05)

    distances = np.arange(0.05, 10 - 0.05, 0.1)
    distances_ths = [0.1, 1, 5]

    rmse_ths = np.array([1])

    data = {"inliers": {"ths": thresholds,
                        "data": [0] * thresholds.shape[0],
                        "pix_count": 0},
            "conditional_inliers": {"th": relative_th_base,
                                    "ths": relative_ths,
                                    "data": [0] * relative_ths.shape[0],
                                    "pix_count": 0},
            "rmse" : {"ths": rmse_ths,
                      "square_error": [0] * rmse_ths.shape[0],
                      "inliers": [0] * rmse_ths.shape[0]

                      }
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

        for i, threshold in enumerate(rmse_ths):
            valid_count = np.sum(np.logical_and(delta < threshold, msk))
            data["rmse"]["inliers"][i] += valid_count
            square_error = np.multiply(delta, delta) * np.logical_and(delta < threshold, msk)
            data["rmse"]["square_error"][0] += np.sum(square_error)

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
    algorithms = ["class_288_r2",
                  "class_384_r2",
                  "class_1280_r2",
                  "class_1920_r2",
                  "class_2688_r2",
                  "class_640_r1",
                  "class_640_r2",
                  "class_640_r3"]

    algorithms = [
                  "class_640_r3_v3"]

    algorithms = ["class_288_r2_v2",
                  "class_384_r2_v2",
                  "class_1280_r2_v2",
                  "class_1536_r2_v2",
                  "class_1920_r2_v2",
                  "class_2688_r2_v2",
                  "class_640_r1_v2",
                  "class_640_r2_v2",
                  "class_640_r3_v2"]

    threading = False

    if threading:
        with Pool(5) as p:
            p.map(process_results, algorithms)
    else:
        for algorithm in algorithms:
            process_results(algorithm)


def create_plot():
    path_results = "/home/simon/datasets/structure_core_unity_test_results/class_mlp_experiments"
    path_results = "/media/simon/T7/datasets/structure_core_unity_test_results/class_mlp_experiments"
    algorithms = ["class_288_r2",
                  "class_384_r2",
                  "class_1280_r2",
                  "class_1920_r2",
                  "class_2688_r2",
                  "class_640_r1",
                  "class_640_r2",
                  "class_640_r3",
                  "class_640_r3_v3",
                  "class_288_r2_v2",
                  "class_384_r2_v2",
                  "class_1280_r2_v2",
                  "class_1536_r2_v2",
                  "class_1920_r2_v2",
                  "class_2688_r2_v2",
                  "class_640_r1_v2",
                  "class_640_r2_v2",
                  "class_640_r3_v2"]
    algorithms = ["class_288_r2",
                  "class_384_r2",
                  "class_1280_r2",
                  "class_1920_r2",
                  "class_2688_r2",
                  "class_640_r1",
                  "class_640_r2",
                  "class_640_r3"]
    legend_names = {"class_288_r2": "288 class 2 layer MLPs",
                    "class_384_r2": "384 class 2 layer MLPs",
                    "class_1280_r2": "1280 class 2 layer MLPs",
                    "class_1920_r2": "1920 class 2 layer MLPs",
                    "class_2688_r2": "2688 class 2 layer MLPs",
                    "class_640_r1": "640 class 1 layer MLPs",
                    "class_640_r2": "640 class 2 layer MLPs",
                    "class_640_r3": "640 class 3 layer MLPs",
                    "class_640_r3_v3": "640 class 3 layer MLPs (3)",
                    "class_288_r2_v2": "288 class 2 layer MLPs (2)",
                    "class_384_r2_v2": "384 class 2 layer MLPs (2)",
                    "class_1280_r2_v2": "1280 class 2 layer MLPs (2)",
                    "class_1536_r2_v2": "1536 class 2 layer MLPs (2)",
                    "class_1920_r2_v2": "1920 class 2 layer MLPs (2)",
                    "class_2688_r2_v2": "2688 class 2 layer MLPs (2)",
                    "class_640_r1_v2": "640 class 1 layer MLPs (2)",
                    "class_640_r2_v2": "640 class 2 layer MLPs (2)",
                    "class_640_r3_v2": "640 class 3 layer MLPs (2)",
                    }
    font = {'family': 'normal',
            #'weight': 'bold',
            'size': 16}

    legends = [legend_names[x] for x in algorithms]

    fig, ax = plt.subplots()
    fig.set_size_inches(4, 2)
    for algorithm in algorithms:
        with open(path_results + f"/{algorithm}.pkl", "rb") as f:
            data = pickle.load(f)
        x = data["inliers"]["ths"]
        y = data["inliers"]["data"] / data["inliers"]["pix_count"]
        #x = x[x < 5]
        #y = y[:len(x)]
        if '640' in algorithm:
            ax.plot(x, y, linestyle='dotted')
        else:
            ax.plot(x, y)

    ax.set(xlim=[0.0, 0.5])
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlabel(xlabel="pixel threshold", fontdict=font)
    ax.set_ylabel(ylabel="inlier ratio", fontdict=font)
    legends = [legend_names[x] for x in algorithms]
    ax.legend(legends)
    #ax.axes([0, 5, 0, 1])

    # NEW VERSIONS WITH WEAKER L2 REGULARIZATION
    algorithms = [
                  "class_288_r2_v2",
                  "class_384_r2_v2",
                  "class_1280_r2_v2",
                  "class_1536_r2_v2",
                  "class_1920_r2_v2",
                  "class_2688_r2_v2",
                  "class_640_r1_v2",
                  "class_640_r2_v2",
                  "class_640_r3_v2"]
    fig, ax = plt.subplots()
    #fig.set_size_inches(4, 2)
    for algorithm in algorithms:

        with open(path_results + f"/{algorithm}.pkl", "rb") as f:
            data = pickle.load(f)

        print(algorithm)
        for i, th in enumerate(data["rmse"]["ths"]):
            print(f"th:{th}")
            square_error = data["rmse"]["square_error"][i]
            inliers = data["rmse"]["inliers"][i]
            inlier_ratio = inliers / data["inliers"]["pix_count"]
            print(f"inlieres: {inlier_ratio}")
            print(f"RMSE: {np.sqrt(square_error / inliers)}")

        x = data["inliers"]["ths"]
        y = data["inliers"]["data"] / data["inliers"]["pix_count"]
        # x = x[x < 5]
        # y = y[:len(x)]
        if '640' in algorithm:
            ax.plot(x, y, linestyle='dotted')
        else:
            ax.plot(x, y)

    ax.set(xlim=[0.0, 0.5])
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlabel(xlabel="pixel threshold", fontdict=font)
    ax.set_ylabel(ylabel="inlier ratio", fontdict=font)
    legends = [legend_names[x][:-4] for x in algorithms]
    ax.legend(legends)

    # COMPARISON WITH REDUCED WEIGHT DECAY (L2 Reg)
    algorithms = ["class_288_r2",
                  "class_1920_r2",
                  "class_2688_r2",
                  "class_288_r2_v2",
                  "class_1920_r2_v2",
                  "class_2688_r2_v2"]
    fig, ax = plt.subplots()
    #fig.set_size_inches(4, 2)
    for algorithm in algorithms:
        with open(path_results + f"/{algorithm}.pkl", "rb") as f:
            data = pickle.load(f)
        x = data["inliers"]["ths"]
        y = data["inliers"]["data"] / data["inliers"]["pix_count"]
        # x = x[x < 5]
        # y = y[:len(x)]
        if 'v2' in algorithm:
            ax.plot(x, y, linestyle='dotted')
        else:
            ax.plot(x, y)

    ax.set(xlim=[0.0, 0.5])
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlabel(xlabel="pixel threshold", fontdict=font)
    ax.set_ylabel(ylabel="inlier ratio", fontdict=font)
    legends = [legend_names[x][:-4] for x in algorithms]
    ax.legend(legends)


    th = 1
    fig, ax = plt.subplots()
    for algorithm in algorithms:
        with open(path_results + f"/{algorithm}.pkl", "rb") as f:
            data = pickle.load(f)
        ax.plot(data[f"inliers_{th}"]["distances"][2:],
                 np.array(data[f"inliers_{th}"]["data"][2:]) / np.array(data[f"inliers_{th}"]["pix_count"][2:]),
                linestyle="dotted")
    ax.legend(legends)
    #ax.axes([0, 10, 0, 1])
    ax.set_xlabel(xlabel="distance [m]", fontdict=font)
    ax.set_ylabel(ylabel=f"inlier ratio ({th} pixel threshold)", fontdict=font)
    plt.show()


#create_data()
create_plot()
