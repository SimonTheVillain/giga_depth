import os.path

import open3d as o3d
import numpy as np
import cv2

import re
from common.common import downsampleDepth, downsampleDisp
import pickle
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from benchmarking.plot_settings import get_line_style

base_path = "/home/simon/datasets"
base_path = "/media/simon/T7/datasets"

def generate_disp(pcd):
    focal = 1.1154399414062500e+03
    baseline = 0.0634
    cxy = (604, 457)
    depth = np.ones((896, 1216), dtype=np.float32) * 100.0
    for pt in pcd.points:
        d = pt[2]
        if d==0.0:
            continue
        xy = [pt[0] * focal / d + cxy[0], pt[1] * focal / d + cxy[1]]
        xy = [int(xy[0] + 0.5), int(xy[1] + 0.5)]
        if xy[0] >=0 and xy[0] < depth.shape[1] and xy[1] >= 0 and xy[1] < depth.shape[0]:
            depth[xy[1], xy[0]] = min(d, depth[xy[1], xy[0]])

    depth[depth == 100.0] = 0


    disp = baseline * focal / depth
    disp[depth == 0] = 0
    cv2.imshow("depth", depth * 0.1)
    cv2.imshow("disp", disp * 0.01)
    cv2.waitKey(100)
    return disp
def generate_pcl(disp):
    focal = 1.1154399414062500e+03
    baseline = 0.0634
    cxy = (604, 457)
    pts = []
    for i in range(disp.shape[0]):
        for j in range(disp.shape[1]):
            d = disp[i, j]
            if d > 10 and d < 100:
                d = baseline * focal * 0.5 / d
                x = (j - cxy[0] * 0.5) * d / (focal * 0.5)
                y = (i - cxy[1] * 0.5) * d / (focal * 0.5)
                pts.append([x, y, d])

    pts = np.array(pts)
    # TODO: RENAME TO PCD
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def process_results(algorithm):
    path_src = f"{base_path}/shapenet_rendered_compressed_test/syn"
    path_results = f"{base_path}/shapenet_rendered_compressed_test_results"

    paths = []
    for i in range(1024):
        for j in range(4):
            ref = f"{path_src}/{i:08}/disp0_{j}.exr"
            res = f"{path_results}/{algorithm}/{i:08}/{j}.exr"
            if not os.path.exists(res):
                res = f"{path_results}/{algorithm}/{i:08}/im0_{j}.exr"

            if not os.path.exists(res):
                res = f"{path_results}/{algorithm}/{i:08}/disp0_{j}.exr"
            if not os.path.exists(res):
                print("shit! no file!?")

            paths.append((ref, res))

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
    for path_gt, path_src in paths:
        disp_gt = cv2.imread(path_gt, cv2.IMREAD_UNCHANGED)

        estimate = cv2.imread(path_src, cv2.IMREAD_UNCHANGED)

        if disp_gt.shape[0] // 2 == estimate.shape[0]:
            # downsample by a factor of 2
            disp_gt = downsampleDisp(disp_gt) * 0.5
            delta = np.abs(disp_gt - estimate) * 2.0 # double the dispari

        else:
            delta = np.abs(disp_gt - estimate)

        cv2.imshow("gt", disp_gt / 100)
        cv2.imshow("estimate", estimate/100)
        delta2 = delta
        delta2[disp_gt == 0.0] = 0.0
        cv2.imshow("delta", np.abs(delta2) / 100)
        cv2.waitKey(1)
        msk = disp_gt > 0
        msk_count = np.sum(msk)
        data["inliers"]["pix_count"] += msk_count
        for i, threshold in enumerate(thresholds):

            valid_count = np.sum(np.logical_and(delta < threshold, msk))
            data["inliers"]["data"][i] += valid_count

        # todo: simon! do you really need this? I THINK IT IS NOT NECESSARY
        msk = np.logical_and(disp_gt > 0, delta < relative_th_base)
        msk_count = np.sum(msk)
        data["conditional_inliers"]["pix_count"] += msk_count
        for i, threshold in enumerate(relative_ths):
            valid_count = np.sum(np.logical_and(delta < threshold, msk))
            data["conditional_inliers"]["data"][i] += valid_count


    f = open(path_results + f"/{algorithm}.pkl", "wb")
    pickle.dump(data, f)
    f.close()

def create_plot():
    path_results = f"{base_path}/shapenet_rendered_compressed_test_results"

    algorithms = ["GigaDepth",
                  "connecting_the_dots",
                  "HyperDepth"]

    legend_names = {"GigaDepth": "GigaDepth",
                    "GigaDepth66": "GigaDepth",
                    "GigaDepth66LCN": "GigaDepth (LCN)",
                    "ActiveStereoNet": "ActiveStereoNet",
                    "ActiveStereoNetFull": "ActiveStereoNet (full)",
                    "connecting_the_dots": "ConnectingTheDots",
                    "connecting_the_dots_stereo": "ConnectingTheDots",
                    "connecting_the_dots_full": "ConnectingTheDots (full)",
                    "HyperDepth": "HyperDepth",
                    "HyperDepth2": "HyperDepth2",
                    "HyperDepthXDomain": "HyperDepthXDomain",
                    "SGBM": "Semi-global Block-matching"}
    legends = [legend_names[x] for x in algorithms]
    font = {'family': 'normal',
            #'weight': 'bold',
            'size': 16}
    #plt.figure(1)
    #for algorithm in algorithms:
    #    with open(path_results + f"/{algorithm}.pkl", "rb") as f:
    #        data = pickle.load(f)
    #    plt.plot(data["inliers"]["ths"], data["inliers"]["data"] / data["inliers"]["pix_count"])
    #plt.legend(algorithms)
    fig, ax = plt.subplots()
    for algorithm in algorithms:
        with open(path_results + f"/{algorithm}.pkl", "rb") as f:
            data = pickle.load(f)
        x = data["inliers"]["ths"]
        y = 1 - data["inliers"]["data"] / data["inliers"]["pix_count"]
        #x = x[x < 5]
        #y = y[:len(x)]
        get_line_style(algorithm)
        color, style = get_line_style(algorithm)
        ax.plot(x, y, color=color, linestyle=style)

    ax.set(xlim=[0.0, 1])
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlabel(xlabel="pixel threshold", fontdict=font)
    ax.set_ylabel(ylabel="outlier ratio", fontdict=font)

    ax.legend(legends, loc='upper right')
    # todo: maybe reinstert the plots by threshold
    #th = 5
    #plt.figure(2)
    #for algorithm in algorithms:
    #    with open(path_results + f"/{algorithm}.pkl", "rb") as f:
    #        data = pickle.load(f)
    #    plt.plot(data[f"inliers_{th}"]["distances"][2:],
    #             np.array(data[f"inliers_{th}"]["data"][2:]) / np.array(data[f"inliers_{th}"]["pix_count"][2:]))
    #plt.legend(algorithms)
    plt.show()

def create_data():
    algorithms = ["GigaDepth",
                  "connecting_the_dots",
                  "HyperDepth"]
    algorithms = ["HyperDepth2"] #TODO: find bug in the hyperdepth implementation!!!!
    #algorithms = ["HyperDepthXDomain"]
    threading = False

    if threading:
        with Pool(5) as p:
            p.map(process_results, algorithms)
    else:
        for algorithm in algorithms:
            process_results(algorithm)
#create_data()
create_plot()