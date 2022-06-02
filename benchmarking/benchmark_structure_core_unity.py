import numpy as np
import cv2
import os
import re
from common.common import downsampleDepth, dilatation
import pickle
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from multiprocessing import Pool
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


def process_results(algorithm):
    path_src = "/media/simon/ssd_datasets/datasets/structure_core_unity_test"
    path_results = "/media/simon/ssd_datasets/datasets/structure_core_unity_test_results"
    path_src = "/home/simon/datasets/structure_core_unity_test"
    path_results = "/home/simon/datasets/structure_core_unity_test_results"
    path_src = "/media/simon/T7/datasets/structure_core_unity_test"
    path_results = "/media/simon/T7/datasets/structure_core_unity_test_results"

    inds = os.listdir(path_src)
    inds = [re.search(r'\d+', s).group() for s in inds]
    inds = set(inds)
    inds = list(inds)
    inds.sort()
    inds = inds[:1000]

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

    edge_ths = [0.5, 1, 2]
    edge_radii = np.arange(2, 32, 4)

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
        output_scale = 1
        if d_gt.shape[0] // 2 == estimate.shape[0]:
            # downsample by a factor of 2
            output_scale = 0.5
            d_gt = downsampleDepth(d_gt)
            disp_gt = focal * baseline * 0.5 / d_gt
            delta = np.abs(disp_gt - estimate) * 2.0 # in case
            d = (focal * baseline * 0.5) / estimate

        else:
            disp_gt = focal * baseline / d_gt
            delta = np.abs(disp_gt - estimate)
            d = (focal * baseline) / estimate

        d[estimate <= 0] = 0
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
                                         "depth_rmse": [0] * (distances.shape[0] - 1),
                                         "pix_count": [0] * (distances.shape[0] - 1)}
            for i in range(len(distances) - 1):
                dist_low = distances[i]
                dist_high = distances[i + 1]
                msk = np.logical_and(d_gt > dist_low, d_gt < dist_high)
                msk_count = np.sum(msk)
                valid_count = np.sum(np.logical_and(delta < threshold, msk))
                depth_rmse = (d-d_gt) * (d-d_gt) * np.logical_and(delta < threshold, msk)
                depth_rmse[np.isnan(depth_rmse)] = 0
                data[f"inliers_{th}"]["depth_rmse"][i] += np.sum(depth_rmse)
                data[f"inliers_{th}"]["data"][i] += valid_count
                data[f"inliers_{th}"]["pix_count"][i] += msk_count

        for th in edge_ths:

            data[f"edge_{th}"] = {
                              "radii": edge_radii,
                              "inlier_count": [0] * edge_radii.shape[0],
                              "depth_rmse": [0] * edge_radii.shape[0],
                              "pix_count": [0] * edge_radii.shape[0]}
            for i in range(len(edge_radii)):
                r = int(edge_radii[i] * output_scale)
                depth_gt = d_gt
                # Create Edge map by executing sobel on depth
                #  threshold
                grad_x = cv2.Sobel(disp_gt, cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(disp_gt, cv2.CV_32F, 0, 1, ksize=3)
                edge_threshold = 2  # a 1 pixel threshold
                edges = (grad_x * grad_x + grad_y * grad_y) > edge_threshold * edge_threshold
                edges = edges.astype(np.float32)
                edges[depth_gt > 5] = 0
                #  dilate
                edges = dilatation(edges, r)

                delta = np.abs(disp_gt - estimate) / output_scale

                inliers = np.logical_and(delta < th, edges)
                inlier_count = np.count_nonzero(inliers)

                depth_rmse = (d-d_gt) * (d-d_gt) * np.logical_and(delta < th, inliers)

                data[f"edge_{th}"]["depth_rmse"][i] += np.sum(depth_rmse)
                data[f"edge_{th}"]["inlier_count"][i] += inlier_count
                data[f"edge_{th}"]["pix_count"][i] += np.sum(edges)

                #cv2.imshow("edges", edges)
                #cv2.waitKey()


        #cv2.imshow("gt", disp_gt * 0.02)
        #cv2.imshow("estimate", estimate * 0.02)
        #cv2.waitKey()
    f = open(path_results + f"/{algorithm}.pkl", "wb")
    pickle.dump(data, f)
    f.close()

def create_data():
    algorithms = ["GigaDepth", "ActiveStereoNet", "connecting_the_dots", "HyperDepth"]#, "GigaDepthLCN"]
    #algorithms = ["HyperDepth"]
    algorithms = ["GigaDepth", "GigaDepth66", "GigaDepth66LCN", "GigaDepth68",
                  "GigaDepth70", "GigaDepth71",
                  "ActiveStereoNet", "ActiveStereoNetFull",
                  "connecting_the_dots_full", "connecting_the_dots_stereo",
                  "HyperDepth"]  #
    algorithms = ["GigaDepth70", "GigaDepth71"]

    threading = True

    if threading:
        with Pool(5) as p:
            p.map(process_results, algorithms)
    else:
        for algorithm in algorithms:
            print(f"evaluating samples for {algorithm}")
            process_results(algorithm)


def create_plot():
    path_results = "/media/simon/ssd_datasets/datasets/structure_core_unity_test_results"
    path_results = "/home/simon/datasets/structure_core_unity_test_results"
    path_src = "/media/simon/T7/datasets/structure_core_unity_test"
    path_results = "/media/simon/T7/datasets/structure_core_unity_test_results"

    algorithms = ["GigaDepth", "ActiveStereoNet", "connecting_the_dots", "HyperDepth"] # "HyperDepth", "GigaDepthLCN"]
    algorithms = ["GigaDepth", "GigaDepth66", "GigaDepth66LCN", "GigaDepth68", "GigaDepth68LCN",
                  "GigaDepth70", "GigaDepth71",
                  "ActiveStereoNet", "ActiveStereoNetFull",
                  "connecting_the_dots_full", "connecting_the_dots_stereo",
                  "HyperDepth"]
    algorithms = ["GigaDepth66", "GigaDepth66LCN",
                  "GigaDepth71",
                  "ActiveStereoNet", "ActiveStereoNetFull",
                  "connecting_the_dots_full", "connecting_the_dots_stereo",
                  "HyperDepth"]
    legend_names = {"GigaDepth": "GigaDepth light",
                    "GigaDepth66": "GigaDepth",
                    "GigaDepth66LCN": "GigaDepth (LCN)",
                    "GigaDepth68": "GigaDepthNew",
                    "GigaDepth68LCN": "GigaDepthNew (LCN)",
                    "GigaDepth70": "GigaDepth70",
                    "GigaDepth71": "GigaDepth71",
                    "ActiveStereoNet": "ActiveStereoNet",
                    "ActiveStereoNetFull": "ActiveStereoNet (full)",
                    "connecting_the_dots_stereo": "ConnectingTheDots",
                    "connecting_the_dots_full": "ConnectingTheDots (full)",
                    "HyperDepth": "HyperDepth"}
    font = {'family': 'normal',
            #'weight': 'bold',
            'size': 16}

    legends = [legend_names[x] for x in algorithms]

    fig, ax = plt.subplots()
    for algorithm in algorithms:
        with open(path_results + f"/{algorithm}.pkl", "rb") as f:
            data = pickle.load(f)
        x = data["inliers"]["ths"]
        y = data["inliers"]["data"] / data["inliers"]["pix_count"]
        #x = x[x < 5]
        #y = y[:len(x)]
        ax.plot(x, y)

    ax.set(xlim=[0.0, 2])
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlabel(xlabel="pixel threshold", fontdict=font)
    ax.set_ylabel(ylabel="inlier ratio", fontdict=font)
    #ax.axes([0, 5, 0, 1])

    ax.legend(legends)

    #plot the inlier ratios over distance
    th = 1
    fig, ax = plt.subplots()
    for algorithm in algorithms:
        with open(path_results + f"/{algorithm}.pkl", "rb") as f:
            data = pickle.load(f)
        ax.plot(data[f"inliers_{th}"]["distances"][2:],
                 np.array(data[f"inliers_{th}"]["data"][2:]) / np.array(data[f"inliers_{th}"]["pix_count"][2:]))
    ax.legend(legends)
    #ax.axes([0, 10, 0, 1])
    ax.set_xlabel(xlabel="distance [m]", fontdict=font)
    ax.set_ylabel(ylabel=f"inlier ratio ({th} pixel threshold)", fontdict=font)


    #plot the RMSE over distance
    th = 5 # this is 1 in the structure core!!!!!
    fig, ax = plt.subplots()
    for algorithm in algorithms:
        print(algorithm)
        with open(path_results + f"/{algorithm}.pkl", "rb") as f:
            data = pickle.load(f)
        rmse = np.sqrt(np.array(data[f"inliers_{th}"]["depth_rmse"][:]) / np.array(data[f"inliers_{th}"]["data"][:]))
        plt.plot(data[f"inliers_{th}"]["distances"][:], rmse)
    plt.legend(legends, loc='upper left')
    ax.set(xlim=[0.0, 6])
    ax.set(ylim=[0.0, 0.12])
    ax.set_xlabel(xlabel="distance [m]", fontdict=font)
    ax.set_ylabel(ylabel=f"RMSE [m]", fontdict=font)


    #plot inlier ratios over pixel proximity to edges
    for th in [0.5, 1, 2]:
        fig, ax = plt.subplots()
        for algorithm in algorithms:
            with open(path_results + f"/{algorithm}.pkl", "rb") as f:
                data = pickle.load(f)
            inliers = np.array(data[f"edge_{th}"]["inlier_count"][:]) / np.array(data[f"edge_{th}"]["pix_count"][:])
            plt.plot(data[f"edge_{th}"]["radii"][:], inliers)
        plt.legend(legends, loc='lower right')
        #ax.set(xlim=[0.0, 6])
        #ax.set(ylim=[0.0, 0.12])
        ax.set_xlabel(xlabel="edge radius", fontdict=font)
        ax.set_ylabel(ylabel=f"inlier ratio ({th} pixel threshold)", fontdict=font)


    #plot RMSE over pixel proximity to edges
    th=1
    fig, ax = plt.subplots()
    for algorithm in algorithms:
        with open(path_results + f"/{algorithm}.pkl", "rb") as f:
            data = pickle.load(f)
        rmse = np.sqrt(np.array(data[f"edge_{th}"]["depth_rmse"][:]) / np.array(data[f"edge_{th}"]["inlier_count"][:]))
        plt.plot( data[f"edge_{th}"]["radii"][:], rmse)
    plt.legend(legends, loc='lower right')
    #ax.set(xlim=[0.0, 6])
    ax.set(ylim=[0.0, 0.3])
    ax.set_xlabel(xlabel="edge radius", fontdict=font)
    ax.set_ylabel(ylabel="RMSE [m]", fontdict=font)
    plt.show()


#create_data()
create_plot()