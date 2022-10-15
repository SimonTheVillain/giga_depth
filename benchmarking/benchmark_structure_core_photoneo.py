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

base_path = "/media/simon/ssd_datasets/datasets"
base_path = "/home/simon/datasets"
base_path = "/media/simon/T7/datasets"
report_outlier=True

def prepare_gt(src_pre="SGBM", src="SGBM", dst="Photoneo"):
    print(dst)
    gt_path = f"{base_path}/structure_core_photoneo_test"
    eval_path = f"{base_path}/structure_core_photoneo_test_results/{src}"#GigaDepth66LCN"
    eval_path_pre = f"{base_path}/structure_core_photoneo_test_results/{src_pre}"#GigaDepth66LCN"
    output_path = f"{base_path}/structure_core_photoneo_test_results/{dst}"
    focal = 1.1154399414062500e+03
    baseline = 0.0634
    cxy = (604, 457)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for i in range(11):
        if not os.path.exists(f"{output_path}/{i:03}"):
            os.mkdir(f"{output_path}/{i:03}")
        for j in range(4):
            pth_src_pre = f"{eval_path_pre}/{i:03}/{j}.exr"
            pth_src = f"{eval_path}/{i:03}/{j}.exr"
            pth_gt = f"{gt_path}/{i:03}/gt.ply"
            pth_out = f"{output_path}/{i:03}/{j}.exr"

            print(pth_src_pre)
            disp = cv2.imread(pth_src_pre, cv2.IMREAD_UNCHANGED)
            pcd = generate_pcl(disp)

            # TODO: RENAME TO PCD_REF
            print("Load groundtruth ply")
            print(pth_gt)
            pcd_ref = o3d.io.read_point_cloud(pth_gt)
            pcd_ref = pcd_ref.scale(1.0 / 1000.0, np.zeros((3, 1)))

            print("Apply point-to-point ICP")
            threshold = 0.2#20cm
            trans_init = np.identity(4)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd_ref, pcd, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            #print(reg_p2p)
            #print("Transformation is:")
            #print(reg_p2p.transformation)
            pcd_ref.transform(reg_p2p.transformation)  # apply first transformation



            print(pth_src)
            disp = cv2.imread(pth_src, cv2.IMREAD_UNCHANGED)
            if src == "DepthInSpaceFTSF":  # todo: remove this as soon as the training of DIS is fixed
                print("todo: remove this as soon as the training of DIS is fixed")
                disp *= 0.0634 / 0.075
            pcd = generate_pcl(disp)

            print("Apply FINE point-to-point ICP")
            threshold = 0.02#2cm
            trans_init = np.identity(4)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd_ref, pcd, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            #print(reg_p2p)
            pcd_ref.transform(reg_p2p.transformation)  # apply second transformation

            #print("Transformation is:")
            #print(reg_p2p.transformation)

            print("Initial alignment")
            evaluation = o3d.pipelines.registration.evaluate_registration(
                pcd, pcd_ref, threshold, trans_init)
            print(evaluation)

            disp = generate_disp(pcd_ref)
            cv2.imwrite(pth_out, disp)




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
    if disp.shape[0] == 448:
        scale = 0.5
    else:
        scale = 1.0
    pts = []
    for i in range(disp.shape[0]):
        for j in range(disp.shape[1]):
            d = disp[i, j]
            if d > 10 and d < 100:
                d = baseline * focal * scale / d
                x = (j - cxy[0] * scale) * d / (focal * scale)
                y = (i - cxy[1] * scale) * d / (focal * scale)
                pts.append([x, y, d])

    pts = np.array(pts)
    # TODO: RENAME TO PCD
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def process_results(algorithm):
    path_src = f"{base_path}/structure_core_photoneo_test_results/GT/{algorithm}"
    path_results = f"{base_path}/structure_core_photoneo_test_results"

    paths = []
    for i in range(11):
        for j in range(4):
            ref = f"{path_src}/{i:03}/{j}.exr"
            res = f"{path_results}/{algorithm}/{i:03}/{j}.exr"
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

    distances = np.arange(0.2, 10 - 0.05, 0.25)
    distances_ths = [0.1, 1, 5]
    distances_ths = [1, 5]

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
            delta = np.abs(disp_gt - estimate) * 2.0 # double the disparity
            d_gt = (focal * baseline * 0.5) / disp_gt
            d = (focal * baseline * 0.5) / estimate

        else:
            delta = np.abs(disp_gt - estimate)
            d_gt = (focal * baseline) / disp_gt
            d = (focal * baseline) / estimate

        if algorithm == "DepthInSpaceFTSF":
            print("todo: remove this as soon as the training of DIS is fixed")
            d *= 0.0634 / 0.075

        d_gt[disp_gt == 0] = 0
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


        #collect inlier ratios and RMSE over distance
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
                depth_rmse = (d-d_gt) * (d-d_gt) *  np.logical_and(delta < threshold, msk)

                data[f"inliers_{th}"]["depth_rmse"][i] += np.sum(depth_rmse)
                data[f"inliers_{th}"]["data"][i] += valid_count
                data[f"inliers_{th}"]["pix_count"][i] += msk_count

        #cv2.imshow("gt", disp_gt * 0.02)
        #cv2.imshow("estimate", estimate * 0.02)
        #cv2.waitKey()
    f = open(path_results + f"/{algorithm}.pkl", "wb")
    pickle.dump(data, f)
    f.close()

def create_plot():
    report_outlier = True
    path_results = f"{base_path}/structure_core_photoneo_test_results"

    algorithms = ["GigaDepth", "GigaDepth66", "GigaDepth66LCN", "GigaDepth68", "GigaDepth68LCN",
                  "GigaDepth70", "GigaDepth71", "GigaDepth74", "GigaDepth75",
                  "GigaDepth76", "GigaDepth76LCN",
                  "GigaDepth77LCN",
                  "ActiveStereoNet", "connecting_the_dots",
                  "HyperDepth", "HyperDepthXDomain",
                  "SGBM"]
    algorithms = ["GigaDepth76j4c1280LCN", # "GigaDepth72UNetLCN", "GigaDepth73LineLCN",# "GigaDepth78Uc1920",
                  "DepthInSpaceFTSF",
                  "ActiveStereoNet",
                  "connecting_the_dots",
                  "HyperDepth", "HyperDepthXDomain",
                  "SGBM"]
    algorithms = ["GigaDepth76j4c1280LCN", # "GigaDepth72UNetLCN", "GigaDepth73LineLCN",# "GigaDepth78Uc1920",
                  "DepthInSpaceFTSF",
                  "ActiveStereoNet",
                  "connecting_the_dots",
                  "HyperDepth", #"HyperDepthXDomain",
                  "SGBM"]

    legend_names = {"GigaDepth": "GigaDepth light",
                    "GigaDepth66": "GigaDepth",
                    "GigaDepth66LCN": "GigaDepth (LCN)",
                    "GigaDepth68": "GigaDepth68",
                    "GigaDepth68LCN": "GigaDepth68 (LCN)",
                    "GigaDepth70": "GigaDepth70",
                    "GigaDepth71": "GigaDepth71",
                    "GigaDepth74": "GigaDepth74",
                    "GigaDepth75": "GigaDepth75",
                    "GigaDepth76": "GigaDepth76",
                    "GigaDepth76LCN": "GigaDepth76 (LCN)",
                    "GigaDepth77LCN": "GigaDepth77 (LCN)",
                    "GigaDepth76c1280LCN": "GigaDepth",
                    "GigaDepth76j4c1280LCN": "GigaDepth",
                    "GigaDepth78Uc1920": "GigaDepth (UNet)",
                    "GigaDepth72UNetLCN": "GigaDepth (UNet)",
                    "GigaDepth73LineLCN": "GigaDepth (Line)",
                    "DepthInSpaceFTSF": "DepthInSpace-FTSF",
                    "ActiveStereoNet": "ActiveStereoNet",
                    "ActiveStereoNetFull": "ActiveStereoNet (full)",
                    "connecting_the_dots": "ConnectingTheDots",
                    "connecting_the_dots_stereo": "ConnectingTheDots",
                    "connecting_the_dots_full": "ConnectingTheDots (full)",
                    "HyperDepth": "HyperDepth",
                    "HyperDepthXDomain": "HyperDepthXDomain",
                    "SGBM": "SGM"}
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
        y = data["inliers"]["data"] / data["inliers"]["pix_count"]
        if report_outlier:
            y = 1 - y
        #x = x[x < 5]
        #y = y[:len(x)]
        color, style = get_line_style(algorithm)
        ax.plot(x, y, color=color, linestyle=style)

    ax.set(xlim=[0.0, 5.0])
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlabel(xlabel="pixel threshold", fontdict=font)
    if report_outlier:
        ax.set_ylabel(ylabel="outlier ratio", fontdict=font)
        ax.legend(legends, loc='upper right')
    else:
        ax.set_ylabel(ylabel="inlier ratio", fontdict=font)
        ax.legend(legends, loc='lower right')

    #ax.legend(legends, loc='lower right')

    #plot the inlier ratios over distance
    th = 5 # this is 1 in the structure core!!!!!
    fig, ax = plt.subplots()
    for algorithm in algorithms:
        with open(path_results + f"/{algorithm}.pkl", "rb") as f:
            data = pickle.load(f)
        x = data[f"inliers_{th}"]["distances"][:]
        y = np.array(data[f"inliers_{th}"]["data"][:]) / np.array(data[f"inliers_{th}"]["pix_count"][:])
        if report_outlier:
            y = 1 - y
        color, style = get_line_style(algorithm)
        ax.plot(x, y, color=color, linestyle=style)
    plt.legend(legends, loc='upper right')
    ax.set(xlim=[0.0, 3.5])
    ax.set_xlabel(xlabel="distance [m]", fontdict=font)
    if report_outlier:
        ax.set_ylabel(ylabel=f"outlier ratio ({th} pixel threshold)", fontdict=font)
    else:
        ax.set_ylabel(ylabel=f"inlier ratio ({th} pixel threshold)", fontdict=font)

    #plot the RMSE over distance
    th = 5 # this is 1 in the structure core!!!!!
    fig, ax = plt.subplots()
    if "HyperDepth" in algorithms:
        algorithms.remove("HyperDepth")
    if "HyperDepthXDomain" in algorithms:
        algorithms.remove("HyperDepthXDomain")
    if "connecting_the_dots" in algorithms:
        algorithms.remove("connecting_the_dots")
    legends = [legend_names[x] for x in algorithms]
    for algorithm in algorithms:
        print(algorithm)
        with open(path_results + f"/{algorithm}.pkl", "rb") as f:
            data = pickle.load(f)
        rmse = np.sqrt(np.array(data[f"inliers_{th}"]["depth_rmse"][:]) / np.array(data[f"inliers_{th}"]["data"][:]))
        x = data[f"inliers_{th}"]["distances"][:]
        if algorithm == "DepthInSpaceFTSF":
            rmse = rmse[x < 2.6]
            x = x[x < 2.6]
        color, style = get_line_style(algorithm)
        ax.plot(x, rmse, color=color, linestyle=style)
    plt.legend(legends, loc='upper left')
    ax.set(xlim=[0.0, 3.5])
    ax.set(ylim=[0.0, 0.03])
    ax.set_xlabel(xlabel="distance [m]", fontdict=font)
    ax.set_ylabel(ylabel=f"RMSE [m]", fontdict=font)

    plt.show()

def create_data():
    algorithms = ["GigaDepth", "GigaDepth66", "GigaDepth66LCN", "GigaDepth68", "GigaDepth68LCN",
                  "GigaDepth70", "GigaDepth71",
                  "ActiveStereoNet",
                  "connecting_the_dots",
                  "HyperDepth", "HyperDepthXDomain",
                  "SGBM"]

    algorithms = [ "DepthInSpaceFTSF"]#"GigaDepth72UNetLCN", "GigaDepth73LineLCN"]
    threading = False

    if threading:
        with Pool(5) as p:
            p.map(process_results, algorithms)
    else:
        for algorithm in algorithms:
            process_results(algorithm)
def prepare_gts():
    algorithms = [("GigaDepth", "GigaDepth"),
                  ("GigaDepth66", "GigaDepth66") ,
                  ("GigaDepth66LCN", "GigaDepth66LCN"),
                  ("GigaDepth68", "GigaDepth68"),
                  ("GigaDepth68LCN", "GigaDepth68LCN"),
                  ("GigaDepth70", "GigaDepth70"),
                  ("GigaDepth71", "GigaDepth71"),
                  ("GigaDepth74", "GigaDepth74"),
                  ("GigaDepth75", "GigaDepth75"),
                  ("GigaDepth76", "GigaDepth76"),
                  ("GigaDepth76LCN", "GigaDepth76LCN"),
                  ("ActiveStereoNet", "ActiveStereoNet"),
                  ("connecting_the_dots", "connecting_the_dots"),
                  ("HyperDepth", "HyperDepth"),
                  ("HyperDepthXDomain", "HyperDepthXDomain"),
                  ("SGBM", "SGM")]
    algorithms = [("GigaDepth77LCN", "GigaDepth77LCN")]
    algorithms = [("GigaDepth76c1280LCN", "GigaDepth76c1280LCN"),
                  ("GigaDepth78Uc1920", "GigaDepth78Uc1920"),
                  ("DepthInSpaceFTSF", "DepthInSpaceFTSF")]
    algorithms = [ #("GigaDepth72UNetLCN", "GigaDepth72UNetLCN"),
                   #,("GigaDepth73LineLCN", "GigaDepth73LineLCN"),
                    ("DepthInSpaceFTSF", "DepthInSpaceFTSF")]
    for alg in algorithms:
        prepare_gt(src_pre="GigaDepth66LCN", src=alg[1], dst=f"GT/{alg[0]}")


#prepare_gts()
#create_data()
create_plot()
