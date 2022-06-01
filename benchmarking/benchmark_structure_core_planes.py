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

base_path = "/media/simon/T7/datasets"

src_res = (1401, 1001)
src_cxy = (700, 500)
tgt_res = (1216, 896)
tgt_cxy = (604, 457)
cxy = (604, 457)
# the focal length is shared between src and target frame
focal = 1.1154399414062500e+03
baseline = 0.0634


def prepare_gt(src_pre="SGBM", src="SGBM", dst="Photoneo"):
    print(dst)
    gt_path = f"{base_path}/structure_core_photoneo_test"
    eval_path = f"{base_path}/structure_core_photoneo_test_results/{src}"#GigaDepth66LCN"
    eval_path_pre = f"{base_path}/structure_core_photoneo_test_results/{src_pre}"#GigaDepth66LCN"
    output_path = f"{base_path}/structure_core_photoneo_test_results/{dst}"

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


    rr = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1], tgt_res[0], tgt_res[1])

    cutoff_dist = 20.0


    thresholds = np.arange(0.05, 20, 0.05)

    relative_th_base = 0.5
    relative_ths = np.arange(0.05, relative_th_base, 0.05)

    distances = np.arange(0.2, 10 - 0.05, 0.20)
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


def processTile(disp):
    rmse_depth = 0
    inliers = 0
    return rmse_depth, inliers


def gen_plane_gt(param, shape):
    gt = np.zeros(shape, dtype=np.float32)
    if shape[0] == 448:
        scale = 0.5
    else:
        scale = 1.0
    cx = cxy[0] * scale
    cy = cxy[1] * scale
    f = focal * scale
    [a, b, c, d] = param

    u = np.expand_dims(np.arange(0, shape[1]), axis=0)
    v = np.expand_dims(np.arange(0, shape[0]), axis=1)
    gt = -d / ((u - cx) * a / f + (v - cy) * b / f + c)

    return gt

def gen_tiled_gt(disp):
    cols = 4
    rows = 3
    #disp[:] = 40
    #o3d.visualization.draw_geometries([to_pcd(to_depth(disp))])

    h = disp.shape[0] // rows
    w = disp.shape[1] // cols
    gt = np.zeros_like(disp)
    for col in range(cols):
        for row in range(rows):
            disp_sub = np.zeros_like(disp)
            disp_sub[h*row: h*(row + 1), w*col: w*(col + 1)] = disp[h*row: h*(row + 1), w*col: w*(col + 1)]
            pcd = to_pcd(to_depth(disp_sub))

            #o3d.visualization.draw_geometries([pcd])
            cv2.imshow("disp_sub", disp_sub * 0.01)
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,#15cm
                                                     ransac_n=3,
                                                     num_iterations=1000)
            gt_partial = gen_plane_gt(plane_model, disp.shape)
            gt[h*row: h*(row + 1), w*col: w*(col + 1)] = gt_partial[h*row: h*(row + 1), w*col: w*(col + 1)]
            cv2.imshow("gt_partial", to_disp(gt_partial) * 0.01)

            cv2.waitKey(1)
            #[a, b, c, d] = plane_model
            #print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

            #inlier_cloud = pcd.select_by_index(inliers)
            #inlier_cloud.paint_uniform_color([1.0, 0, 0])
            #outlier_cloud = pcd.select_by_index(inliers, invert=True)
            #o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, to_pcd(gt_partial)])
            #o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
            #                                  zoom=0.8,
            #                                  front=[-0.4999, -0.1659, -0.8499],
            #                                  lookat=[2.1813, 2.0619, 2.0999],
            #                                  up=[0.1204, -0.9852, 0.1215])
    gt = to_disp(gt)
    return gt


def to_depth(disp):
    tgt_res = (1216, 896)
    # the focal length is shared between src and target frame
    focal = 1.1154399414062500e+03
    baseline = 0.0634
    disp[disp == 0] = 1e8
    if disp.shape[0] == tgt_res[1] // 2:
        # downsample by a factor of 2
        d = (focal * baseline * 0.5) / disp
        print("half_res")
    else:
        d = (focal * baseline) / disp
    d[disp == 1e8] = 0
    return d

def to_disp(depth):
    tgt_res = (1216, 896)
    # the focal length is shared between src and target frame
    focal = 1.1154399414062500e+03
    baseline = 0.0634

    if depth.shape[0] == tgt_res[1] // 2:
        # downsample by a factor of 2
        d = (focal * baseline * 0.5) / depth

    else:
        d = (focal * baseline) / depth
    d[depth == 0] = 0
    return d

def to_pcd(depth):
    focal = 1.1154399414062500e+03
    baseline = 0.0634
    cxy = (604, 457)
    if depth.shape[0] == 448:
        scale = 0.5
        print("half resolution")
    else:
        scale = 1.0
    pts = []
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            d = depth[i, j]
            if d > 0.1 and d < 10:
                x = (j - cxy[0] * scale) * d / (focal * scale)
                y = (i - cxy[1] * scale) * d / (focal * scale)
                pts.append([x, y, d])

    pts = np.array(pts)
    # TODO: RENAME TO PCD
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def process_data_of_algorithm(algorithm):

    path_src_base = f"{base_path}/structure_core_plane_results/{algorithm}"
    paths = []

    for i in range(4):
        for j in range(4):
            ref = f"{path_src_base}/{i:03}/{j}.exr"
            paths.append(ref)

    src_res = (1401, 1001)
    src_cxy = (700, 500)
    tgt_res = (1216, 896)
    tgt_cxy = (604, 457)
    # the focal length is shared between src and target frame
    focal = 1.1154399414062500e+03
    baseline = 0.0634
    rr = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1], tgt_res[0], tgt_res[1])


    distances = np.arange(0.2, 10 - 0.05, 0.5)
    distances_ths = [1, 5]

    data = {}
    for path_src in paths:
        disp = cv2.imread(path_src, cv2.IMREAD_UNCHANGED)
        d = to_depth(disp)
        disp_gt = gen_tiled_gt(disp)
        d_gt = to_depth(disp_gt)
        # collect inlier ratios and RMSE over distance
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
                delta_disp = np.abs(disp_gt - disp)
                if disp_gt.shape[0] == tgt_res[1] // 2:
                    delta_disp *= 2 # double the disparity so we see it as it is in full resolution

                msk = np.logical_and(msk, delta_disp < th)
                msk_count = np.sum(msk)

                squared_error = (d - d_gt) * (d - d_gt) * msk
                squared_error[np.isnan(squared_error)] = 0
                se = np.sum(squared_error)

                #data[f"inliers_{th}"]["depth_rmse"][i] += np.sum(depth_rmse)
                #data[f"inliers_{th}"]["data"][i] += valid_count
                data[f"inliers_{th}"]["pix_count"][i] += msk_count
                data[f"inliers_{th}"]["depth_rmse"][i] += se

        # cv2.imshow("gt", disp_gt * 0.02)
        # cv2.imshow("estimate", estimate * 0.02)
        # cv2.waitKey()
    f = open(base_path + f"/structure_core_plane_results/{algorithm}.pkl", "wb")
    pickle.dump(data, f)
    f.close()


def process_data():
    path_results = f"{base_path}/structure_core_plane_results"
    algorithms = ["GigaDepth", "GigaDepth66", "GigaDepth66LCN", "GigaDepth68", "GigaDepth68LCN",
                  "ActiveStereoNet", "connecting_the_dots",
                  "HyperDepth", "HyperDepthXDomain",
                  "SGBM"]
    algorithms = ["GigaDepth", "GigaDepth66", "GigaDepth66LCN", "ActiveStereoNet", "SGBM"]
    algorithms = ["GigaDepth68LCN"]

    for algorithm in algorithms:
        process_data_of_algorithm(algorithm)
    return


def create_plot():
    path_results = f"{base_path}/structure_core_plane_results"

    algorithms = ["GigaDepth", "GigaDepth66", "GigaDepth66LCN", "GigaDepth68", "GigaDepth68LCN",
                  "ActiveStereoNet", "connecting_the_dots",
                  "HyperDepth", "HyperDepthXDomain",
                  "SGBM"]
    algorithms = ["GigaDepth", "GigaDepth66", "GigaDepth66LCN", "GigaDepth68", "GigaDepth68LCN", "ActiveStereoNet",
                  "HyperDepth", "HyperDepthXDomain", "SGBM"]

    legend_names = {"GigaDepth": "GigaDepth light",
                    "GigaDepth66": "GigaDepth",
                    "GigaDepth66LCN": "GigaDepth (LCN)",
                    "GigaDepth68": "GigaDepthNew",
                    "GigaDepth68LCN": "GigaDepthNew (LCN)",
                    "ActiveStereoNet": "ActiveStereoNet",
                    "ActiveStereoNetFull": "ActiveStereoNet (full)",
                    "connecting_the_dots": "ConnectingTheDots",
                    "connecting_the_dots_stereo": "ConnectingTheDots",
                    "connecting_the_dots_full": "ConnectingTheDots (full)",
                    "HyperDepth": "HyperDepth",
                    "HyperDepthXDomain": "HyperDepthXDomain",
                    "SGBM": "SGBM"}

    legends = [legend_names[x] for x in algorithms]
    font = {'family': 'normal',
            #'weight': 'bold',
            'size': 16}

    # plot the RMSE over distance
    th = 1  # this is 1 in the structure core!!!!!
    fig, ax = plt.subplots()
    for algorithm in algorithms:
        with open(path_results + f"/{algorithm}.pkl", "rb") as f:
            data = pickle.load(f)
        rmse = np.sqrt(np.array(data[f"inliers_{th}"]["depth_rmse"][:]) / np.array(data[f"inliers_{th}"]["pix_count"][:]))
        plt.plot(data[f"inliers_{th}"]["distances"][:], rmse)

        #plt.plot(data[f"inliers_{th}"]["distances"], data[f"inliers_{th}"]["pix_count"])
    plt.legend(legends, loc='upper left')
    ax.set(xlim=[0.0, 4.0])
    ax.set(ylim=[0.0, 0.04])
    ax.set_xlabel(xlabel="distance [m]", fontdict=font)
    ax.set_ylabel(ylabel=f"RMSE [m]", fontdict=font)
    plt.show()


process_data()
create_plot()
