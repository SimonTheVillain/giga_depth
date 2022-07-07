import os
# os.chdir("../")
import torch
from model.composite_model import CompositeModel
from common.common import downsampleDepth
import cv2
import numpy as np
import pickle
import timeit
import re
from pathlib import Path
from multiprocessing import Pool
import matplotlib
from si_prefix import si_format

matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


# Work in the parent directory
def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def apply_recursively(model, input_root, output_root, measure_time=False):
    src_res = (1401, 1001)
    src_cxy = (700, 500)
    tgt_res = (1216, 896)
    tgt_cxy = (604, 457)
    # the focal length is shared between src and target frame
    focal = 1.1154399414062500e+03

    rr = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1], tgt_res[0], tgt_res[1])
    half_res = False
    regressor_conv = False
    inds = os.listdir(input_root)
    inds = [re.search(r'\d+', s).group() for s in inds]
    inds = set(inds)
    inds = list(inds)
    inds.sort()
    inds = inds[:1000]
    paths = []
    for ind in inds:
        paths.append((input_root + f"/{ind}_left.jpg", output_root + f"/{int(ind):05d}"))

    if measure_time:
        paths = paths[:1000]
        time = 0
        count = 0
    with torch.no_grad():
        for p, p_tgt in paths:
            irl = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if len(irl.shape) == 3:
                # the rendered images are 3 channel bgr
                irl = cv2.cvtColor(irl, cv2.COLOR_BGR2GRAY)
            else:
                # the rendered images are 16
                irl = irl / 255.0
            irl = irl[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]]
            irl = irl.astype(np.float32) * (1.0 / 255.0)

            if half_res:
                irl = cv2.resize(irl, (int(irl.shape[1] / 2), int(irl.shape[0] / 2)))
                irl = irl[:448, :608]
            cv2.imshow("irleft", irl)
            irl = torch.tensor(irl).cuda().unsqueeze(0).unsqueeze(0)

            if measure_time:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                x = model.backbone(irl)
                start.record()
                x = model.regressor(x)
                end.record()
                torch.cuda.synchronize()
                time += start.elapsed_time(end)
                count += 1
                continue

            x = model(irl)
            x = x[0, 0, :, :]
            x = x * x.shape[1]
            x_0 = torch.arange(0, x.shape[1]).unsqueeze(0).cuda()
            x -= x_0
            x *= -1.0

            result = x.cpu()[:, :].numpy()
            # msk = np.clip(msk.cpu()[0, 0, :, :].numpy() * 255, 0, 255).astype(np.uint8)

            # result = coresup_pred.cpu()[0, 0, :, :].numpy()

            p = str(p_tgt) + ".exr"  # = path_out + f"/{int(ind):05d}.exr"
            cv2.imshow("result", result * (1.0 / 50.0))
            cv2.imwrite(p, result)

            p = str(p_tgt) + "_msk.png"  # path_out + f"/{int(ind):05d}_msk.png"
            # cv2.imshow("mask", msk)
            # cv2.imwrite(p, result)
            cv2.waitKey(1)
            print(p)
    if measure_time:
        print(f"time: {time / count}")


path_dst = "/media/simon/T7/datasets/structure_core_unity_test_ablation_results"
path_src = "/media/simon/T7/datasets/structure_core_unity_test"

experiments = [
    ("unet", "full_72_unet"),
    ("unet_line", "full_73_line"),
    ("unet_c1920", "full_78_unet_j2_c1920"),
    ("c288", "full_76_lcn_j2_c288"),
    ("c384", "full_76_lcn_j2_c384"),
    ("c640_r1", "full_76_lcn_j2_c640_r1"),
    ("c640_r2", "full_76_lcn_j2_c640_r2"),
    ("c640_r3", "full_76_lcn_j2_c640_r3"),
    ("c1280", "full_76_lcn_j2_c1280"),
    ("c1536", "full_76_lcn_j2_c1536"),
    ("c1920", "full_76_lcn_j2_c1920"),
    ("c1920_nolcn", "full_76_j2_c1920"),
    ("c2688", "full_76_lcn_j2_c2688"), ]
algorithms = [algorithm for _, algorithm in experiments]


def apply_model():
    measure_time = False
    for experiment, net in experiments:
        print(experiment)
        model_pth = f"trained_models/{net}.pt"
        model = torch.load(model_pth)
        model.cuda()
        print(f"model parameters: {get_n_params(model)}")
        out_folder_experiment = f"{path_dst}/{net}"
        if not os.path.exists(out_folder_experiment):
            os.mkdir(out_folder_experiment)
        model.cuda()
        apply_recursively(model, path_src, out_folder_experiment, measure_time=measure_time)


def process_results(algorithm):
    path_results = path_dst + "/" + algorithm
    inds = os.listdir(path_results)
    inds = [re.search(r'\d+', s).group() for s in inds]
    inds = set(inds)
    inds = list(inds)
    inds.sort()

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
    distances_ths = [0.1, 0.5, 1, 5]

    rmse_ths = np.array([0.1, 0.5, 1, 2, 5])

    data = {"inliers": {"ths": thresholds,
                        "data": [0] * thresholds.shape[0],
                        "pix_count": 0},
            "conditional_inliers": {"th": relative_th_base,
                                    "ths": relative_ths,
                                    "data": [0] * relative_ths.shape[0],
                                    "pix_count": 0},
            "rmse": {"ths": rmse_ths,
                     "square_error": [0] * rmse_ths.shape[0],
                     "inliers": [0] * rmse_ths.shape[0]

                     }
            }
    for ind in inds:
        path_gt = path_src + f"/{int(ind)}_left_d.exr"
        d_gt = cv2.imread(path_gt, cv2.IMREAD_UNCHANGED)
        d_gt = d_gt[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]]

        path_results = path_dst + "/" + algorithm + f"/{int(ind):05d}.exr"
        if not os.path.exists(path_results):
            path_results = path_dst + "/" + algorithm + f"/{int(ind)}.exr"
        estimate = cv2.imread(path_results, cv2.IMREAD_UNCHANGED)

        if d_gt.shape[0] // 2 == estimate.shape[0]:
            # downsample by a factor of 2
            d_gt = downsampleDepth(d_gt)
            disp_gt = focal * baseline * 0.5 / d_gt
            delta = np.abs(disp_gt - estimate) * 2.0  # in case

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

        # cv2.imshow("gt", disp_gt * 0.02)
        # cv2.imshow("estimate", estimate * 0.02)
        # cv2.waitKey()
    f = open(path_dst + f"/{algorithm}.pkl", "wb")
    pickle.dump(data, f)
    f.close()


def process_data(algorithms):
    threading = True

    if threading:
        print("processing the results for all algorithms")
        with Pool(12) as p:
            p.map(process_results, algorithms)
    else:
        for algorithm in algorithms:
            print(f"processing the results for {algorithm}")
            process_results(algorithm)


def create_plot(algorithms):
    report_outliers = True
    legend_names = {"full_76_lcn_j2_c288": "288 class 2 layer MLPs",
                    "full_76_lcn_j2_c384": "384 class 2 layer MLPs",
                    "full_76_lcn_j2_c640_r1": "640 class 1 layer MLPs",
                    "full_76_lcn_j2_c640_r2": "640 class 2 layer MLPs",
                    "full_76_lcn_j2_c640_r3": "640 class 3 layer MLPs",
                    "full_76_lcn_j2_c1280": "1280 class 2 layer MLPs",
                    "full_76_lcn_j2_c1536": "1536 class 2 layer MLPs",
                    "full_76_j2_c1920": "1920 class 2 layer MLPs (no LCN)",
                    "full_76_lcn_j2_c1920": "1920 class 2 layer MLPs",
                    "full_76_lcn_j2_c2688": "2688 class 2 layer MLPs",
                    "full_72_unet": "UNet",
                    "full_73_line": "UNet with per-line output layers",
                    "full_78_unet_j2_c1920": "UNet + 1920 class MLPs",
                    }
    font = {'family': 'normal',
            # 'weight': 'bold',
            'size': 16}

    legends = [legend_names[x] for x in algorithms]

    fig, ax = plt.subplots()
    fig.set_size_inches(4, 2)
    for algorithm in algorithms:
        with open(path_dst + f"/{algorithm}.pkl", "rb") as f:
            data = pickle.load(f)
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
    legends = [legend_names[x] for x in algorithms]
    ax.legend(legends)
    # ax.axes([0, 5, 0, 1])

    fig, ax = plt.subplots()
    # fig.set_size_inches(4, 2)
    for algorithm in algorithms:

        with open(path_dst + f"/{algorithm}.pkl", "rb") as f:
            data = pickle.load(f)

        report_str = f"{algorithm} "
        table_str = ""
        for i, th in enumerate(data["rmse"]["ths"]):
            #print(f"th:{th}")
            square_error = data["rmse"]["square_error"][i]
            inliers = data["rmse"]["inliers"][i]
            inlier_ratio = inliers / data["inliers"]["pix_count"]
            if report_outliers:
                report_str += f"o({th}): {(1 - inlier_ratio) * 100:.2f} "
                table_str += f"& {(1 - inlier_ratio) * 100:.2f} "
            else:
                report_str += f"i({th}): {inlier_ratio} "
            #report_str += f"RMSE for threshold {th}: {np.sqrt(square_error / inliers):.3f}"

        print(report_str)
        print(table_str)
        x = data["inliers"]["ths"]
        y = data["inliers"]["data"] / data["inliers"]["pix_count"]

        if report_outliers:
            y = 1 - y
        if '640' in algorithm:
            ax.plot(x, y, linestyle='dotted')
        else:
            ax.plot(x, y)

    ax.set(xlim=[0.0, 0.5])
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlabel(xlabel="pixel threshold", fontdict=font)
    if report_outliers:
        ax.set_ylabel(ylabel="outlier ratio", fontdict=font)
    else:
        ax.set_ylabel(ylabel="inlier ratio", fontdict=font)
    legends = [legend_names[x][:-4] for x in algorithms]
    ax.legend(legends)

    fig, ax = plt.subplots()
    # fig.set_size_inches(4, 2)
    for algorithm in algorithms:
        with open(path_dst + f"/{algorithm}.pkl", "rb") as f:
            data = pickle.load(f)
        x = data["inliers"]["ths"]
        y = data["inliers"]["data"] / data["inliers"]["pix_count"]
        if report_outliers:
            y = 1 - y

        if 'v2' in algorithm:
            ax.plot(x, y, linestyle='dotted')
        else:
            ax.plot(x, y)

    ax.set(xlim=[0.0, 0.5])
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_xlabel(xlabel="pixel threshold", fontdict=font)
    if report_outliers:
        ax.set_ylabel(ylabel="outlier ratio", fontdict=font)
    else:
        ax.set_ylabel(ylabel="inlier ratio", fontdict=font)
    legends = [legend_names[x][:-4] for x in algorithms]
    ax.legend(legends)

    th = 1
    fig, ax = plt.subplots()
    for algorithm in algorithms:
        with open(path_dst + f"/{algorithm}.pkl", "rb") as f:
            data = pickle.load(f)

        y = np.array(data[f"inliers_{th}"]["data"][2:]) / np.array(data[f"inliers_{th}"]["pix_count"][2:])
        if report_outliers:
            y = 1 - y
        ax.plot(data[f"inliers_{th}"]["distances"][2:],
                y,
                linestyle="dotted")
    ax.legend(legends)
    # ax.axes([0, 10, 0, 1])
    ax.set_xlabel(xlabel="distance [m]", fontdict=font)
    if report_outliers:
        ax.set_ylabel(ylabel=f"outlier ratio ({th} pixel threshold)", fontdict=font)
    else:
        ax.set_ylabel(ylabel=f"inlier ratio ({th} pixel threshold)", fontdict=font)
    plt.show()


def report_parameter_counts():
    param_counts = {}
    measure_time = False
    for experiment, net in experiments:
        print(experiment)
        model_pth = f"trained_models/{net}.pt"
        model = torch.load(model_pth)
        model.cuda()
        param_counts[net] = get_n_params(model)

    for key in param_counts:
        print(f"model {key} has {si_format(param_counts[key], precision=0)} params")


#report_parameter_counts()
#apply_model()
#process_data(algorithms)
create_plot(algorithms)
