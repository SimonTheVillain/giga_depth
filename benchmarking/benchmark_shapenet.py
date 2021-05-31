import torch
from model.composite_model import CompositeModel
import cv2
import os
import numpy as np
from pathlib import Path
import pickle

#Work in the parent directory
os.chdir("../")

path_src = Path("/media/simon/ext_ssd/datasets/shapenet_rendered_test_compressed")
path_results = Path("/media/simon/ext_ssd/datasets/shapenet_rendered_test_results")
algorithms = ["HyperDepth", "GigaDepth", "GigaDepthNoLCN", "connecting_the_dots", "connecting_the_dots_direct"]
scales = {"HyperDepth": 1.0, "GigaDepth": 1.0, "GigaDepthNoLCN": 1.0, "connecting_the_dots": 1.0, "connecting_the_dots_direct": 1.0}

def files_recursively(input_root, output_root, current_sub, algorithms, result):
    current_src = input_root / current_sub

    for path in os.listdir(current_src):
        current = current_src / path
        if os.path.isdir(current):

            files_recursively(input_root, output_root, current_sub / path, algorithms, result)
            continue
        if path[-3:] == 'png' and path[:3] == "im0":
            for alg in algorithms:
                result[alg]["im"].append(current)
                result[alg]["disp_gt"].append(current_src / f"disp0{path[3:-4]}.exr")

                current_dst = output_root / alg / current_sub
                result[alg]["disp"].append(current_dst / f"disp0{path[3:-4]}.exr")
                msk_pth = current_dst / f"mask0{path[3:-4]}.png"
                if os.path.exists(msk_pth):
                    result[alg]["msk"].append(msk_pth)


serialized_data = {"results": {}, "sources": {}}
files = {}
for alg in algorithms:
    files[alg] = {"im": [], "disp_gt": [], "disp": [], "msk": []}
files_recursively(path_src, path_results, Path(""), algorithms, files)

for key, value in files.items():
    serialized_data["sources"][key] = value
    scale = scales[key]
    absolute_outlier_ths = [0.5, 1, 2, 5]
    absolute_outlier_ths_to_print = [0.5, 1, 2, 5]
    absolute_outlier_ths = list(np.arange(0.05, 5, 0.05))
    relative_th = 2
    relative_outlier_ths = list(np.arange(0.05, 0.5, 0.05))

    serialized_data["absolute_outlier_ths"] = absolute_outlier_ths
    serialized_data["relative_outlier_ths"] = relative_outlier_ths
    serialized_data["relative_th"] = relative_th

    absolute_count = 0
    absolute_outlier_counts = [0] * len(absolute_outlier_ths)

    relative_count = 0
    relative_outlier_counts = [0] * len(relative_outlier_ths)


    for i in range(len(value["im"])):
        path = value["im"][i]
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        path = value["disp_gt"][i]
        disp_gt = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        path = value["disp"][i]
        disp = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) * scale

        delta = np.abs(disp - disp_gt)

        absolute_count += disp.shape[0] * disp.shape[1]
        for j, th in enumerate(absolute_outlier_ths):
            count = np.sum(delta > th)
            absolute_outlier_counts[j] += count

        relative_count += np.sum(delta < relative_th)
        for j, th in enumerate(relative_outlier_ths):
            count = np.sum(np.logical_and(delta > th, delta < relative_th))
            relative_outlier_counts[j] += count

    print(key)
    absolute_outliers = []
    outlier_ratios = []
    for j, th in enumerate(absolute_outlier_ths):
        ratio = absolute_outlier_counts[j] / absolute_count
        outlier_ratios.append(ratio)
        if th in absolute_outlier_ths_to_print:
            print(f"outliers for th {th:.2f} o({th:.2f}) = {ratio}")
    serialized_data["results"][key] = {"outlier_ratios": outlier_ratios}

    outlier_ratios = []
    for j, th in enumerate(relative_outlier_ths):
        ratio = relative_outlier_counts[j] / relative_count
        outlier_ratios.append(ratio)
        print(f"relative outliers for th {th:.2f} o({th:.2f}|{relative_th}) = {ratio}")
    serialized_data["results"][key]["relative_outlier_ratios"] = outlier_ratios

f = open("benchmarking/shapenet_results.pkl", "wb")
pickle.dump(serialized_data, f)
f.close()