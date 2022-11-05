import matplotlib
from matplotlib import pyplot as plt
import torch
import cv2
import numpy as np

# Work in the parent directory
import os
import model
import torch
import os
import cv2
import numpy as np
import re
from pathlib import Path
import yaml
import argparse
import gc
gc.collect()
matplotlib.use('tkAgg')
cmap_name = "viridis"
def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]

    return color_range.reshape(256, 1, 3)


def color_code(im, start, stop):
    msk = np.zeros_like(im)
    msk[np.logical_and(start < im, im < stop)] = 1.0
    im = np.clip(im, start, stop)
    im = (im-start) / float(stop-start)
    im = im * 255.0
    im = im.astype(np.uint8)
    im = cv2.applyColorMap(im, get_mpl_colormap(cmap_name))
    im[msk != 1.0] = 0
    return im


def to_disp(depth, baseline= 0.0634, correct_scale=False):

    focal = 1.1154399414062500e+03
    if hasattr(depth, "shape"):
        if len(depth.shape) == 2:
            if depth.shape[1] == 608 and correct_scale:
                focal *= 0.5

    disp = focal * baseline / np.clip(depth, 0.1, 200)
    return disp

captured = False
rendered = True
if captured:
    pth_in = "/media/simon/T7/datasets/structure_core/sequences_combined_all"
    pth_hd = "/media/simon/T7/datasets/structure_core/sequences_combined_all_hyperdepth"
    pth_gd = "/media/simon/T7/datasets/structure_core/sequences_combined_all_gigadepth"
    pth_as = "/media/simon/T7/datasets/structure_core/sequences_combined_all_activestereonet"
    pth_bm = "/media/simon/T7/datasets/structure_core/sequences_combined_all_SGBM"

    pth_out = "/media/simon/T7/datasets/structure_core/sequences_combined_all_out"

    for i in range(864, 967):
        ir = cv2.imread(pth_in + f"/{i:03d}/ir0.png", cv2.IMREAD_GRAYSCALE)
        ir = ir[:, 1216:]
        hd = cv2.imread(pth_hd + f"/{i:03d}/0.exr", cv2.IMREAD_UNCHANGED)
        gd = cv2.imread(pth_gd + f"/{i:03d}/ir0_left_disp.exr", cv2.IMREAD_UNCHANGED)
        asn = cv2.imread(pth_as + f"/{i:03d}/0.exr", cv2.IMREAD_UNCHANGED)
        sgbm = cv2.imread(pth_bm + f"/{i:03d}/disp0.exr", cv2.IMREAD_UNCHANGED)

        sgbm = sgbm# * (0.0634/0.07501) #baseline between cameras is different than between camera and emitter
        sgbm[sgbm < 1] = 0
        sgbm[np.isnan(sgbm)] = 0
        sgbm[np.isinf(sgbm)] = 0
        #sgbm_flat = sgbm[:]
        upper_bound = np.percentile(sgbm, 95) * 1.3
        lower_bound = np.percentile(sgbm, 10) * 0.9

        sgbm_c = color_code(sgbm, lower_bound, upper_bound)
        hd_c = color_code(hd, lower_bound, upper_bound)
        gd_c = color_code(gd * 2, lower_bound, upper_bound)
        asn_c = color_code(asn * 2, lower_bound, upper_bound)

        cv2.imshow("ir", ir)
        cv2.imshow("sgbm", sgbm*0.01 )
        cv2.imshow("sgbm_c", sgbm_c)
        cv2.imshow("hd", hd * 0.01)
        cv2.imshow("hd_c", hd_c)
        cv2.imshow("gd_c", hd_c)
        cv2.imshow("asn_c", asn_c)
        cv2.waitKey(1)

        # save as plt figure:
        fig = plt.figure(figsize=(12*2*2, 9 * 2 ))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(ir.astype(float) / 256.0, cmap='gray', vmin=-0.01, vmax=1.1)
        ax.set_title("IR (left)")

        ax = fig.add_subplot(2, 4, 3)
        ax.imshow(sgbm_c[:, :, [2, 1, 0]])
        ax.set_title("Semi-Global Block Matching")

        ax = fig.add_subplot(2, 4, 4)
        ax.imshow(hd_c[:, :, [2, 1, 0]])
        ax.set_title("HyperDepth")

        ax = fig.add_subplot(2, 4, 7)
        ax.imshow(asn_c[:, :, [2, 1, 0]])
        ax.set_title("Active Stereo Net")

        ax = fig.add_subplot(2, 4, 8)
        ax.imshow(gd_c[:, :, [2, 1, 0]])
        ax.set_title("GigaDepth")

        pos = ax.get_position()
        x0, y0, width, height = pos.bounds
        cb_ax = fig.add_axes([x0+width*1.1, y0, width*0.05, height]) #left, bottom width height
        fig.colorbar(plt.cm.ScalarMappable(matplotlib.colors.Normalize(lower_bound, upper_bound),
                                           cmap=plt.get_cmap(cmap_name)), cax=cb_ax)

        # Save the full figure...
        fig.savefig(pth_out + f"/{i}.png")
        plt.close(fig)

        plt.clf()
        plt.cla()
        plt.close("all")
        gc.collect()

if rendered:
    # basic parameters of the sensor
    tgt_res = (1216, 896)
    src_cxy = (700, 500)
    tgt_cxy = (604, 457)
    rr = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1], tgt_res[0], tgt_res[1])
    baselines = {"right": 0.07501 - 0.0634, "left": -0.0634}
    disp_error_th = 5.0


    pth_in = "/media/simon/T7/datasets/structure_core_unity_test"
    pth_hd = "/media/simon/T7/datasets/structure_core_unity_test_results/HyperDepth"
    pth_gd = "/media/simon/T7/datasets/structure_core_unity_test_results/GigaDepth76j4c1280LCN"
    pth_as = "/media/simon/T7/datasets/structure_core_unity_test_results/ActiveStereoNet"
    pth_ds = "/media/simon/T7/datasets/structure_core_unity_test_results/DepthInSpaceFTSF"


    pth_out = "/media/simon/T7/datasets/structure_core_unity_test_results/combined"

    for i in range(240, 1000):
        ir = cv2.imread(pth_in + f"/{i}_left.jpg", cv2.IMREAD_GRAYSCALE)
        ir = ir[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]]
        depth_gt = cv2.imread(pth_in + f"/{i}_left_d.exr", cv2.IMREAD_UNCHANGED)
        depth_gt = depth_gt[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]]
        disp_gt = to_disp(depth_gt, np.abs(baselines["left"])) * 0.5
        disp_gt_2 = cv2.resize(disp_gt, (608, 448), interpolation=cv2.INTER_NEAREST)

        hd = cv2.imread(pth_hd + f"/{i}.exr", cv2.IMREAD_UNCHANGED) * 0.5
        gd = cv2.imread(pth_gd + f"/{i:05d}.exr", cv2.IMREAD_UNCHANGED)
        asn = cv2.imread(pth_as + f"/{i:05d}.exr", cv2.IMREAD_UNCHANGED)
        dis = cv2.imread(pth_ds + f"/{i}.exr", cv2.IMREAD_UNCHANGED)

        #sgbm_flat = sgbm[:]
        lower_bound = np.min(disp_gt)
        upper_bound = np.max(disp_gt)

        # calculate color coded deltas
        gt_c = color_code(disp_gt, lower_bound, upper_bound)
        hd_c = color_code(hd, lower_bound, upper_bound)
        gd_c = color_code(gd, lower_bound, upper_bound)
        asn_c = color_code(asn, lower_bound, upper_bound)
        dis_c = color_code(dis, lower_bound, upper_bound)

        # calculate the color coded deltas
        hd_cd = color_code(np.abs(disp_gt - hd), -0.01, disp_error_th)
        gd_cd = color_code(np.abs(disp_gt_2 - gd), -0.01, disp_error_th)
        asn_cd = color_code(np.abs(disp_gt_2 - asn), -0.01, disp_error_th)
        dis_cd = color_code(np.abs(disp_gt_2 - dis), -0.01, disp_error_th)

        cv2.imshow("ir", ir)
        cv2.imshow("gt", gt_c)
        cv2.imshow("hd_c", hd_c)
        cv2.imshow("gd_c", gd_c)
        cv2.imshow("asn_c", asn_c)
        cv2.imshow("dis_c", dis_c)
        cv2.imshow("hd_cd", hd_cd)
        cv2.imshow("gd_cd", gd_cd)
        cv2.imshow("asn_cd", asn_cd)
        cv2.imshow("dis_cd", dis_cd)
        cv2.waitKey(1)


        # save as plt figure:
        fig = plt.figure(figsize=(12*5, 9 * 2 ))
        ax = fig.add_subplot(2, 5, 1)
        ax.imshow(ir.astype(float) / 256.0, cmap='gray', vmin=-0.01, vmax=1.1)
        ax.set_title("IR (left)")

        ax = fig.add_subplot(2, 5, 6)
        ax.imshow(gt_c[:, :, [2, 1, 0]])
        ax.set_title("Ground-truth")

        ax = fig.add_subplot(2, 5, 2)
        ax.imshow(hd_c[:, :, [2, 1, 0]])
        ax.set_title("HyperDepth")

        ax = fig.add_subplot(2, 5, 3)
        ax.imshow(asn_c[:, :, [2, 1, 0]])
        ax.set_title("Active Stereo Net")

        ax = fig.add_subplot(2, 5, 4)
        ax.imshow(dis_c[:, :, [2, 1, 0]])
        ax.set_title("Depth In Space")

        ax = fig.add_subplot(2, 5, 5)
        ax.imshow(gd_c[:, :, [2, 1, 0]])
        ax.set_title("GigaDepth")

        #ax = fig.add_subplot(2, 6, 6, aspect=10.0)
        pos = ax.get_position()
        x0, y0, width, height = pos.bounds
        cb_ax = fig.add_axes([x0+width*1.1, y0, width*0.05, height]) #left, bottom width height
        fig.colorbar(plt.cm.ScalarMappable(matplotlib.colors.Normalize(lower_bound, upper_bound),
                                           cmap=plt.get_cmap(cmap_name)), cax=cb_ax)

        ax = fig.add_subplot(2, 5, 7)
        ax.imshow(hd_cd[:, :, [2, 1, 0]])
        ax.set_title("HyperDepth (error)")

        ax = fig.add_subplot(2, 5, 8)
        ax.imshow(asn_cd[:, :, [2, 1, 0]])
        ax.set_title("Active Stereo Net (error)")

        ax = fig.add_subplot(2, 5, 9)
        ax.imshow(dis_cd[:, :, [2, 1, 0]])
        ax.set_title("Depth In Space (error)")

        ax = fig.add_subplot(2, 5, 10)
        ax.imshow(gd_cd[:, :, [2, 1, 0]])
        ax.set_title("GigaDepth (error)")

        pos = ax.get_position()
        x0, y0, width, height = pos.bounds
        cb_ax = fig.add_axes([x0+width*1.1, y0, width*0.05, height]) #left, bottom width height
        fig.colorbar(plt.cm.ScalarMappable(matplotlib.colors.Normalize(0, disp_error_th),
                                           cmap=plt.get_cmap(cmap_name)), cax=cb_ax)

        # Save the full figure...
        fig.savefig(pth_out + f"/{i}.png")
        plt.close(fig)
        plt.clf()
        plt.cla()
        plt.close("all")
        gc.collect()