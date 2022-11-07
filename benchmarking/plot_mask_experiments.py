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

matplotlib.use('tkAgg')


def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]

    return color_range.reshape(256, 1, 3)


def to_depth(disp):

    focal = 1.1154399414062500e+03
    if hasattr(disp, "shape"):
        if disp.shape[1] == 608:
            focal *= 0.5
    baseline = 0.0634

    depth = focal * baseline / np.clip(disp, 0.1, 200)
    return depth


def to_disp(depth, baseline= 0.0634, correct_scale=False):

    focal = 1.1154399414062500e+03
    if hasattr(depth, "shape"):
        if len(depth.shape) == 2:
            if depth.shape[1] == 608 and correct_scale:
                focal *= 0.5

    disp = focal * baseline / np.clip(depth, 0.1, 200)
    return disp


def color_code(im, start, stop):
    msk = np.zeros_like(im)
    msk[np.logical_and(start < im, im < stop)] = 1.0
    im = np.clip(im, start, stop)
    im = (im-start) / float(stop-start)
    im = im * 255.0
    im = im.astype(np.uint8)
    im = cv2.applyColorMap(im, get_mpl_colormap("viridis"))
    im[msk != 1.0] = 0
    return im

def process_frame(model, ir, side, path_output, args):
    ir = torch.tensor(ir, device="cuda").unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        x = model(ir)
        x = x[0, 0, :, :].cpu()
        x = x * x.shape[1] - torch.arange(0, x.shape[1]).unsqueeze(0)
        x *= np.sign(args.baselines[side])
        baseline = args.baselines[side]
        x = x.cpu().numpy()
        lower_limit = to_disp(args.depth_range[1], abs(baseline))
        upper_limit = to_disp(args.depth_range[0], abs(baseline))
        im = color_code(x, lower_limit, upper_limit)
        ir = ir[0, 0, :, :].cpu().numpy()

        if "raw" in args.plot_mode:
            cv2.imwrite(path_output + f"_{side}_disp.exr", x)

        if "colored" in args.plot_mode:
            cv2.imwrite(path_output + f"_{side}_disp.png", im)

        if "combined" in args.plot_mode:
            # save as plt figure:
            fig = plt.figure(figsize=(12, 9*2))
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.imshow(ir, cmap='gray', vmin=-0.01, vmax=1.1)

            ax2 = fig.add_subplot(2, 1, 2)
            ax2.imshow(im[:, :, [2, 1, 0]])

            # Save the full figure...
            fig.savefig(path_output + f"_{side}_combined.png")
            plt.close(fig)

        if "live" in  args.plot_mode:
            cv2.imshow("ir", ir)
            cv2.imshow("disp", im)
            cv2.waitKey(1)

def apply_recursively(model, model_invalidation, input_root, output_root, args):
    # basic parameters of the sensor
    tgt_res = (1216, 896)
    src_cxy = (700, 500)
    tgt_cxy = (604, 457)
    rr = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1], tgt_res[0], tgt_res[1])
    baselines = {"right": 0.07501 - 0.0634, "left": -0.0634}

    # basic parameters for visualization
    depth_range = (0.5, 5.0)
    disp_error_th = 5.0

    if not os.path.exists(output_root):
        os.mkdir(output_root)
    files = os.listdir(input_root)
    for file in files:
        path_input = Path(input_root) / file
        path_output = Path(output_root) / file
        if os.path.isdir(path_input):
            apply_recursively(model, model_invalidation, path_input, path_output, args)
            print(path_input)
        elif len(file) > 3 and (file[:2] == "ir" and file[-3:] == "png"):
            ir = cv2.imread(str(path_input), cv2.IMREAD_UNCHANGED)
            ir = ir.astype(np.float32) / 65536.0
            path_output = str(path_output)[:-4] # take away file_ending
            ir_r = ir[:, :tgt_res[0]]
            ir_l = ir[:, tgt_res[0]:]
            process_frame(model, ir_r, "right", path_output, args)
            process_frame(model, ir_l, "left", path_output, args)

        elif len(file) > 9 and (file[-8:] == "left.jpg" or file[-9:] == "right.jpg"):
            path_gt = str(path_input)[:-4] + "_d.exr"
            ir = cv2.imread(str(path_input), cv2.IMREAD_GRAYSCALE)
            ir = ir[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]]
            ir = ir.astype(np.float32) / 255.0
            ir = torch.tensor(ir, device="cuda").unsqueeze(0).unsqueeze(0)
            depth_gt = cv2.imread(path_gt, cv2.IMREAD_UNCHANGED)
            depth_gt = depth_gt[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]]
            path_output = str(path_output)[:-4]
            # todo, this!!!
            if file[-8:] == "left.jpg":
                baseline = baselines["left"]
            else: # "right"
                baseline = baselines["right"]
            disp_gt = to_disp(depth_gt, baseline, True)
            disp_gt = cv2.resize(disp_gt, (608, 448), interpolation=cv2.INTER_NEAREST) * 0.5 * np.sign(baseline)
            with torch.no_grad():
                x, entropy1, entropy2, entropy3 = model(ir, output_entropies=True)

                focal = 1.1154399414062500e+03
                x_0 = torch.arange(0, x.shape[3]).reshape([1, 1, 1, x.shape[3]]).cuda()
                disp = x * x.shape[3] - x_0
                disp *= -2.0  # scale the disparity
                depth = focal * baseline / disp
                depth[torch.isnan(depth)] = 0
                depth[depth < 0] = 0
                depth[depth > 20] = 0
                features = torch.cat([depth, entropy1, entropy2, entropy3], dim=1)
                invalid_pixel_estimate = model_invalidation(features)


                x = x[0, 0, :, :].cpu()
                x = x * x.shape[1]
                x_0 = torch.arange(0, x.shape[1]).unsqueeze(0)
                x -= x_0
                x *= np.sign(baseline)
                x = x.cpu().numpy()
                lower_limit = np.min(disp_gt)
                upper_limit = np.max(disp_gt)
                im = color_code(x, lower_limit, upper_limit)

                delta = np.abs(x - disp_gt)
                delta_color = color_code(delta, -0.01, disp_error_th)

                if "raw" in args.plot_mode:
                    cv2.imwrite(path_output + "_disp.exr", x)

                if "colored" in args.plot_mode:
                    cv2.imwrite(path_output + "_disp.png", im)
                    cv2.imwrite(path_output + "_delta.png", delta_color)

                if "combined" in args.plot_mode:
                    # save as plt figure:
                    ir = ir[0, 0, :, :].cpu().numpy()
                    fig = plt.figure(figsize=(12, 9*3))
                    ax1 = fig.add_subplot(3, 1, 1)
                    ax1.imshow(ir, cmap='gray', vmin=-0.01, vmax=1.1)

                    ax2 = fig.add_subplot(3, 1, 2)
                    ax2.imshow(im[:, :, [2, 1, 0]])
                    ax3 = fig.add_subplot(3, 1, 3)
                    ax3.imshow(delta_color[:, :, [2, 1, 0]])

                    # Save the full figure...
                    fig.savefig(path_output + '_combined.png')
                    plt.close(fig)

                if "live":
                    cv2.imshow("delta", delta)
                    cv2.imshow("disp", x / upper_limit)
                    cv2.imshow("disp_gt", disp_gt / upper_limit)
                    cv2.imshow("ir", ir)
                    cv2.imshow("im", im)
                    cv2.imshow("delta_color", delta_color)
                    cv2.imshow("invalidation_mask", invalid_pixel_estimate[0, 0, :, :].cpu().numpy())
                    cv2.waitKey(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", dest="input_path", action="store",
                        help="Path to the dataset.")
    parser.add_argument("-o", "--output_path", dest="output_path", action="store",
                        help="Path to the outputs. The folder structure of the inputs will be mirrored there.")
    parser.add_argument("-m", "--model_path", dest="model_path", action="store",
                        help="Path to the model.")
    parser.add_argument("--mask_network_path", dest="mask_network_path", action="store",
                        help="Store")
    parser.add_argument("-p", "--plot_mode", dest="plot_mode", action="store", default="raw_colored_combined_live",
                        help="Mode(s) in which the files are plotted: "
                             "Raw... output exr files with content, colored... color coded output, "
                             "combined... matplotlib it into a file, live... show images on screen")
    args = parser.parse_args()

    # basic parameters of the sensor
    args.tgt_res = (1216, 896)
    args.src_cxy = (700, 500)
    args.tgt_cxy = (604, 457)
    args.rr = (args.src_cxy[0] - args.tgt_cxy[0], args.src_cxy[1] - args.tgt_cxy[1], args.tgt_res[0], args.tgt_res[1])
    args.baselines = {"right": 0.07501 - 0.0634, "left": -0.0634}

    # basic parameters for visualization
    args.depth_range = (0.5, 5.0)
    args.disp_error_th = 5.0
    return args


def run():
    args = parse_args()

    model = torch.load(args.model_path, map_location="cuda")
    model.eval()

    model_invalidation = torch.load(args.mask_network_path, map_location="cuda")
    model_invalidation.eval()

    input_root = args.input_path
    output_root = args.output_path

    apply_recursively(model, model_invalidation, input_root, output_root, args)


if __name__ == '__main__':
    run()