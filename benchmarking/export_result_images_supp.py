import cv2
import numpy as np
import matplotlib
matplotlib.use('tkAgg')
from matplotlib import pyplot as plt
import os
import torch

from common.common import LCN_np

def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]

    return color_range.reshape(256, 1, 3)

pth_out = "/home/simon/Pictures/images_paper"

base = "/home/simon/datasets"


def to_depth(disp):

    focal = 1.1154399414062500e+03
    if hasattr(disp, "shape"):
        if disp.shape[1] == 608:
            focal *= 0.5
    baseline = 0.0634

    depth = focal * baseline / np.clip(disp, 0.1, 200)
    return depth

def to_disp(depth, correct_scale=False):

    focal = 1.1154399414062500e+03
    if hasattr(depth, "shape"):
        if len(depth.shape) == 2:
            if depth.shape[1] == 608 and correct_scale:
                focal *= 0.5
    baseline = 0.0634

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

def comparison_structure_ours():
    for i in range(770, 900, 1):
        i = 783
        print(i)
        min_dist = 1
        max_dist = 4
        file = f"{base}/structure_core/sequences_combined_all/{i:03}/ir0.png"
        ir = cv2.imread(file)
        ir = ir[:, 1216:] # extract the left image
        cv2.imshow("ir", ir)

        file = f"{base}/structure_core/sequences_combined_all/{i:03}/depth0.png"
        depth = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000.0

        depth_structure = color_code(depth, min_dist, max_dist)
        cv2.imshow("depth", depth_structure)



        file = f"{base}/structure_core/sequences_combined_all_GigaDepth66LCN/{i:03}/0.exr"
        disp = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        depth = to_depth(disp)
        depth_new = color_code(depth, min_dist, max_dist)
        cv2.imshow("depth_new", depth_new)
        cv2.imwrite(f"{pth_out}/base/comp_st_core_base_ir.png", ir)
        cv2.imwrite(f"{pth_out}/base/comp_st_core_base.png", depth_structure)
        cv2.imwrite(f"{pth_out}/base/comp_st_core_base_new.png", depth_new)

        cv2.waitKey()
        break

def comparison_rendered():
    pth_out = "/home/simon/Pictures/images_paper/supplemental/results_unity"
    c_res = (1401, 1001)
    src_cxy = (700, 500)
    tgt_res = (1216, 896)
    tgt_cxy = (604, 457)
    # the focal length is shared between src and target frame
    focal = 1.1154399414062500e+03
    baseline = 0.0634
    rr = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1], tgt_res[0], tgt_res[1])

    frame_inds = [83, 82, 77, 99]
    algorithms = ["HyperDepth",
                  "ActiveStereoNetFull",
                  "ActiveStereoNet",
                  "connecting_the_dots_stereo",
                  "connecting_the_dots_full",
                  "GigaDepth66LCN", "GigaDepth"]

    diff_limit = 5
    ind = 0
    for frame_ind in frame_inds:
        ind += 1
        print(frame_ind)
        path = f"{base}/structure_core_unity_test/{frame_ind}_left.jpg"
        ir = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        ir = ir[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]]

        path = f"{base}/structure_core_unity_test/{frame_ind}_left_d.exr"
        gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        depth_gt = gt[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]]
        lower_limit = np.min(gt)*0.9
        upper_limit = np.max(gt)*1.1
        disp_gt_raw = to_disp(depth_gt)
        disp_gt_color = color_code(disp_gt_raw, to_disp(upper_limit), to_disp(lower_limit))
        cv2.imshow("disp_gt", disp_gt_color)


        os.mkdir(f"{pth_out}/{ind}")
        pth = f"{pth_out}/{ind}/ir.png"
        cv2.imwrite(pth, ir)
        pth = f"{pth_out}/{ind}/gt.png"
        cv2.imwrite(pth, disp_gt_color)

        cv2.imshow("ir", ir)
        #cv2.imshow("depth_gt", depth_gt/5.0)
        cv2.waitKey(1)

        for algorithm in algorithms:
            path = f"{base}/structure_core_unity_test_results/{algorithm}/{frame_ind}.exr"
            if not os.path.exists(path):
                path = f"{base}/structure_core_unity_test_results/{algorithm}/{frame_ind:05}.exr"

            disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            disp_gt=disp_gt_raw
            if disp.shape[1] == 608:
                disp *= 2.0
                disp_gt = cv2.resize(disp_gt_raw, (608, 448), interpolation = cv2.INTER_NEAREST)

            disp_color = color_code(disp, to_disp(upper_limit), to_disp(lower_limit))
            cv2.imshow("disp", disp_color)

            delta = np.abs(disp - disp_gt)
            delta_color = color_code(delta, 0, diff_limit)
            cv2.imshow("delta", delta_color)

            pth = f"{pth_out}/{ind}/disp_{algorithm}.png"
            cv2.imwrite(pth, disp_color)
            pth = f"{pth_out}/{ind}/delta_{algorithm}.png"
            cv2.imwrite(pth, delta_color)

            #depth = to_depth(disp)
            #cv2.imshow("depth", depth / 5.0)


            cv2.waitKey(1)


def comparison_captured():
    pth_out = "/home/simon/Pictures/images_paper/supplemental/results_structure"
    c_res = (1401, 1001)
    src_cxy = (700, 500)
    tgt_res = (1216, 896)
    tgt_cxy = (604, 457)
    # the focal length is shared between src and target frame
    focal = 1.1154399414062500e+03
    baseline = 0.0634
    rr = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1], tgt_res[0], tgt_res[1])

    algorithms = ["HyperDepth",
                  "ActiveStereoNet",
                  "connecting_the_dots",
                  "GigaDepth66LCN", "GigaDepth"]

    diff_limit = 5
    for frame_ind in range(11):
        print(frame_ind)
        path = f"{base}/structure_core_photoneo_test/{frame_ind:03}/ir0.png"
        ir = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        ir = ir[:, 1216:]

        path = f"{base}/structure_core_photoneo_test_results/GT/GigaDepth66LCN/{frame_ind:03}/0.exr"
        disp_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        cv2.imshow("disp_gt_raw", disp_gt / 100)
        lower_limit = np.min(disp_gt)*0.9
        upper_limit = np.max(disp_gt)*1.1
        disp_gt_color = color_code(disp_gt, lower_limit, upper_limit)
        cv2.imshow("disp_gt", disp_gt_color)

        if not os.path.exists(f"{pth_out}/{frame_ind}"):
            os.mkdir(f"{pth_out}/{frame_ind}")
        pth = f"{pth_out}/{frame_ind}/ir.png"
        cv2.imwrite(pth, ir)
        pth = f"{pth_out}/{frame_ind}/gt.png"
        cv2.imwrite(pth, disp_gt_color)

        cv2.imshow("ir", ir)
        #cv2.imshow("depth_gt", depth_gt/5.0)
        cv2.waitKey()
        for algorithm in algorithms:
            print(algorithm)

            path = f"{base}/structure_core_photoneo_test_results/GT/{algorithm}/{frame_ind:03}/0.exr"
            disp_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            path = f"{base}/structure_core_photoneo_test_results/{algorithm}/{frame_ind:03}/0.exr"
            if not os.path.exists(path):
                path = f"{base}/structure_core_photoneo_test_results/{algorithm}/{frame_ind:03}/0.exr"

            disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            cv2.imshow("disp_raw", disp / 100)
            if disp.shape[1] == 608:
                disp *= 2.0
                disp_gt = cv2.resize(disp_gt, (608, 448), interpolation=cv2.INTER_NEAREST)

            disp_color = color_code(disp, lower_limit, upper_limit)
            #disp_color[disp_gt == 0] = 0
            cv2.imshow("disp", disp_color)

            delta = np.abs(disp - disp_gt)
            cv2.imshow("delta_raw", delta / 100)
            delta_color = color_code(delta, -0.01, diff_limit)
            delta_color[disp_gt == 0] = 0
            cv2.imshow("delta", delta_color)

            pth = f"{pth_out}/{frame_ind}/disp_{algorithm}.png"
            cv2.imwrite(pth, disp_color)
            pth = f"{pth_out}/{frame_ind}/delta_{algorithm}.png"
            cv2.imwrite(pth, delta_color)

            #depth = to_depth(disp)
            #cv2.imshow("depth", depth / 5.0)


            cv2.waitKey(1)
        cv2.waitKey(1)
        #return

def comparison_shapenet():
    c_res = (1401, 1001)
    src_cxy = (700, 500)
    tgt_res = (1216, 896)
    tgt_cxy = (604, 457)
    # the focal length is shared between src and target frame
    focal = 1.1154399414062500e+03
    baseline = 0.0634
    rr = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1], tgt_res[0], tgt_res[1])

    frame_inds = [13,21,22,24,26]
    algorithms = ["GigaDepth",
                  "HyperDepth",
                  "connecting_the_dots"]

    diff_limit = 5
    frame_ind = 22
    print(frame_ind)
    path = f"{base}/shapenet_rendered_compressed_test/syn/{frame_ind:08}/im0_0.png"
    ir = cv2.imread(path, cv2.IMREAD_GRAYSCALE)


    path = f"{base}/shapenet_rendered_compressed_test/syn/{frame_ind:08}/disp0_0.exr"
    disp_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    lower_limit = np.min(disp_gt)*0.9
    upper_limit = np.max(disp_gt)*1.1
    disp_gt_color = color_code(disp_gt, lower_limit, upper_limit)
    disp_gt_raw = disp_gt
    cv2.imshow("disp_gt", disp_gt_color)

    pth = f"{pth_out}/shapenet/ir.png"
    cv2.imwrite(pth, ir)
    pth = f"{pth_out}/shapenet/gt.png"
    cv2.imwrite(pth, disp_gt_color)

    cv2.imshow("ir", ir)
    #cv2.imshow("depth_gt", depth_gt/5.0)
    cv2.waitKey(1)

    for algorithm in algorithms:
        path = f"{base}/shapenet_rendered_compressed_test_results/{algorithm}/{frame_ind:08}/0.exr"
        if not os.path.exists(path):
            path = f"{base}/shapenet_rendered_compressed_test_results/{algorithm}/{frame_ind:08}/im0_0.exr"

        if not os.path.exists(path):
            path = f"{base}/shapenet_rendered_compressed_test_results/{algorithm}/{frame_ind:08}/disp0_0.exr"

        disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        disp_gt=disp_gt_raw
        if disp.shape[1] == 608:
            disp *= 2.0
            disp_gt = cv2.resize(disp_gt_raw, (608, 448), interpolation = cv2.INTER_NEAREST)

        disp_color = color_code(disp, lower_limit, upper_limit)
        cv2.imshow("disp", disp_color)

        delta = np.abs(disp - disp_gt)
        delta_color = color_code(delta, 0, diff_limit)
        cv2.imshow("delta", delta_color)

        pth = f"{pth_out}/shapenet/disp_{algorithm}.png"
        cv2.imwrite(pth, disp_color)
        pth = f"{pth_out}/shapenet/delta_{algorithm}.png"
        cv2.imwrite(pth, delta_color)

        #depth = to_depth(disp)
        #cv2.imshow("depth", depth / 5.0)


        cv2.waitKey()


def store_lcn():
    model = torch.load("trained_models/full_66_j4_backbone.pt")
    model.cpu()
    model = model.slices[0]
    for module in model.block1:
        if isinstance(module, torch.nn.ConstantPad2d):
            module.padding = (0, 0, module.padding[2], module.padding[2])
    for frame_ind in range(11):
        path = f"{base}/structure_core_photoneo_test/{frame_ind:03}/ir0.png"
        ir = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        ir = ir[:, 1216:].astype(np.float32) / 256.0
        lcn = LCN_np(ir)
        lcn = np.clip(lcn * 2 - 1, 0, 1)
        cv2.imshow("ir", ir)
        irt = torch.tensor(ir).unsqueeze(0).unsqueeze(0)
        latent_full = model(irt)
        latent_rgb = np.zeros((latent_full.shape[2], latent_full.shape[3], 3))
        for i in range(3):
            latent = latent_full[0,i,:,:].detach().numpy()
            latent = (latent - np.min(latent)) / (np.max(latent) - np.min(latent))
            latent_rgb[:, :, i] = latent
        #latent_rgb = 0.5 + latent_rgb * 0.5
        latent_rgb = 1- latent_rgb
        cv2.imshow("latent", latent_rgb)

        cv2.imshow("lcn", lcn)


        algorithm = "GigaDepth66LCN"
        path = f"{base}/structure_core_photoneo_test_results/GT/{algorithm}/{frame_ind:03}/0.exr"
        disp_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        lower_limit = np.min(disp_gt)*0.9
        upper_limit = np.max(disp_gt)*1.1
        path = f"{base}/structure_core_photoneo_test_results/{algorithm}/{frame_ind:03}/0.exr"
        disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if disp.shape[1] == 608:
            disp *= 2.0
        disp_color = color_code(disp, lower_limit, upper_limit)
        cv2.imshow("disp", disp_color)


        pth = f"{pth_out}/lcn/disp.png"
        cv2.imwrite(pth, disp_color)

        pth = f"{pth_out}/lcn/ir.png"
        cv2.imwrite(pth, (ir * 255).astype(np.uint8))
        pth = f"{pth_out}/lcn/lcn.png"
        cv2.imwrite(pth, (lcn * 255).astype(np.uint8))
        pth = f"{pth_out}/lcn/latent.png"
        cv2.imwrite(pth, (latent_rgb * 255).astype(np.uint8))
        cv2.waitKey()



#comparison_structure_ours()

#comparison_rendered()
comparison_captured()
#comparison_shapenet()
#store_lcn()