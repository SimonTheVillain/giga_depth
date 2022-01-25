import numpy as np
import cv2
import os
import open3d as o3d
from common.common import LCN_np
from dataset.datasets import GetDataset

#todo: remove
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

dataset_path = os.path.expanduser("~/datasets/structure_core_unity_sequences")
dataset_version = "structure_core_unity_sequences"
tgt_res = (1216, 896)#(1216, 896)
principal = (604, 457)
focal = 1.1154399414062500e+03
baselines = {"right": 0.07501 - 0.0634, "left": -0.0634}


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
    im = cv2.applyColorMap(im, get_mpl_colormap("viridis"))
    im[msk != 1.0] = 0
    return im

#we work on half the resolution
focal *= 0.5
principal = (principal[0] * 0.5, principal[1] * 0.5)

datasets, baselines, has_lr, focal, principal, tgt_res = GetDataset(dataset_path, tgt_res, version=dataset_version, debug=True)
dataset = datasets["train"]

def display_pcl(z, fx= 1115.44 * 0.5, cxr= 604.0*0.5, cyr=896*0.5*0.5):
    print(z.shape)
    pts = []
    for i in range(0, z.shape[0]):
        for j in range(0, z.shape[1]):
            y = i - cyr
            x = j - cxr
            depth = z[i, j]
            if 0 < depth < 5:
                pts.append([x*depth/fx, y*depth/fx, depth])
    xyz = np.array(pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])

for i, data in enumerate(dataset):
    ir, gt, mask, edges, depth = data
    cv2.imshow("ir", ir[0, :, :])
    cv2.imshow("gt", gt[0, :, :])
    cv2.imshow("mask", mask[0, :, :])
    cv2.imshow("depth", depth*0.1)
    cv2.imshow("edges", edges[0, :, :])
    cv2.waitKey()

    side = "left"
    if i % 2 == 1 and has_lr:
        side = "right"

    #calculating depth from gt! (according to the side of the ir camera)
    bl = baselines[side]
    x = np.arange(0, gt.shape[2]).astype(np.float32)
    x = np.expand_dims(np.expand_dims(x, 0), 0)
    disp = gt * gt.shape[2] - x
    depth = focal[0] * bl / disp
    cv2.imshow("disparity", disp[0, :, :] * 0.1) #left camera with lower baseline to projector
    cv2.imshow("neg_disparity", -disp[0, :, :] * 0.1)#right camera with higher baseline
    cv2.imshow("depth_by_gt", depth[0, :, :]*0.1)
    cv2.waitKey()


    #TODO: delete
    continue
    pth_out = "/home/simon/Pictures/images_paper/supplemental/edge"

    cv2.imwrite(f"{pth_out}/ir.png", (ir[0, :, :]*255).astype(np.uint8))
    depth = color_code(depth[0,:,:], np.min(depth), np.max(depth))
    cv2.imwrite(f"{pth_out}/depth.png", depth)
    cv2.imwrite(f"{pth_out}/edge.png", (edges[0, :, :]*255).astype(np.uint8))


