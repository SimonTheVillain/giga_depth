from dataset_rendered_2 import *
import numpy as np
import cv2
import os
import open3d as o3d
from common.common import LCN_np

dataset_path = os.path.expanduser("/media/simon/ssd_data/data/datasets/structure_core_unity_3")
dataset_version = "structure_core_unity_4"

dataset_path = os.path.expanduser("/media/simon/ssd_data/data/datasets/shapenet_rendered")
dataset_version = "shapenet_half_res"
tgt_res = (1216, 896)#(1216, 896)
principal = (604, 457)
focal = 1.1154399414062500e+03
# according to the simulation in unity & the dotpattern extractor (check if this still holds true)
focal_projector = 850
res_projector = 1024
baselines = {"right": 0.07501 - 0.0634, "left": -0.0634}

#we work on half the resolution
focal *= 0.5
principal = (principal[0] * 0.5, principal[1] * 0.5)

datasets, baselines, has_lr, focal, principal, tgt_res = GetDataset(dataset_path, False, tgt_res, version=dataset_version, debug=True)
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
    x = np.arange(0, tgt_res[0]/2).astype(np.float32)
    den = focal_projector * (x[np.newaxis, ...] - principal[0]) - focal * (gt[0, :, :] * res_projector - 511.5)
    cv2.imshow("disparity", den * (1.0 / (focal_projector * focal))) #left camera with lower baseline to projector
    cv2.imshow("neg_disparity", -den * (1.0 / (focal_projector * focal)) * 0.1)#right camera with higher baseline
    d = -np.divide(bl * (focal_projector * focal), den)

    d = (focal * 0.5) * bl / (gt[0, :, :]*tgt_res[0] / 2.0 - np.expand_dims(np.arange(0, gt.shape[2]), 0))
    cv2.imshow("depth_by_gt", d*0.1)
    #cv2.imshow("depth_error", np.abs(d-depth))
    cv2.waitKey()
    #display_pcl(depth)


    #print(ir)

