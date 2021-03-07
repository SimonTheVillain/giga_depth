from dataset_rendered_2 import *
import numpy as np
import cv2
import os
import open3d as o3d

dataset_path = os.path.expanduser("/media/simon/ssd_data/data/datasets/structure_core_unity")
tgt_res = (1216, 896)#(1216, 896)
principal = (604, 457)
focal = 1.1154399414062500e+03
# according to the simulation in unity & the dotpattern extractor (check if this still holds true)
focal_projector = 850
res_projector = 1024
baselines = [0.0634 - 0.07501, 0.0634 - 0.0] # left, right

#we work on half the resolution
focal *= 0.5
principal = (principal[0] * 0.5, principal[1] * 0.5)

datasets = GetDataset(dataset_path, False, tgt_res, version=2, debug=True)
dataset = datasets["train"]

def display_pcl(z):
    fx = 1115.44
    cxr = 604.0
    cyr = 896.0 * 0.5
    fx = fx * 0.5
    cxr = cxr * 0.5
    cyr = cyr * 0.5
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
    ir, gt, mask, depth = data
    cv2.imshow("ir", ir[0, :, :])
    cv2.imshow("gt", gt[0, :, :])
    cv2.imshow("mask", mask[0, :, :])
    print(f"mask_minmax = {np.min(mask), np.max(mask)}")
    cv2.imshow("depth", depth*0.1)
    #calculating depth from gt! (according to the side of the ir camera)
    bl = baselines[(i + 0) % 2]
    x = np.arange(0, tgt_res[0]/2).astype(np.float32)
    den = focal_projector * (x[np.newaxis, ...] - principal[0]) - focal * (gt[0, :, :] * res_projector - 511.5)
    cv2.imshow("disparity", den * (1.0 / (focal_projector * focal))) #left camera with lower baseline to projector
    cv2.imshow("neg_disparity", -den * (1.0 / (focal_projector * focal)) * 0.1)#right camera with higher baseline
    d = -np.divide(bl * (focal_projector * focal), den)
    cv2.imshow("depth_by_gt", d*0.1)
    #cv2.imshow("depth_error", np.abs(d-depth))
    cv2.waitKey()
    #display_pcl(depth)


    #print(ir)

