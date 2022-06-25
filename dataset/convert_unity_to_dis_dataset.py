import h5py
import numpy as np
import cv2
import pickle
import yaml
import json
import os
import os.path
from pathlib import Path
from common.common import downsampleDisp, downsample, downsampleDepth
import open3d as o3d
import multiprocessing



with open("configs/paths-local.yaml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)

in_root_path = str(Path("/media/simon/T7/datasets/structure_core_unity_sequences"))
out_root_path = str(Path("/media/simon/sandisk/datasets/structure_core_unity_sequences_DIS"))

src_cxy = (700, 500)
tgt_res = (1216, 896)
tgt_cxy = (604, 457)
# the focal length is shared between src and target frame
focal = 1.1154399414062500e+03
baseline = 0.0634
baselineLR = 0.07501

rr = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1], tgt_res[0], tgt_res[1])
f_2 = 0.5 * focal
cx = 0.5 * tgt_cxy[0]
cy = 0.5 * tgt_cxy[1]
K = np.array([[f_2, 0, cx], [0, f_2, cy], [0, 0, 1]])
imsize = (448, 608)

#todo: get the proper pattern
pattern = np.zeros([457, 604, 3])
pattern = cv2.imread("common/square_pattern_12_x2.png")
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)
# point at the top left of the reference pattern.
# as the pattern is of twice the resolution we need to upscale it
# TODO: check if the sampling of the pattern position should not be shifted by half a pixel!?
tl = [int(pattern.shape[0] * 0.5 - tgt_cxy[1] * 2), int(pattern.shape[1] * 0.5 - tgt_cxy[0] * 2)] # point at the top left
pattern = pattern[tl[0]:tl[0] + tgt_res[1] * 2, tl[1]:tl[1] + tgt_res[0] * 2]
pattern = downsample(pattern)
pattern = downsample(pattern) / 255.0
pattern = np.dstack((pattern, pattern, pattern))

cv2.imshow("pattern", pattern)
cv2.waitKey(1)
settings = {
    "baseline": baseline,
    "baselineLR": baselineLR,
    "K": K.astype(np.float32),
    "imsize": imsize,
    "pattern": pattern.astype(np.float32)
}
with open(f"{out_root_path}/settings.pkl", 'wb') as handle:
    pickle.dump(settings, handle, protocol=4)#pickle.HIGHEST_PROTOCOL)

files = os.listdir(in_root_path)
files.sort()
folders = files
if False:
    # filter the folders that do not already have a proper output
    filtered = []
    for folder in folders:
        folder_ind = int(folder)
        out_path = f"{out_root_path}/{folder_ind:08d}"
        if not os.path.exists(out_path):
            filtered.append(folder)
        elif len(os.listdir(out_path)) == 0:
            print(f"output folder {out_path} was empty!")
            filtered.append(folder)

    print(f"{len(filtered)} sequences left to fill!")
    folders = filtered

if True:

    print(f"reading and sorting the depth images")
    # filter by nr of pixel with disparity smaller than 128 and distance < than 6 meters:
    disp_th = 128
    dist_th = 5
    filtered = []
    for folder in folders:
        print(folder)
        in_path = f"{in_root_path}/{folder}"
        gt = cv2.imread(f"{in_path}/0_left_gt.exr", cv2.IMREAD_UNCHANGED)
        depth = gt[:, :, 0]
        disp = focal * baseline / depth
        disp = disp[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]]
        disp = downsampleDisp(disp) / 2.0
        depth = depth[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]]
        depth = downsampleDepth(depth)

        in_range_pixel = np.count_nonzero(np.logical_and(disp < disp_th, depth < dist_th))
        filtered.append((in_range_pixel, folder))

    filtered.sort(key=lambda y: y[0])
    folders = [y for (x, y) in filtered[-9216:]]
    folders.sort()



#http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(f":: Apply fast global registration with distance threshold {distance_threshold}")
    #result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
    #    source_down, target_down, source_fpfh, target_fpfh,
    #    o3d.pipelines.registration.FastGlobalRegistrationOption(
    #        maximum_correspondence_distance=distance_threshold))
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def generate_pcl(depth, focal, cxy):
    #depth = np.expand_dims(depth, 2)
    y = np.reshape(np.arange(depth.shape[0]), [depth.shape[0], 1])
    x = np.reshape(np.arange(depth.shape[1]), [1, depth.shape[1]])
    x = (x - cxy[0]) * depth / focal
    y = (y - cxy[1]) * depth / focal
    pts = np.stack((x.flatten(), y.flatten(), depth.flatten()), axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def process(folder, folder_ind, visualize):
    print(folder)
    in_path = f"{in_root_path}/{folder}"
    out_path = f"{out_root_path}/{folder_ind:08d}"
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    im = np.zeros([4, 1, 448, 608], dtype=np.float32)
    imr = np.zeros([4, 1, 448, 608], dtype=np.float32)
    grad = np.zeros([4, 1, 448, 608], dtype=np.float32)
    ambient = np.zeros([4, 1, 448, 608], dtype=np.float32)
    disp = np.zeros([4, 1, 448, 608], dtype=np.float32)
    R = np.zeros([4, 3, 3], dtype=np.float32)
    t = np.zeros([4, 3], dtype=np.float32)
    pcds = []
    for i in range(4):
        #print(f"{folder}_{i}")
        imi = cv2.imread(f"{in_path}/{i}_left.png")
        imi = cv2.cvtColor(imi, cv2.COLOR_BGR2GRAY)
        imi = imi[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]] / 255.0
        imi = downsample(imi)
        im[i, 0, :, :] = imi

        imri = cv2.imread(f"{in_path}/{i}_right.png")
        imri = cv2.cvtColor(imri, cv2.COLOR_BGR2GRAY)
        imri = imri[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]] / 255.0
        imri = downsample(imri)
        imr[i, 0, :, :] = imri

        ambi = cv2.imread(f"{in_path}/{i}_left_amb.png")
        ambi = cv2.cvtColor(ambi, cv2.COLOR_BGR2GRAY)
        ambi = ambi[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]] / 255.0
        ambi = downsample(ambi)
        ambient[i, 0, :, :] = ambi

        gt = cv2.imread(f"{in_path}/{i}_left_gt.exr", cv2.IMREAD_UNCHANGED)
        depth = gt[:, :, 0]
        dispi = focal * baseline / depth
        dispi = dispi[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]]
        dispi = downsampleDisp(dispi) / 2.0
        disp[i, 0, :, :] = dispi
        if visualize:
            cv2.imshow("imi", imi)
            cv2.imshow("ambienti", ambi)
            cv2.imshow("dispi", dispi / 100)
            cv2.waitKey(1)

        pcd = generate_pcl(depth[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]], focal, tgt_cxy)
        # o3d.visualization.draw_geometries([pcd])
        pcds.append(pcd)

        with open(f"{in_path}/{i}.json") as f:
            data = json.load(f)
            Ri = data["R"]
            ti = data["t"]
            R[i, :] = np.array(Ri).reshape([3, 3])
            t[i, :] = ti

    # o3d.visualization.draw_geometries(pcds)
    if os.path.exists(f"{in_path}/icp_results.pkl"):
        with open(f"{in_path}/icp_results.pkl", 'rb') as handle:
            transforms = pickle.load(handle)
            for i, transform in enumerate(transforms):
                transform = np.linalg.inv(transform)
                R[i, :] = transform[0:3, 0:3]
                t[i, :] = transform[0:3, 3]
    else:
        R[0, :] = np.identity(3)
        t[0, :] = 0

        voxel_size = 0.1  # 5 cm voxel size
        tgt_down, tgt_fpfh = preprocess_point_cloud(pcds[0], voxel_size)
        icp_results = [np.identity(4)]
        for i in range(1, 4):
            # TODO: Is this ICP doing what we want it to?
            print("Apply coarse alignment")
            src_down, src_fpfh = preprocess_point_cloud(pcds[i], voxel_size)
            result_fast = execute_fast_global_registration(src_down, tgt_down,
                                                           src_fpfh, tgt_fpfh,
                                                           voxel_size)
            print(result_fast)
            transform = result_fast.transformation
            if True:
                print("Apply point-to-point ICP")
                threshold = 0.2  # 20cm
                # trans_init = np.identity(4)
                trans_init = transform
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    src_down, tgt_down, threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane())  # TransformationEstimationPointToPoint())
                # reg_p2p = o3d.pipelines.registration.registration_icp(
                #    pcds[i], pcds[0], threshold, trans_init,
                #    o3d.pipelines.registration.TransformationEstimationPointToPoint())
                transform = reg_p2p.transformation
                icp_results.append(transform)
            pcds[i].transform(transform)
            transform = np.linalg.inv(transform)
            #print(transform)
            R[i, :, :] = transform[0:3, 0:3]
            t[i, :] = transform[0:3, 3]
        print("dumping ICP results to source dataset")
        with open(f"{in_path}/icp_results.pkl", 'wb') as handle:
            pickle.dump(icp_results, handle, protocol=4)
        if visualize:
            o3d.visualization.draw_geometries(pcds)

    print(f"Writing hdf5 ({folder})")
    with h5py.File(f"{out_path}/frames.hdf5", "w") as f:
        f.create_dataset("im", data=im)
        f.create_dataset("imr", data=imr)
        f.create_dataset("grad", data=grad)
        f.create_dataset("ambient", data=ambient)
        f.create_dataset("disp", data=disp)
        f.create_dataset("R", data=R)
        f.create_dataset("t", data=t)
    print(f"Done writing hdf5 ({folder})")


for ind, folder in enumerate(folders):
    process(folder, ind, False)


# get only the first 9216
