import h5py
import numpy as np
import cv2
import pickle
import yaml
import json
import os
import os.path
from pathlib import Path
from common.common import downsampleDisp, downsample
import open3d as o3d
import multiprocessing



with open("configs/paths-local.yaml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)

in_root_path = str(Path("/media/simon/sandisk/datasets/structure_core_unity_sequences_3"))
out_root_path = str(Path("/media/simon/T7/datasets/structure_core_unity_sequences_DIS"))

src_cxy = (700, 500)
tgt_res = (1216, 896)
tgt_cxy = (604, 457)
# the focal length is shared between src and target frame
focal = 1.1154399414062500e+03
baseline = 0.0634

rr = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1], tgt_res[0], tgt_res[1])
f = 0.5 * focal
cx = 0.5 * tgt_cxy[0]
cy = 0.5 * tgt_cxy[1]
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
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

cv2.imshow("pattern", pattern)
cv2.waitKey(1)
settings = {
    "baseline": baseline,
    "K": K,
    "imsize": imsize,
    "pattern": pattern
}
with open(f"{out_root_path}/settings.pkl", 'wb') as handle:
    pickle.dump(settings, handle, protocol=pickle.HIGHEST_PROTOCOL)

files = os.listdir(in_root_path)
files.sort()
folders = files
#folders = filter(lambda file: os.path.isdir(f"{in_path}/{files}"), files)
#folders = folders[:9216]


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

def process(folder, visualize):
    print(folder)
    in_path = f"{in_root_path}/{folder}"
    folder_ind = int(folder)
    out_path = f"{out_root_path}/{folder_ind:08d}"
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    im = np.zeros([4, 1, 448, 608])
    grad = np.zeros([4, 1, 448, 608])
    ambient = np.zeros([4, 1, 448, 608])
    disp = np.zeros([4, 1, 448, 608])
    R = np.zeros([4, 3, 3])
    t = np.zeros([4, 3])
    pcds = []
    for i in range(4):
        #print(f"{folder}_{i}")
        imi = cv2.imread(f"{in_path}/{i}_left.png")
        imi = cv2.cvtColor(imi, cv2.COLOR_BGR2GRAY)
        imi = imi[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]] / 255.0
        imi = downsample(imi)
        im[i, 0, :, :] = imi

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
        pickle.dump(icp_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if visualize:
        o3d.visualization.draw_geometries(pcds)

    print(f"Writing hdf5 ({folder})")
    with h5py.File(f"{out_path}/frames.hdf5", "w") as f:
        f.create_dataset("im", data=im)
        f.create_dataset("grad", data=grad)
        f.create_dataset("ambient", data=ambient)
        f.create_dataset("disp", data=disp)
        f.create_dataset("R", data=R)
        f.create_dataset("t", data=t)


multithreaded = True

if multithreaded:
    pool = multiprocessing.Pool(12)
    pool.starmap(process, zip(folders, [False] * len(folders)))
else:
    for folder in folders:
        process(folder, True)


# get only the first 9216
