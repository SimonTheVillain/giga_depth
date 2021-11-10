import open3d as o3d
import os
import numpy as np
import cv2
import copy

def generate_pcd(disp, noproj=False):
    focal = 1.1154399414062500e+03
    baseline = 0.0634
    cxy = (604, 457)
    pts = []
    for i in range(disp.shape[0]):
        for j in range(disp.shape[1]):
            d = disp[i, j]
            if d > 3 and d < 50:
                d = baseline * focal * 0.5 / d
                z = d
                if noproj:
                    d = 3.0 # this might be helpful to remove outliers
                x = (j - cxy[0] * 0.5) * d / (focal * 0.5)
                y = (i - cxy[1] * 0.5) * d / (focal * 0.5)
                pts.append([x, y, z])

    pts = np.array(pts)
    # TODO: RENAME TO PCD
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


# http://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_outlier_removal.html
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

def generate_disp(pcd, half_res=True):
    focal = 1.1154399414062500e+03
    baseline = 0.0634
    cxy = (604, 457)
    res = (896, 1216)
    if half_res:
        cxy = (604.0 * 0.5, 457.0 * 0.5)
        res = (896 // 2, 1216 // 2)
        focal = focal * 0.5

    depth = np.ones(res, dtype=np.float32) * 100.0
    count = np.zeros_like(depth)
    th_rate = 0.01# merge pixel that have close pixel
    for pt in pcd.points:
        d = pt[2]
        if d == 0.0:
            continue
        xy = [pt[0] * focal / d + cxy[0], pt[1] * focal / d + cxy[1]]
        xy = [int(xy[0] + 0.5), int(xy[1] + 0.5)]
        th = th_rate*d
        if 0 <= xy[0] < depth.shape[1] and 0 <= xy[1] < depth.shape[0]:
            depth_old = depth[xy[1], xy[0]]
            if d > 0:
                if d < (depth_old - th):
                    depth[xy[1], xy[0]] = d
                    count[xy[1], xy[0]] = 1
                else:
                    if (depth_old - th) < d < (depth_old + th):
                        depth[xy[1], xy[0]] = (depth_old * count[xy[1], xy[0]] + d) / (count[xy[1], xy[0]] + 1)
                        count[xy[1], xy[0]] += 1

    depth[depth == 100.0] = 0

    disp = baseline * focal / np.clip(depth, a_min=0.1, a_max=400)
    disp[depth == 0] = 0
    #cv2.imshow("depth", depth * 0.1)
    #cv2.imshow("disp", disp * 0.01)
    #cv2.waitKey()
    return disp


#http://www.open3d.org/docs/latest/tutorial/Advanced/global_registration.html#Fast-global-registration
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
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

pth_in = "/media/simon/ssd_datasets/datasets/" \
       "structure_core/sequences_combined_all_GigaDepth66LCN"
pth_out = "/media/simon/ssd_datasets/datasets/" \
          "structure_core/sequences_combined_all_GigaDepth66LCN_filled"

print(pth_in)

print(pth_out)
for k in range(967):

    pcds = []
    disps = []
    for i in range(4):

        print(i)
        pth = f"{pth_in}/{k:03}/{i}.exr"
        disp = cv2.imread(pth, cv2.IMREAD_UNCHANGED)
        disps.append(disp)
        #cv2.imshow("disp", disp)
        #cv2.waitKey()
        pcd_noproj = generate_pcd(disp, True)
        pcd = generate_pcd(disp, False)

        #cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
        #                                                    std_ratio=2.0)
        #TODO: this is proper filtering, apply the indices to a real pointcloud, ICP, reproject and store the results!!!!
        cl, ind = pcd_noproj.remove_radius_outlier(nb_points=50, radius=0.05)
        pcd = pcd.select_by_index(ind)
        #o3d.visualization.draw_geometries([pcd])
        #display_inlier_outlier(pcd_noproj, ind)
        pcds.append(pcd)
        #o3d.visualization.draw_geometries([pcd])


    #o3d.visualization.draw_geometries(pcds)

    #generate_disp(pcds[0])
    transforms = [np.identity(4)]
    uber_pcd = copy.deepcopy(pcds[0])
    min_fitness = 1.0

    voxel_size = 0.05
    for i in range(1, 4):
        source_down, source_fpfh = preprocess_point_cloud(pcds[i], voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(uber_pcd, voxel_size)

        result_fast = execute_fast_global_registration(source_down, target_down,
                                                       source_fpfh, target_fpfh,
                                                       voxel_size)

        #draw_registration_result(pcds[i], uber_pcd, result_fast.transformation)

        threshold = 0.02  # 2cm
        registration = o3d.pipelines.registration.registration_icp(
            pcds[i], uber_pcd, threshold, result_fast.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

        threshold = 0.04# 4cm
        evaluation = o3d.pipelines.registration.evaluate_registration(
            uber_pcd, pcds[i],
            threshold,
            np.linalg.inv(registration.transformation))

        print(f"fitness = {evaluation.fitness}")
        min_fitness = min(min_fitness, evaluation.fitness)
        pcds[i].transform(registration.transformation)
        uber_pcd += pcds[i]
        transforms.append(registration.transformation)

        #o3d.visualization.draw_geometries(pcds[0:i])

    if False:
        for i in range(1, 4):
            print("Apply coarse point-to-point ICP")
            threshold = 1.0  # 1 meter!?
            trans_init = np.identity(4)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcds[i], uber_pcd, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            old_transform = reg_p2p.transformation
            print("Apply FINE point-to-point ICP")
            threshold = 0.02  # 2cm
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcds[i], uber_pcd, threshold, reg_p2p.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            pcds[i].transform(reg_p2p.transformation)
            print(np.dot(np.linalg.inv(old_transform), reg_p2p.transformation))

            evaluation = o3d.pipelines.registration.evaluate_registration(
                uber_pcd, pcds[i], threshold, trans_init)
            print(evaluation.fitness)

            uber_pcd += pcds[i]
            o3d.visualization.draw_geometries(pcds[0:i])
            transforms.append(reg_p2p.transformation)

        #o3d.visualization.draw_geometries(pcds)

    if min_fitness < 0.5:
        continue

    pth = f"{pth_out}/{k:03}"
    if not os.path.exists(pth):
        os.mkdir(pth)

    #cl, ind = uber_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    cl = uber_pcd
    for i in range(4):
        #pcd_out = cl.transform(transforms[i])#
        pcd_out = copy.deepcopy(cl).transform(np.linalg.inv(transforms[i]))
        disp = generate_disp(pcd_out)
        cv2.imshow(f"disp{i}", disps[i] / 100.0)
        cv2.imshow(f"disp{i}_corrected", disp / 100.0)
        cv2.waitKey(100)
        pth = f"{pth_out}/{k:03}/{i}.exr"
        cv2.imwrite(pth, disp)
#final_pcd = generate_pcd(disp)
#o3d.visualization.draw_geometries([final_pcd])
#display_inlier_outlier(uber_pcd, ind)


#o3d.visualization.draw_geometries([cl])




#todo: merge point
#todo: backproject transformations
#sequences = os.ldir(pth_in)
#for(sequences)