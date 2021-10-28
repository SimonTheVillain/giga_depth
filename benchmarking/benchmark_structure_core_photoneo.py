import open3d as o3d
import numpy as np
import cv2

gt_path = "/home/simon/datasets/structure_core_photoneo_test/000"


focal = 1.1154399414062500e+03,
cxy=(604, 457)
depth = cv2.imread(gt_path + "/depth0.png", cv2.IMREAD_UNCHANGED)
pts = []
for i in range(depth.shape[0]):
    for j in range(depth.shape[1]):
        d = depth[i, j]
        d = d / 1000.0
        if d > 0.1 and d < 10:
            # assume the depth is in millimeters:
            x = (j - cxy[0]) * d / focal
            y = (i - cxy[1]) * d / focal
            pts.append([x, y, d])

pts = np.array(pts)
pcd_base = o3d.geometry.PointCloud()
pcd_base.points = o3d.utility.Vector3dVector(pts)


print("Load groundtruth ply")
pcd = o3d.io.read_point_cloud(gt_path + "/gt.ply")
pcd = pcd.scale(1.0/1000.0, np.zeros((3, 1)))


if True:

    print("Apply point-to-point ICP")
    threshold = 0.02
    threshold = 0.2
    trans_init = np.identity(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_base, pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)

    pcd_base.transform(reg_p2p.transformation)

    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        pcd_base, pcd, threshold, trans_init)
    print(evaluation)

#print(pcd)
#print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd, pcd_base])