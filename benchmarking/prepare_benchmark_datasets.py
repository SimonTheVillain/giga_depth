
import open3d as o3d
import numpy as np
import os
import cv2


def pointcloudify(mesh, supersample=1):
    indices = np.array(mesh.triangles)
    vertices = np.array(mesh.vertices)

def pointcloudify_file(path, scale_factor=1.0, supersample=1):

    print(f"loading pointcloud {path}")
    mesh = o3d.io.read_triangle_mesh(path)

    vertices = np.array(mesh.vertices) * scale_factor
    triangles = np.array(mesh.triangles)
    print(triangles)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    path = path[:-3] + "pcd"
    print(f"storing pointcloud {path}")
    o3d.io.write_point_cloud(path, pcd)


def depth_to_pcd(z, f, c, offset, half_res=True, max_dist=4):
    fx = f[0]
    cxr = c[0]
    cyr = c[1]
    if half_res:
        fx = fx * 0.5
        cxr = cxr * 0.5
        cyr = cyr * 0.5
    print(z.shape)
    pts = []
    for i in range(0, z.shape[0]):
        for j in range(0, z.shape[1]):
            y = i + offset - cyr
            x = j - cxr
            depth = z[i, j]
            if 0 < depth < max_dist:
                pts.append([x*depth/fx, y*depth/fx, depth])
    xyz = np.array(pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def pointcloudify_structure_stereo(src, dst):
    f = (1.1154399414062500e+03, 1.1154399414062500e+03)
    c = (604, 457)
    bl = 0.07501
    ir = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    ir = (ir / 256).astype(np.ubyte)
    irl = ir[:, int(ir.shape[1] / 2):]
    irr = ir[:, :int(ir.shape[1] / 2)]

    # irr[:, :-100] = irl[:, 100:]
    # cv2.imshow("irl", irl)
    # cv2.imshow("irr", irr)
    # cv2.waitKey()
    block_size = 5
    P1 = block_size ** 2 * 8
    P2 = block_size ** 2 * 32
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=256+128, blockSize=block_size, P1=P1, P2=P2)
    # stereo = cv2.StereoBM_create(numDisparities=256, blockSize=15)
    disparity = stereo.compute(irl, irr)
    disparity[disparity == 0] = -16
    disparity = disparity.astype(np.float32) * (1.0 / 16.0)  # who ever said this is factor 16?
    # print(np.max(disparity))
    z = bl * f[0] / disparity.astype(np.float32)
    pcd = depth_to_pcd(z, f, c, 0, False, max_dist=3)
    o3d.io.write_point_cloud(dst, pcd)


if __name__ == "__main__":
    if True:
        root_path = "/media/simon/ext_ssd/datasets/structure_core/benchmark/"
        for scene in ["drill", "pipe1", "pipe2", "pipe3", "pipe4", "pipe6"]:
            pointcloudify_file(os.path.join(root_path, f"{scene}/model.ply")) #todo: actually use mesh.sample_points_uniformly(number_of_points=500)
            files = os.listdir(os.path.join(root_path, f"{scene}"))
            for file in files:
                if file[:2] == "ir":
                    src = os.path.join(root_path, f"{scene}/{file}")
                    dst = os.path.join(root_path, f"{scene}/sgbm{file[2:-4]}.pcd")
                    pointcloudify_structure_stereo(src, dst)


    #it is expected to first run convert_mrf.py & also put the model files in the folders of each object
    bl = 0.075 # 75mm baseline
    f = (578, 578)
    c = (640/2, 480/2)# TODO: FIND OUT THIS!!!!
    root_path = "/media/simon/ext_ssd/datasets/mrf/raw_py"
    for scene in ["angel", "arch", "fox", "gargoyle", "lion"]:
        pointcloudify_file(os.path.join(root_path, f"{scene}/model.ply"), scale_factor=(1.0 / 1000.0))

        for i in range(1, 13):
            path = os.path.join(root_path, f"{scene}/disp{i:02d}.exr")
            disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            depth = bl * f[0] / disp
            pcd = depth_to_pcd(depth, f, c, offset=0, half_res=False, max_dist=2)
            path = os.path.join(root_path, f"{scene}/raw{i:02d}.pcd")
            o3d.io.write_point_cloud(path, pcd)






