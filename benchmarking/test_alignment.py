import copy
import os
import open3d as o3d
import numpy as np
import yaml


def pointcloudify(mesh, supersample=1):
    indices = np.array(mesh.triangles)
    vertices = np.array(mesh.vertices)

def show_alignment(model, pcl_path):
    pcd = o3d.io.read_point_cloud(pcl_path)

    pose_path = pcl_path[:-3] + "poses.yaml"
    if not os.path.exists(pose_path):
        return

    with open(pose_path) as file:
        poses = yaml.safe_load(file)
        pose = poses["children"][0]["pose"]
        print(pose)
        R = pose["orientation"]
        R = [R["w"], R["x"], R["y"], R["z"]]
        pos = pose["position"]
        pos = [pos["x"], pos["y"], pos["z"]]
        print(R)
        print(pos)

    mesh2 = copy.deepcopy(mesh)
    R = mesh2.get_rotation_matrix_from_quaternion(R)
    print(R)
    center = np.array([0, 0, 0], dtype=np.float)
    mesh2.rotate(R, center)
    mesh2.translate(pos)

    o3d.visualization.draw_geometries([pcd, mesh2])


if __name__ == "__main__":

    if False:
        root_path = "/media/simon/ext_ssd/datasets/structure_core/benchmark/"
        for scene in ["drill", "pipe1", "pipe2", "pipe3", "pipe4", "pipe6"]:
            mesh = o3d.io.read_triangle_mesh(os.path.join(root_path, f"{scene}/model.ply"))

            files = os.listdir(os.path.join(root_path, f"{scene}"))
            for file in files:
                if file[:2] == "ir":
                    filepath = os.path.join(root_path, f"{scene}/sgbm{file[2:-4]}.pcd")
                    show_alignment(mesh, filepath)


    root_path = "/media/simon/ext_ssd/datasets/mrf/raw_py"
    for scene in ["angel", "arch", "fox", "gargoyle", "lion"]:
        mesh = o3d.io.read_triangle_mesh(os.path.join(root_path, f"{scene}/model.ply"))
        mesh.scale(1.0/1000.0, (0, 0, 0))
        for i in range(1, 13):
            filepath = os.path.join(root_path, f"{scene}/raw{i:02d}.pcd")
            show_alignment(mesh, filepath)
