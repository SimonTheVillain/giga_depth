import copy

import open3d as o3d
import numpy as np
import yaml


def pointcloudify(mesh, supersample=1):
    indices = np.array(mesh.triangles)
    vertices = np.array(mesh.vertices)


if __name__ == "__main__":

    with open("/media/simon/ext_ssd/datasets/structure_core/benchmark/drill/poses.yaml") as file:
        poses = yaml.safe_load(file)
        pose = poses["children"][0]["pose"]
        print(pose)
        R = pose["orientation"]
        R = [R["w"], R["x"], R["y"], R["z"]]
        pos = pose["position"]
        pos = [pos["x"], pos["y"], pos["z"]]
        print(R)
        print(pos)

    pcd = o3d.io.read_point_cloud("/media/simon/ext_ssd/datasets/structure_core/benchmark/drill/ir000000.pcd")

    mesh = o3d.io.read_triangle_mesh("/media/simon/ext_ssd/datasets/structure_core/benchmark/drill/model.ply")

    mesh2 = copy.deepcopy(mesh)
    R = mesh2.get_rotation_matrix_from_quaternion(R)
    print(R)
    center = np.array([0, 0, 0], dtype=np.float)
    mesh2.rotate(R, center)
    mesh2.translate(pos)

    o3d.visualization.draw_geometries([pcd, mesh, mesh2])
