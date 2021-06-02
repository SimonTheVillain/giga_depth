
import open3d as o3d
import numpy as np


def pointcloudify(mesh, supersample=1):
    indices = np.array(mesh.triangles)
    vertices = np.array(mesh.vertices)


if __name__ == "__main__":


    print("Testing IO for point cloud ...")
    pcd = o3d.io.read_point_cloud("../../TestData/fragment.pcd")
    print(pcd)
    o3d.io.write_point_cloud("copy_of_fragment.pcd", pcd)

    print("Testing IO for meshes ...")
    mesh = o3d.io.read_triangle_mesh("/media/simon/ext_ssd/datasets/structure_core/benchmark/drill/model.ply")
    print(mesh)
    print(np.array(mesh.vertices))
    print(np.array(mesh.triangles))
    np.array(mesh.vertices)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(mesh.vertices))
    o3d.io.write_point_cloud("/media/simon/ext_ssd/datasets/structure_core/benchmark/drill/model.pcd", pcd)
    exit(0)
    o3d.io.write_triangle_mesh("copy_of_knot.ply", np.array(mesh.vertices))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    print("Testing IO for textured meshes ...")
    textured_mesh = o3d.io.read_triangle_mesh("../../TestData/crate/crate.obj")
    print(textured_mesh)
    o3d.io.write_triangle_mesh("copy_of_crate.obj",
                               textured_mesh,
                               write_triangle_uvs=True)
    copy_textured_mesh = o3d.io.read_triangle_mesh('copy_of_crate.obj')
    print(copy_textured_mesh)

    print("Testing IO for images ...")
    img = o3d.io.read_image("../../TestData/lena_color.jpg")
    print(img)
    o3d.io.write_image("copy_of_lena_color.jpg", img)