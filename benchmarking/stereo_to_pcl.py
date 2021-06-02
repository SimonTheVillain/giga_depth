import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

#todo: do stereomatching on the stereo data of the structure core and create pointclouds from that

file1 = "/media/simon/ext_ssd/datasets/structure_core/benchmark/drill/single_shots/ir000000.png"

def display_pcl(pcd):
    o3d.visualization.draw_geometries([pcd])

def depth_to_pcd(z, f, c, offset, half_res=True):
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
            if 0 < depth < 4:
                pts.append([x*depth/fx, y*depth/fx, depth])
    xyz = np.array(pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd



f = (1.1154399414062500e+03, 1.1154399414062500e+03)
c = (604, 457)
bl = 0.07501

ir = cv2.imread(file1, cv2.IMREAD_UNCHANGED)
ir = (ir / 256).astype(np.ubyte)
irl = ir[:, int(ir.shape[1] / 2):]
irr = ir[:, :int(ir.shape[1] / 2)]

#irr[:, :-100] = irl[:, 100:]
#cv2.imshow("irl", irl)
#cv2.imshow("irr", irr)
#cv2.waitKey()
block_size = 5
P1 = block_size ** 2 * 8
P2 = block_size ** 2 * 32
stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=256, blockSize=block_size, P1=P1, P2=P2)
#stereo = cv2.StereoBM_create(numDisparities=256, blockSize=15)
disparity = stereo.compute(irl, irr)
disparity[disparity == 0] = -16
disparity = disparity.astype(np.float32) * (1.0/16.0)# who ever said this is factor 16?
#print(np.max(disparity))
z = bl * f[0] / disparity.astype(np.float32)
pcd = depth_to_pcd(z, f, c, 0, False)
o3d.io.write_point_cloud("/media/simon/ext_ssd/datasets/structure_core/benchmark/drill/single_shots/ir000000.pcd", pcd)
o3d.visualization.draw_geometries([pcd])

plt.imshow(disparity + irl.astype(np.float32)*0, 'gray')
#plt.imshow(z, 'gray')
plt.show()

