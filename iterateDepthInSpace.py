import h5py
import numpy as np
import cv2
import pickle
import open3d as o3d


def generate_pcl(disp, focal, baseline, cxy, R, t):
    pts = []
    for i in range(disp.shape[0]):
        for j in range(disp.shape[1]):
            d = disp[i, j]
            if d > 0 and d < 1000:
                d = baseline * focal / d
                x = (j - cxy[0]) * d / (focal)
                y = (i - cxy[1]) * d / (focal)
                p = np.array([x, y, d])
                #this is how it is supposed to go:
                p = np.matmul(R.transpose(), p) - np.matmul(R.transpose(), t)
                #p = np.matmul(R, p) + t
                #p = p - t  #np.matmul(R.transpose(), p) + np.matmul(R.transpose(), t) (THIS NOT!)
                pts.append(p)

    pts = np.array(pts)
    # TODO: RENAME TO PCD
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


mode = "unity"  # unity or dis

if mode == "unity":
    base_path = "/media/simon/T7/datasets/structure_core_unity_sequences_DIS"
elif mode == "dis":
    base_path = "/media/simon/T7/datasets/DepthInSpace/rendered_default"  # TODO: this will be moved to the LaCie
else:
    print("not a valid data source")

settings_path = f'{base_path}/settings.pkl'
with open(str(settings_path), 'rb') as f:
    settings = pickle.load(f)
    K = settings["K"]
    focal = K[0, 0]  # assume only one focal length
    baseline = settings["baseline"]
    cxy = (K[0, 2], K[1, 2])  # settings[]
print(settings)

for ind in range(150):
    print(ind)
    f = h5py.File(f'{base_path}/{ind:08d}/frames.hdf5', 'r')
    l = list(f.keys())
    R = np.array(f.get("R"))  # ["R"]
    ambient = np.array(f["ambient"])
    disp = np.array(f["disp"])
    grad = np.array(f["grad"])
    im = np.array(f["im"])
    # sgm_disp = np.array(f["sgm_disp"]) # semiglobal matching disp?
    t = np.array(f["t"])
    print(im.shape)
    pcds = []
    for i in range(4):
        cv2.imshow(f"im_{i}", im[i, 0, :, :])
        cv2.imshow(f"disp_{i}", disp[i, 0, :, :] / 100)

        pcd = generate_pcl(disp[i, 0, :, :], focal, baseline, cxy, R[i, :], t[i, :])
        pcds.append(pcd)

    o3d.visualization.draw_geometries(pcds)
    cv2.waitKey()
