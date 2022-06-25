import h5py
import numpy as np
import cv2
import pickle
import open3d as o3d
import multiprocessing
import os


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
    pass
    #base_path = "/media/simon/sandisk/datasets/DepthInSpace/rendered_default"  # TODO: this will be moved to the LaCie
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


def convert(ind):
    print(ind)
    pth = f'{base_path}/{ind:08d}/frames.hdf5'
    f = h5py.File(pth, 'r')
    l = list(f.keys())
    im = np.array(f["im"]).astype(np.float32)
    grad = np.array(f["grad"]).astype(np.float32)
    ambient = np.array(f["ambient"]).astype(np.float32)
    disp = np.array(f["disp"]).astype(np.float32)
    R = np.array(f.get("R")).astype(np.float32)
    t = np.array(f["t"]).astype(np.float32)
    f.close()
    print(f"Writing hdf5 ({ind})")
    with h5py.File(pth, "w") as f:
        f.create_dataset("im", data=im)
        f.create_dataset("grad", data=grad)
        f.create_dataset("ambient", data=ambient)
        f.create_dataset("disp", data=disp)
        f.create_dataset("R", data=R)
        f.create_dataset("t", data=t)
    print(f"Done writing hdf5 ({ind})")


inds = list(range(0, 15300))
inds_delete = []
inds_reduce = []
for ind in inds:
    print(ind)
    pth = f'{base_path}/{ind:08d}/frames.hdf5'
    sz = os.path.getsize(pth)
    sz /= (1024 * 1024)
    if sz < 13:
        inds_delete.append(ind)

    with h5py.File(pth, 'r') as f:
        if len(f.keys()) != 6:
            inds_delete.append(ind)
        if np.array(f["t"]).dtype == np.float64:
            inds_reduce.append(ind)


print(f"Nr paths to delete: {len(inds_delete)}")
for ind in inds_delete:
    pth = f'{base_path}/{ind:08d}/frames.hdf5'
    os.remove(pth)

print(f"Nr paths to reduce: {len(inds_reduce)}")

threaded = False
if threaded:
    pool = multiprocessing.Pool(12)
    pool.map(convert, inds_reduce)
else:
    for ind in inds_reduce:
        convert(ind)
