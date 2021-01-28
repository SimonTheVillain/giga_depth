import numpy as np
from dataset_rendered_2 import DatasetRendered2
import os

src_dir = os.path.expanduser("~/datasets/structure_core_unity")
dst_dir = os.path.expanduser("~/datasets/structure_core_unity_slice_100_35_2")

src_dir = os.path.expanduser("/media/simon/ssd_data/data/datasets/structure_core_unity")
dst_dir = os.path.expanduser("/media/simon/ssd_data/data/datasets/structure_core_unity_slice_100_90")

dataset = DatasetRendered2(src_dir, 0, 80000)

slice_in = (100, 100 + 17 * 2 + 56)
slice_gt = (50 + 8, 50 + 8 + 28)

for i, data in enumerate(dataset):
    print(i)
    ir, gt, mask = data

    ir = ir[:, slice_in[0]:slice_in[1], :]
    gt = gt[:, slice_gt[0]:slice_gt[1], :]
    mask = mask[:, slice_gt[0]:slice_gt[1], :]
    np.save(f"{dst_dir}/{i}_ir.npy", ir)
    np.save(f"{dst_dir}/{i}_gt.npy", gt)
    np.save(f"{dst_dir}/{i}_mask.npy", mask)


