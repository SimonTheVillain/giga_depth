import os
import cv2
import numpy as np
from pathlib import Path

in_path = Path("/home/simon/datasets/structure_core/sequences_combined")
out_path = Path("/home/simon/datasets/structure_core/sequences_combined_SGBM")


#in_path = Path("/home/simon/datasets/structure_core/sequences_combined_ambient")
#out_path = Path("/home/simon/datasets/structure_core/sequences_combined_ambient_SGBM")

in_path = Path("/home/simon/datasets/structure_core/sequences_combined_all")
out_path = Path("/home/simon/datasets/structure_core/sequences_combined_all_SGBM")
baseline_from = 0.07501
baseline_to = 0.0634

block_size = 5
P1 = block_size ** 2 * 8
P2 = block_size ** 2 * 32

stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=block_size, P1=P1, P2=P2)
#stereo = cv2.StereoBM_create(numDisparities=256, blockSize=15)


scenes = os.listdir(in_path)
for scene in scenes:
    if not os.path.isdir(in_path / scene):
        continue
    if not os.path.exists(out_path / scene):
        os.makedirs(out_path / scene)

    for i in range(4):
        impath = in_path / scene / f"ir{i}.png"
        ir = cv2.imread(str(impath), cv2.IMREAD_UNCHANGED)
        ir = (ir / 256).astype(np.ubyte)
        irl = ir[:, int(ir.shape[1] / 2):]
        irr = ir[:, :int(ir.shape[1] / 2)]

        disparity = stereo.compute(irl, irr)
        disparity[disparity == 0] = -16
        disparity = disparity.astype(np.float32) * (1.0 / 16.0)
        #TODO: convert to float
        disparity = disparity * baseline_from / baseline_to
        impath = out_path / scene / f"disp{i}.exr"
        cv2.imwrite(str(impath), disparity)

        cv2.imshow("ir", irl)
        cv2.imshow("disp", disparity / 100)
        cv2.waitKey(10)
