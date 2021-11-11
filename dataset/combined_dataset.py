import random

import torch
import torch.utils.data as data
import os
from pathlib import Path
import numpy as np
import cv2
from common.common import *

class DatasetCombined(data.Dataset):

    def __init__(self, path, type, vertical_jitter=3, depth_threshold=15, noise=0.3, backward_compability=True):
        path = Path(path)
        path_captured = path / "structure_core/sequences_combined_all"
        path_captured_gt = path / "structure_core/sequences_combined_all_GigaDepth66LCN_filled"
        path_unity = path / "structure_core_unity_sequences"

        self.backward_compability = backward_compability
        self.depth_th = depth_threshold
        self.noise = noise
        self.vertical_jitter = vertical_jitter
        sequences = os.listdir(path_captured_gt)
        sequences.sort()
        paths = [(path_captured / x, path_captured_gt / x)  for x in sequences if os.path.isdir(path_captured / x)]
        split = len(paths) - 16
        if type == "train":
            paths = paths[:split]
            #paths = paths[:100]# TODO: remove debug
            #paths=[] #TODO: remove! DEBUG: remove the captured part of the dataset
        if type == "val":
            paths = paths[split:]
        self.sequences_captured = paths

        sequences = os.listdir(path_unity)
        paths = [path_unity / x for x in sequences if os.path.isdir(path_unity / x)]
        paths.sort()
        split = len(paths) - 64
        if type == "train":
            paths = paths[:split]
            #paths = paths[:100]#TODO: remove debug
        if type == "val":
            paths = paths[split:]
        self.sequences_unity = paths

        self.baseline_stereo = 0.07501
        self.baseline_proj = 0.0634
        self.focal = 1.1154399414062500e+03

    def __len__(self):
        return len(self.sequences_unity) + len(self.sequences_captured) * 4

    def __getitem__(self, idx):
        src_res = (1401, 1001)
        src_cxy = (700, 500)
        tgt_res = (1216, 896)
        tgt_cxy = (604, 457)

        if idx < len(self.sequences_unity):
            has_gt = True
            frm = random.randint(0, 3)
            path_seq = self.sequences_unity[idx]

            v_offset = np.random.randint(-self.vertical_jitter, self.vertical_jitter)
            #readout rect on the right side
            rr = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1] + v_offset,
                            tgt_res[0], tgt_res[1])

            ir = cv2.imread(f"{path_seq}/{frm}_left.png", cv2.IMREAD_UNCHANGED)
            ir = ir[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2], :].astype(np.float32) * 1.0 / 255.0
            irr = cv2.imread(f"{path_seq}/{frm}_right.png", cv2.IMREAD_UNCHANGED)
            irr = irr[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2], :].astype(np.float32) * 1.0 / 255.0

            channel_weights = np.random.random(3) * 2
            channel_weights = channel_weights / (np.sum(channel_weights) + 0.01)
            ir = ir[:, :, 0] * channel_weights[0] + \
                 ir[:, :, 1] * channel_weights[1] + \
                 ir[:, :, 2] * channel_weights[2]

            irr = irr[:, :, 0] * channel_weights[0] + \
                  irr[:, :, 1] * channel_weights[1] + \
                  irr[:, :, 2] * channel_weights[2]

            ir += np.random.rand(ir.shape[0], ir.shape[1]) * np.random.rand() * self.noise

            gt = cv2.imread(f"{path_seq}/{frm}_left_gt.exr", cv2.IMREAD_UNCHANGED)
            depth = gt[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2], 0]
            msk = cv2.imread(f"{path_seq}/{frm}_left_msk.png", cv2.IMREAD_UNCHANGED)
            msk = msk[rr[1]:rr[1] + rr[3], rr[0]:rr[0] + rr[2]].astype(np.float32) * (1.0 / 255.0)

            msk[depth > self.depth_th] = 0
            depth = downsampleDepth(depth)

            grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
            edge_threshold = 0.1  # a 10 centimeter threshold!!!!
            edges = (grad_x * grad_x + grad_y * grad_y) > edge_threshold * edge_threshold
            edges = edges.astype(np.float32)
            #  dilate
            gt_edges = dilatation(edges, 10)

            disp = self.baseline_proj * (self.focal / 2) / depth
            x_d = -disp + np.expand_dims(np.arange(0, tgt_res[0] // 2), 0).astype(np.float32)
            x_d = x_d.astype(np.float32)
            x_d = x_d * (2.0 / float(tgt_res[0]))  # normalize between 0 and 1 (above and below are possbile too)
            gt_x_pos = x_d

            # downsample the mask. (prioritize invalid pixel!!!)
            msk[msk == 0] = 2
            msk = downsampleDepth(msk)
            msk[msk == 2] = 0
            gt_msk = msk

        else:
            has_gt = False
            idx = idx - len(self.sequences_unity)
            frm = idx % 4
            idx = idx // 4
            path_seq, path_gt_seq = self.sequences_captured[idx]

            ir = cv2.imread(f"{path_seq}/ir{frm}.png", cv2.IMREAD_UNCHANGED)
            irr = ir[:, : ir.shape[1] // 2].astype(np.float32) * (1.0 / 2.0**16)
            ir = ir[:, ir.shape[1] // 2:].astype(np.float32) * (1.0 / 2.0**16)

            disp = cv2.imread(f"{path_gt_seq}/{frm}.exr", cv2.IMREAD_UNCHANGED)
            disp = disp.astype(np.float32)
            gt_x_pos = -disp + np.expand_dims(np.arange(0, tgt_res[0] // 2), 0).astype(np.float32)

            gt_x_pos = gt_x_pos * (2.0 / float(tgt_res[0]))
            #gt_x_pos = np.zeros((ir.shape[0] // 2, ir.shape[1] // 2), dtype=np.float32)
            gt_msk = np.zeros_like(gt_x_pos)
            gt_msk[disp != 0] = 1.0
            gt_edges = np.zeros_like(gt_x_pos)

        ir = np.expand_dims(ir, 0)
        irr = np.expand_dims(irr, 0)
        gt_x_pos = np.expand_dims(gt_x_pos, 0)
        gt_msk = np.expand_dims(gt_msk, 0)
        gt_edges = np.expand_dims(gt_edges, 0)
        has_gt = np.array([[[float(has_gt)]]], dtype=np.float32)

        if self.backward_compability:
            return ir, gt_x_pos, gt_msk, gt_edges
        return ir, irr, gt_x_pos, gt_msk, gt_edges, has_gt


def test_dataset():
    path = "/home/simon/datasets"
    path = "/media/simon/ssd_datasets/datasets"


    dataset = DatasetCombined(path, "val", vertical_jitter=4, depth_threshold=15, noise=0.1, backward_compability=False)

    for i in range(64, len(dataset)):
        ir, irr, gt_x_pos, gt_msk, gt_edges, has_gt = dataset[i]
        cv2.imshow("ir", ir[0, :, :])
        cv2.imshow("irr", irr[0, :, :])
        cv2.imshow("gt_x_pos", gt_x_pos[0, :, :])
        cv2.imshow("gt_msk", gt_msk[0, :, :])
        cv2.imshow("gt_edges", gt_edges[0, :, :])
        cv2.waitKey()



if __name__ == "__main__":
    test_dataset()