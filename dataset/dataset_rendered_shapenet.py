import torch.utils.data as data
import numpy as np
import cv2
import os
import re
import math
from common.common import *


class DatasetRenderedShapenet(data.Dataset):


    def __init__(self, root_dir, type="train", noise=0.1, full_res=False, use_npy=True, debug=False):
        self.root_dir = root_dir
        self.use_npy = use_npy
        self.noise = noise
        self.focal = 567.6 * 2 # approximately
        self.baseline = -0.075
        self.debug = debug
        self.full_res = full_res
        if type == "train":
            self.from_idx = 1024
            self.to_idx = 1024*9
        if type == "val":
            self.from_idx = 0
            self.to_idx = 1024
        if type == "test":
            self.from_idx = 0
            self.to_idx = 1024

    def __len__(self):
        return self.to_idx - self.from_idx

    def __getitem__(self, idx):
        idx = idx + self.from_idx
        scene_idx = idx
        frame_idx = np.random.randint(0, 4)

        if self.use_npy:
            ir = np.load(f"{self.root_dir}/syn/{scene_idx:08d}/im0_{frame_idx}.npy")
            disp = np.load(f"{self.root_dir}/syn/{scene_idx:08d}/disp0_{frame_idx}.npy")
            mask = np.load(f"{self.root_dir}/syn/{scene_idx:08d}/mask0_{frame_idx}.npy")
        else:
            ir = cv2.imread(f"{self.root_dir}/syn/{scene_idx:08d}/im0_{frame_idx}.png", cv2.IMREAD_UNCHANGED)
            if self.full_res:
                disp = cv2.imread(f"{self.root_dir}/syn/{scene_idx:08d}/disp1_{frame_idx}.exr", cv2.IMREAD_UNCHANGED)
                assert ir is not None, f"File {self.root_dir}/syn/{scene_idx:08d}/im0_{frame_idx}.exr exists?"
                assert disp is not None, f"File {self.root_dir}/syn/{scene_idx:08d}/disp1_{frame_idx}.exr exists?"
                ir = np.expand_dims(ir[:960, :], 0).astype(np.float32) * (1.0/65536.0)
                disp = np.expand_dims(disp[:480, :], 0)
                mask = np.ones_like(disp)
            else:
                disp = cv2.imread(f"{self.root_dir}/syn/{scene_idx:08d}/disp0_{frame_idx}.exr", cv2.IMREAD_UNCHANGED)

        grey = ir
        grey += np.random.rand(grey.shape[0], grey.shape[1], grey.shape[2]) * np.random.rand() * self.noise
        grey *= np.random.uniform(0.5, 1.0)

        x_d = disp * math.copysign(1.0, self.baseline) + np.expand_dims(np.arange(0, disp.shape[2]), axis=(0, 1))
        x_d = x_d * (1.0 / disp.shape[2])

        depth = self.focal * self.baseline / disp[0, :, :]

        # Create Edge map by executing sobel on depth
        #  threshold
        grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        edge_threshold = 0.1  # a 10 centimeter threshold!!!!
        edges = (grad_x * grad_x + grad_y * grad_y) > edge_threshold * edge_threshold
        edges = edges.astype(np.float32)
        #  dilate
        edges = dilatation(edges, 10)
        edges = np.expand_dims(edges, 0)

        if self.debug:
            return grey, x_d, mask, edges, depth
        return grey, x_d, mask, edges
