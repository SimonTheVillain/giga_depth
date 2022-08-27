import torch.utils.data as data
import numpy as np
import cv2
import os
import re
from common.common import *
from dataset.dataset_rendered_shapenet import DatasetRenderedShapenet
from pathlib import Path



class DatasetRenderedSequences(data.Dataset):

    def __init__(self, sequence_dirs,
                 vertical_jitter=3, depth_threshold=15, noise=0.1,
                 tgt_res=(1216, 896),
                 tgt_cxy=(604, 457),
                 focal=1.1154399414062500e+03,
                 use_all_frames=False,
                 left_only=False,
                 debug=False):

        self.sequence_dirs = sequence_dirs
        self.depth_threshold = depth_threshold
        self.noise = noise

        # these are the camera parameters applied to
        self.depth_threshold = 20
        self.src_res = (1401, 1001)
        self.src_cxy = (700, 500)
        self.tgt_res = tgt_res  # (1216, 896)
        self.tgt_cxy = tgt_cxy  # (604, 457)
        # the focal length is shared between src and target frame
        self.focal = focal
        self.use_all_frames = use_all_frames
        self.left_only = left_only

        self.readout_rect = (self.src_cxy[0] - self.tgt_cxy[0], self.src_cxy[1] - self.tgt_cxy[1],
                             self.tgt_res[0], self.tgt_res[1])
        self.vertical_jitter = vertical_jitter
        self.debug = debug

        # compared to dataset v2 the baselines switched around in dataset v3.
        # as with the original camera the baseline between left sensor and emitter is bigger than the one
        # to the right sensor
        self.baselines = {"right": 0.07501 - 0.0634, "left": -0.0634}

    def __len__(self):
        count = len(self.sequence_dirs)
        if not self.left_only:
            count *= 2
        if self.use_all_frames:
            count *= 4
        # * 2 since we are working with stereo images
        return count

    def __getitem__(self, idx):
        # index within sequence:
        if self.use_all_frames:
            side = "left"
            if self.left_only:
                idx = idx
                frm = idx % 4
                idx = idx // 4
            else:
                if idx % 2 == 1:
                    side = "right"
                idx = idx // 2
                frm = idx % 4
                idx = idx // 4
        else:
            side = "left"
            if not self.left_only:
                if idx % 2 == 1:
                    side = "right"
                idx = int(idx / 2)


            #frame index:
            frm = np.random.randint(0, 3)


        sequence = self.sequence_dirs[idx]
        suffix = ""
        if os.path.isfile(f"{sequence}/{frm}_{side}.png"):
            suffix = "png"
        if os.path.isfile(f"{sequence}/{frm}_{side}.jpg"):
            suffix = "jpg"
        if suffix == "":
            print(f"No compatible image file found {sequence}/{frm}_{side}.(png/jpg")

        bgr = cv2.imread(f"{sequence}/{frm}_{side}.{suffix}")
        gt = cv2.imread(f"{sequence}/{frm}_{side}_gt.exr", cv2.IMREAD_UNCHANGED)

        if os.path.isfile(f"{sequence}/{frm}_{side}_msk.{suffix}"):
            #load an existing mask
            msk = cv2.imread(f"{sequence}/{frm}_{side}_msk.{suffix}", cv2.IMREAD_UNCHANGED)
        else:
            #create a mask
            ir_no = cv2.imread(f"{sequence}/{frm}_{side}_noproj.exr", cv2.IMREAD_UNCHANGED)
            ir_msk = cv2.imread(f"{sequence}/{frm}_{side}_msk.exr", cv2.IMREAD_UNCHANGED)
            msk = np.zeros((ir_no.shape[0], ir_no.shape[1]), dtype=np.ubyte)
            th = 0.0001
            delta = np.abs(ir_no - ir_msk)
            msk[(delta[:, :, 0] + delta[:, :, 1] + delta[:, :, 2]) > th] = 255
            msk[gt[:, :, 1] > 0] = 0

        #msk = cv2.imread(f"{sequence}/{frm}_{side}_msk.png", cv2.IMREAD_UNCHANGED)
        d = gt[:, :, 0]
        #d = cv2.imread(f"{sequence}/{frm}_{side}_d.exr", cv2.IMREAD_UNCHANGED)
        v_offset = np.random.randint(-self.vertical_jitter, self.vertical_jitter)


        if d is None:
            print(f"the depth file {sequence} is invalid nonetype. selecting random other!")
            return self.__getitem__(np.random.randint(0, len(self.filenames)) * 2 + idx % 2)
        else:
            if np.any(np.isnan(d)) or np.any(np.isinf(d)):
                print(f"the depth file {sequence} is invalid (nan/inf). selecting random other!")
                return self.__getitem__(np.random.randint(0, len(self.filenames)) * 2 + idx % 2)

        rr = self.readout_rect
        bgr = bgr[rr[1] + v_offset:rr[1] + v_offset + rr[3], rr[0]:rr[0] + rr[2], :].astype(np.float32) * (
                    1.0 / 255.0)
        channel_weights = np.random.random(3) * 2
        channel_weights = channel_weights / (np.sum(channel_weights) + 0.01)

        grey = bgr[:, :, 0] * channel_weights[0] + \
               bgr[:, :, 1] * channel_weights[1] + \
               bgr[:, :, 2] * channel_weights[2]
        grey += np.random.rand(grey.shape[0], grey.shape[1]) * np.random.rand() * self.noise
        grey = grey.astype(np.float32)

        msk = msk / 255
        msk[d > self.depth_threshold] = 0
        msk = msk[rr[1] + v_offset:rr[1] + v_offset + rr[3], rr[0]:rr[0] + rr[2]].astype(np.float32)

        # depth is used to generate the x-position(groundtruth) in the dot-projector.
        depth = d[rr[1] + v_offset:rr[1] + v_offset + rr[3], rr[0]:rr[0] + rr[2]]
        # calculate x-poistion in real-world coordinates
        depth = downsampleDepth(depth)  # half the resolution of depth taking the closest sample
        x_d = (np.arange(0, int(self.tgt_res[0] / 2)) - self.tgt_cxy[0] * 0.5) * depth / (self.focal * 0.5)

        if np.any(np.isnan(x_d)):
            print("shit")
        if np.any(np.isnan(d)):
            print("shit")

        # for the right sensor we want to shift the points 6.34cm to the right
        # for the left sensor we want to shift the points approx 1.1cm to the left
        disp = self.baselines[side] * (self.focal / 2) / depth
        x_d = disp + np.expand_dims(np.arange(0, int(self.tgt_res[0] / 2)), 0).astype(np.float32)
        x_d = x_d.astype(np.float32)
        x_d = x_d * (2.0/float(self.tgt_res[0])) #normalize between 0 and 1 (above and below are not impossible

        # downsample the mask. (prioritize invalid pixel!!!)
        msk[msk == 0] = 2
        msk = downsampleDepth(msk)
        msk[msk == 2] = 0

        # cv2.imshow("x_d", x_d)
        # cv2.imshow("diff", np.abs(x_d - x_gt) * 100.0)

        grey = np.expand_dims(grey, 0)
        x_d = np.expand_dims(x_d, 0)
        mask = np.expand_dims(msk, 0)

        # Create Edge map by executing sobel on depth
        #  threshold
        depth_1 = 1.0/depth
        depth_1[np.isnan(depth_1)] = 0
        grad_x = cv2.Sobel(depth_1, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_1, cv2.CV_32F, 0, 1, ksize=3)
        edge_threshold = 0.1  # a 10 centimeter threshold!!!!
        edges = (grad_x * grad_x + grad_y * grad_y) > edge_threshold * edge_threshold
        edges = edges.astype(np.float32)
        #  dilate
        edges = dilatation(edges, 10)
        edges = np.expand_dims(edges, 0)

        if self.debug:
            return grey, x_d, mask, edges, depth
        return grey, x_d, mask, edges


def add_msk():
    dataset_paths = ["/media/simon/ssd_datasets/datasets/structure_core_unity_sequences"]
    paths = []
    for dataset_path in dataset_paths:
        print(dataset_path)
        folders = os.listdir(dataset_path)
        folders.sort()
        folders = [Path(dataset_path) / x for x in folders if os.path.isdir(Path(dataset_path) / x)]
        paths += folders

    lcn_module = LCN()
    for idx, sequence in enumerate(paths):
        print(f"{idx/len(paths)}")
        for side in ["left", "right"]:
            for i in range(4):
                # create a mask
                gt = cv2.imread(f"{sequence}/{i}_{side}_gt.exr", cv2.IMREAD_UNCHANGED)
                ir = cv2.imread(f"{sequence}/{i}_{side}.png", cv2.IMREAD_UNCHANGED)
                ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY).astype(np.float32) * 1.0/255.0
                lcnp = LCN_np(ir)
                ir = torch.tensor(ir).unsqueeze(0).unsqueeze(0)
                lcn, mean, std = LCN_tensors(ir)
                lcn2 = lcn_module(ir)
                lcn = lcn * 0.5 + 0.5
                cv2.imshow("ir", ir[0,0,:,:].cpu().numpy())
                cv2.imshow("lcn", lcn[0,0,:,:].cpu().numpy())
                cv2.imshow("mean", mean[0,0,:,:].cpu().numpy())
                cv2.imshow("std", std[0,0,:,:].cpu().numpy())
                cv2.imshow("lcn2", lcn2[0,0,:,:].cpu().numpy())
                cv2.imshow("lcnp", lcnp)
                cv2.imshow("gt",gt)
                cv2.waitKey()

#def run_trough_dataset(path):
#    DatasetRenderedSequences(path)

if __name__ == "__main__":

    add_msk()
