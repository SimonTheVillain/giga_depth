import torch.utils.data as data
import numpy as np
import cv2
import os
import re
from common.common import *
from dataset.dataset_rendered_shapenet import DatasetRenderedShapenet



class DatasetRendered2(data.Dataset):

    def __init__(self, root_dir, start_ind, stop_ind, vertical_jitter=2, depth_threshold=15, noise=0.1,
                 tgt_res=(1216, 896),
                 tgt_cxy=(604, 457),
                 is_npy=False,
                 debug=False,
                 result_dir=""):
        self.is_npy = is_npy
        self.debug = debug
        self.result_dir = result_dir
        self.from_ind = start_ind
        self.to_ind = stop_ind
        self.root_dir = root_dir
        self.depth_threshold = depth_threshold
        self.noise = noise


        #these are the camera parameters applied to
        self.depth_threshold = 15
        self.src_res = (1401, 1001)
        self.src_cxy = (700, 500)
        self.tgt_res = tgt_res#(1216, 896)
        self.tgt_cxy = tgt_cxy#(604, 457)
        #the focal length is shared between src and target frame
        self.focal = 1.1154399414062500e+03

        self.readout_rect = (self.src_cxy[0]-self.tgt_cxy[0], self.src_cxy[1]-self.tgt_cxy[1],
                             self.tgt_res[0], self.tgt_res[1])
        self.vertical_jitter = vertical_jitter

        # the projector intrinsics are only needed to calculate depth
        # values are in the configuration of the unity rendering project
        self.focal_projector = 850 #todo: get real value!
        self.res_projector = 1024
        self.baselines = {"left": 0.0634 - 0.07501, "right": 0.0634 - 0.0}

    def __len__(self):
        if self.is_npy:
            return self.to_ind - self.from_ind
        # * 2 since we are working with stereo images
        return (self.to_ind - self.from_ind) * 2

    def __getitem__(self, idx):
        if self.is_npy:
            ir = np.load(f"{self.root_dir}/{idx}_ir.npy")
            x = np.load(f"{self.root_dir}/{idx}_gt.npy")
            mask = np.load(f"{self.root_dir}/{idx}_mask.npy")
            if self.result_dir == "":
                return ir, x, mask
            else:
                x_result = np.load(f"{self.result_dir}/{idx}.npy")
                return ir, x, mask, x_result

        side = "left"
        if idx % 2 == 1:
            side = "right"
        idx = int(idx / 2)

        v_offset = np.random.randint(-self.vertical_jitter, self.vertical_jitter)

        bgr = cv2.imread(self.root_dir + f"/{idx}_{side}.png")
        rr = self.readout_rect
        bgr = bgr[rr[1] + v_offset:rr[1] + v_offset + rr[3], rr[0]:rr[0]+rr[2], :].astype(np.float32) * (1.0/255.0)
        channel_weights = np.random.random(3) * 2
        channel_weights = channel_weights / (np.sum(channel_weights) + 0.01)
        channel_weights

        grey = bgr[:, :, 0] * channel_weights[0] + \
               bgr[:, :, 1] * channel_weights[1] + \
               bgr[:, :, 2] * channel_weights[2]
        grey += np.random.rand(grey.shape[0], grey.shape[1]) * np.random.rand() * self.noise
        grey = grey.astype(np.float32)
        #todo: add noise and scaling of intensity here!!!!



        gt = cv2.imread(self.root_dir + f"/{idx}_{side}_gt.exr", cv2.IMREAD_UNCHANGED)

        # calculate the (normalized) pixel coordinate in the pattern projector
        # since unity only allowes us to store float16, the offset had to be stored normalized +
        # subtracted from its x-position
        # TODO: delete this since it is hard to use this for subsampling (bilinear is wrong) max filtering as well
        # guiding by depth is not right either. (calculating depth from this & downsampling would be the way to go)
        x_gt = gt[:, :, 2] + np.expand_dims(np.arange(0, gt.shape[1]), axis=0) * (1.0 / float(gt.shape[1]))
        x_gt = x_gt[rr[1] + v_offset:rr[1] + v_offset + rr[3], rr[0]:rr[0]+rr[2]]
        # simple downsampling actually is not the way to go!!!!
        x_gt = x_gt.astype(np.float32)
        x_gt = downsample(x_gt)

        mask = np.logical_and(gt[:, :, 0] < self.depth_threshold, gt[:, :, 1] == 0)
        mask = mask[rr[1] + v_offset:rr[1] + v_offset + rr[3], rr[0]:rr[0]+rr[2]].astype(np.float32)

        # depth is used to generate the x-position(groundtruth) in the dot-projector.
        depth = gt[rr[1] + v_offset:rr[1] + v_offset + rr[3], rr[0]:rr[0]+rr[2], 0]
        # calculate x-poistion in real-world coordinates
        depth = downsampleDepth(depth) # half the resolution of depth taking the closest sample
        x_d = (np.arange(0, int(self.tgt_res[0]/2)) - self.tgt_cxy[0] * 0.5) * depth / (self.focal * 0.5)
        # for the right sensor we want to shift the points 6.34cm to the right
        # for the left sensor we want to shift the points approx 1.1cm to the left
        x_d += self.baselines[side]
        x_d = x_d * (self.focal_projector / depth) + float(self.res_projector - 1)/2.0
        x_d = x_d * (1.0/float(self.res_projector))
        x_d = x_d.astype(np.float32)

        #downsample the mask. (prioritize invalid pixel!!!)
        mask[mask == 0] = 2
        mask = downsampleDepth(mask)
        mask[mask == 2] = 0
        mask[np.logical_or(x_d < 0, x_d >= 1.0)] = 0

        #cv2.imshow("x_d", x_d)
        #cv2.imshow("diff", np.abs(x_d - x_gt) * 100.0)

        grey = np.expand_dims(grey, 0)
        x_d = np.expand_dims(x_d, 0)
        mask = np.expand_dims(mask, 0)
        if self.debug:
            return grey, x_d, mask, depth
        return grey, x_d, mask


class DatasetRendered3(data.Dataset):

    def __init__(self, root_dir, filenames,
                 vertical_jitter=2, depth_threshold=15, noise=0.1,
                 tgt_res=(1216, 896),
                 tgt_cxy=(604, 457),
                 debug=False,
                 result_dir=""):

        self.debug = debug
        self.result_dir = result_dir
        self.filenames = filenames
        self.root_dir = root_dir
        self.depth_threshold = depth_threshold
        self.noise = noise


        #these are the camera parameters applied to
        self.depth_threshold = 20
        self.src_res = (1401, 1001)
        self.src_cxy = (700, 500)
        self.tgt_res = tgt_res#(1216, 896)
        self.tgt_cxy = tgt_cxy#(604, 457)
        #the focal length is shared between src and target frame
        self.focal = 1.1154399414062500e+03

        self.readout_rect = (self.src_cxy[0]-self.tgt_cxy[0], self.src_cxy[1]-self.tgt_cxy[1],
                             self.tgt_res[0], self.tgt_res[1])
        self.vertical_jitter = vertical_jitter

        # the projector intrinsics are only needed to calculate depth
        # values are in the configuration of the unity rendering project
        self.focal_projector = 850 #todo: get real value!
        self.res_projector = 1024
        # compared to dataset v2 the baselines switched around in dataset v3.
        # as with the original camera the baseline between left sensor and emitter is bigger than the one
        # to the right sensor
        self.baselines = {"right": 0.07501 - 0.0634, "left": -0.0634}

    def __len__(self):
        # * 2 since we are working with stereo images
        return len(self.filenames) * 2

    def __getitem__(self, idx):
        side = "left"
        if idx % 2 == 1:
            side = "right"
        idx = int(idx / 2)

        file = self.filenames[idx]
        bgr = cv2.imread(f"{self.root_dir}/{file}_{side}.jpg")
        msk = cv2.imread(f"{self.root_dir}/{file}_{side}_msk.png", cv2.IMREAD_UNCHANGED)
        d = cv2.imread(f"{self.root_dir}/{file}_{side}_d.exr", cv2.IMREAD_UNCHANGED)
        v_offset = np.random.randint(-self.vertical_jitter, self.vertical_jitter)

        if msk is None:
            print(f"The msk file {file} is invalid nonetype. select random other!")
            return self.__getitem__(np.random.randint(0, len(self.filenames)) * 2 + idx %2)

        if d is None:
            print(f"the depth file {file} is invalid nonetype. selecting random other!")
            return self.__getitem__(np.random.randint(0, len(self.filenames)) * 2 + idx % 2)
        else:
            if np.any(np.isnan(d)) or np.any(np.isinf(d)):
                print(f"the depth file {file} is invalid (nan/inf). selecting random other!")
                return self.__getitem__(np.random.randint(0, len(self.filenames)) * 2 + idx % 2)


        rr = self.readout_rect
        bgr = bgr[rr[1] + v_offset:rr[1] + v_offset + rr[3], rr[0]:rr[0]+rr[2], :].astype(np.float32) * (1.0/255.0)
        channel_weights = np.random.random(3) * 2
        channel_weights = channel_weights / (np.sum(channel_weights) + 0.01)
        channel_weights

        grey = bgr[:, :, 0] * channel_weights[0] + \
               bgr[:, :, 1] * channel_weights[1] + \
               bgr[:, :, 2] * channel_weights[2]
        grey += np.random.rand(grey.shape[0], grey.shape[1]) * np.random.rand() * self.noise
        grey = grey.astype(np.float32)

        msk = msk / 255
        msk[d > self.depth_threshold] = 0
        msk = msk[rr[1] + v_offset:rr[1] + v_offset + rr[3], rr[0]:rr[0]+rr[2]].astype(np.float32)

        # depth is used to generate the x-position(groundtruth) in the dot-projector.
        depth = d[rr[1] + v_offset:rr[1] + v_offset + rr[3], rr[0]:rr[0]+rr[2]]
        # calculate x-poistion in real-world coordinates
        depth = downsampleDepth(depth) # half the resolution of depth taking the closest sample
        x_d = (np.arange(0, int(self.tgt_res[0]/2)) - self.tgt_cxy[0] * 0.5) * depth / (self.focal * 0.5)

        if np.any(np.isnan(x_d)):
            print("shit")
        if np.any(np.isnan(d)):
            print("shit")

        # for the right sensor we want to shift the points 6.34cm to the right
        # for the left sensor we want to shift the points approx 1.1cm to the left
        x_d += self.baselines[side]
        x_d = x_d * (self.focal_projector / depth) + float(self.res_projector - 1)/2.0
        x_d = x_d * (1.0/float(self.res_projector))
        x_d = x_d.astype(np.float32)

        #downsample the mask. (prioritize invalid pixel!!!)
        msk[msk == 0] = 2
        msk = downsampleDepth(msk)
        msk[msk == 2] = 0
        msk[np.logical_or(x_d < 0, x_d >= 1.0)] = 0

        #cv2.imshow("x_d", x_d)
        #cv2.imshow("diff", np.abs(x_d - x_gt) * 100.0)

        grey = np.expand_dims(grey, 0)
        x_d = np.expand_dims(x_d, 0)
        mask = np.expand_dims(msk, 0)

        # todo: sobel on depth
        #  threshold
        grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        edge_threshold = 0.1 # a 10 centimeter threshold!!!!
        edges = (grad_x * grad_x + grad_y * grad_y) > edge_threshold * edge_threshold
        edges = edges.astype(np.float32)
        #  dilate
        edges = dilatation(edges, 10)
        edges = np.expand_dims(edges, 0)

        if self.debug:
            return grey, x_d, mask, edges, depth
        return grey, x_d, mask, edges

class DatasetRendered4(data.Dataset):

    def __init__(self, root_dir, filenames,
                 vertical_jitter=1, depth_threshold=15, noise=0.1,
                 tgt_res=(1216, 896),
                 tgt_cxy=(604, 457),
                 focal=1.1154399414062500e+03,
                 debug=False,
                 result_dir=""):

        self.debug = debug
        self.result_dir = result_dir
        self.filenames = filenames
        self.root_dir = root_dir
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

        self.readout_rect = (self.src_cxy[0] - self.tgt_cxy[0], self.src_cxy[1] - self.tgt_cxy[1],
                             self.tgt_res[0], self.tgt_res[1])
        self.vertical_jitter = vertical_jitter

        # compared to dataset v2 the baselines switched around in dataset v3.
        # as with the original camera the baseline between left sensor and emitter is bigger than the one
        # to the right sensor
        self.baselines = {"right": 0.07501 - 0.0634, "left": -0.0634}

    def __len__(self):
        # * 2 since we are working with stereo images
        return len(self.filenames) * 2

    def __getitem__(self, idx):
        side = "left"
        if idx % 2 == 1:
            side = "right"
        idx = int(idx / 2)

        file = self.filenames[idx]
        bgr = cv2.imread(f"{self.root_dir}/{file}_{side}.jpg")
        msk = cv2.imread(f"{self.root_dir}/{file}_{side}_msk.png", cv2.IMREAD_UNCHANGED)
        d = cv2.imread(f"{self.root_dir}/{file}_{side}_d.exr", cv2.IMREAD_UNCHANGED)
        v_offset = np.random.randint(-self.vertical_jitter, self.vertical_jitter)

        if msk is None:
            print(f"The msk file {file} is invalid nonetype. select random other!")
            return self.__getitem__(np.random.randint(0, len(self.filenames)) * 2 + idx % 2)

        if d is None:
            print(f"the depth file {file} is invalid nonetype. selecting random other!")
            return self.__getitem__(np.random.randint(0, len(self.filenames)) * 2 + idx % 2)
        else:
            if np.any(np.isnan(d)) or np.any(np.isinf(d)):
                print(f"the depth file {file} is invalid (nan/inf). selecting random other!")
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


