import torch
import torch.nn as nn
import torch.utils.data as data
from skimage import io, transform
import numpy as np
import numpy.matlib
from pathlib import Path
import cv2
import random

import matplotlib.pyplot as plt


def downsample(image):
    image = image.reshape(image.shape[0]/2,2,image.shape[1]/2,2)
    image = np.mean(image, axis=3)
    image = np.mean(image, axis=1)
    return image

def downsampleDepth(d):
    d = d.reshape(d.shape[0]/2,2,d.shape[1]/2,2)
    d = np.mean(d, axis=3)
    d = np.mean(d, axis=1)
    return d


class DatasetRendered2(data.Dataset):

    def __init__(self, root_dir, start_ind, stop_ind, vertical_jitter=2, depth_threshold=15, noise=0.01):
        self.from_ind = start_ind
        self.to_ind = stop_ind
        self.root_dir = root_dir
        self.depth_threshold = depth_threshold
        self.noise = noise

        #these are the camera parameters applied to
        self.depth_threshold = 15
        self.src_res = (1401, 1001)
        self.src_cxy = (700, 500)
        self.tgt_res = (1216, 896)
        self.tgt_cxy = (604, 457)
        #the focal length is shared between src and target frame
        self.focal = 1.1154399414062500e+03

        self.readout_rect = (self.src_cxy[0]-self.tgt_cxy[0], self.src_cxy[1]-self.tgt_cxy[1],
                             self.tgt_res[0], self.tgt_res[1])
        self.vertical_jitter = vertical_jitter

        # the projector intrinsics are only needed to calculate depth
        # values are in the configuration of the unity rendering project
        self.focal_projector = 850 #todo: get real value!
        self.res_projector = 1024
        self.baselines = {"left": 0.0634 - 0.0, "right": 0.0634 - 0.07501} #todo: is the baseline correct!?

    def __len__(self):
        # * 2 since we are working with stereo images
        return (self.to_ind - self.from_ind) * 2

    def __getitem__(self, idx):
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
        grey += np.random.rand(grey.shape[0], grey.shape[1]).astype(np.float32) * self.noise
        #todo: add noise and scaling of intensity here!!!!



        gt = cv2.imread(self.root_dir + f"/{idx}_{side}_gt.exr", cv2.IMREAD_UNCHANGED)

        # calculate the (normalized) pixel coordinate in the pattern projector
        # since unity only allowes us to store float16, the offset had to be stored normalized +
        # subtracted from its x-position
        x_gt = gt[:, :, 2] + np.arange(0, gt.shape[1]) * (1.0 / float(gt.shape[1]))
        x_gt = x_gt[rr[1] + v_offset:rr[1] + v_offset + rr[3], rr[0]:rr[0]+rr[2]]

        mask = np.logical_and(gt[:, :, 0] < self.depth_threshold, gt[:, :, 1] == 0)
        mask = mask[rr[1] + v_offset:rr[1] + v_offset + rr[3], rr[0]:rr[0]+rr[2]]

        #depth only is needed to guide the sampling of x_gt and mask!!! (although it definitely is complicated)
        #todo: downsample x_gt and mask!!!!
        depth = gt[rr[1] + v_offset:rr[1] + v_offset + rr[3], rr[0]:rr[0]+rr[2], 0]

        x_d = (np.arange(0, self.tgt_res[0]) - self.tgt_cxy[0]) * depth / self.focal
        x_d += self.baselines[side]
        x_d = x_d * self.focal_projector / depth + float(self.res_projector[0])/2.0
        #todo: maybe freshly calculate x_gt from depth!!!!
        return grey, x_gt, mask

