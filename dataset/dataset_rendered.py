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



class DatasetRendered(data.Dataset):

    def __init__(self, root_dir, from_ind, to_ind, half_res=False, crop_res=(896, 1216), noise=0.0, vertical_jitter=0,
                 npy_files=False,
                 crop_gt_top=0, crop_gt_bottom=0):
        self.root_dir = root_dir
        self.crop_res = (int(crop_res[0]), int(crop_res[1]))
        self.from_ind = from_ind
        self.to_ind = to_ind
        self.half_res = half_res
        self.npy_files = npy_files
        self.crop_top = crop_gt_top
        self.crop_bottom = crop_gt_bottom
        self.noise = noise
        self.vertical_jitter = vertical_jitter
        pass
        #if partition =='train':

    def __len__(self):
        return self.to_ind - self.from_ind

    @staticmethod
    def to_grey(image):
        return 0.2125 * image[:, :, 0] + 0.7154 * image[:, :, 1] + 0.0721 * image[:, :, 2]

    def shift(self, image, y_shift):
        height = image.shape[0]
        image = np.pad(image, ((self.vertical_jitter, self.vertical_jitter), (0, 0)), mode='edge')
        image = image[y_shift:y_shift+height, :]
        return image

    def shift3(self, image, y_shift):
        height = image.shape[1]
        image = np.pad(image, ((0, 0), (self.vertical_jitter, self.vertical_jitter), (0, 0)), mode='edge')
        image = image[:, y_shift:y_shift+height, :]
        return image

    def __getitem__(self, idx):
        y_shift = np.random.randint(0, self.vertical_jitter*2)

        resolutionProjector = 1280
        resolutionIRCams = 1128
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = idx + self.from_ind
        if self.npy_files:
            path = self.root_dir + "/" + str(idx) + "_r.npy"  # putting this into a additional variable as a debug measure
            image = np.load(path)
        else:
            path = self.root_dir + "/" + str(idx) + "_r.exr"  # putting this into a additional variable as a debug measure
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            image = self.to_grey(image)

        image = self.shift(image, y_shift)


        vertical = np.asmatrix(np.array(range(0, image.shape[0])) / image.shape[0])
        vertical = np.transpose(np.matlib.repeat(vertical, image.shape[1], 0))
        v_offset = random.uniform(-0.5, 0.5)
        v_offset = 0
        vertical = vertical + v_offset#random.uniform(-0.01, 0.01)
        image = np.array([image, vertical])
        image = image.astype(np.float32)

        #image = np.array([image])
        #print(image.shape)
        #image = np.expand_dims(np.array([image]), axis=0)
        #print(image.shape)

        #todo:split mask up in gt and mask
        if self.npy_files:
            mask = np.load(self.root_dir + "/" + str(idx) + "_gt_r.npy")
            gt = np.array([mask[:, :, 2]])
            gt_d = np.array([mask[:, :, 0]])
        else:
            mask = cv2.imread(self.root_dir + "/" + str(idx) + "_gt_r.exr", cv2.IMREAD_UNCHANGED)
            gt = np.array([mask[:, :, 2]])
            gt_d = np.array([mask[:, :, 0]])

        gt = self.shift3(gt, y_shift)
        gt_d = self.shift3(gt_d, y_shift)
        x_offset = np.asmatrix(range(0, image.shape[2])).astype(np.float32)# * (1.0 / float(image.shape[2]))
        x_offset = np.asarray(np.matlib.repeat(x_offset, image.shape[1], 0))
        x_offset = np.expand_dims(x_offset, axis=0).astype(np.float32)
        #groundtruth = groundtruth + x_offset
        gt = gt * 1.0 + x_offset * (1.0 / 1216.0)
        mask1 = mask[:, :, 1] == 2.0
        #mask = np.array([mask])
        if self.npy_files:
            mask = np.array([mask1]).astype(np.float32)
        else:
            w = cv2.imread(self.root_dir + "/" + str(idx) + "_r_w.exr", cv2.IMREAD_UNCHANGED)
            w = self.to_grey(w)
            wo = cv2.imread(self.root_dir + "/" + str(idx) + "_r_wo.exr", cv2.IMREAD_UNCHANGED)
            wo = self.to_grey(wo)
            mask2 = (w-wo > 0.09)# 0.05
            mask = np.logical_and(mask1, mask2)
            mask = np.array([mask]).astype(np.float32)
            if abs(v_offset) > 0.25:
                mask[:] = 0

        mask = self.shift3(mask, y_shift)

        # cropping out part of the image
        offset_x = random.randrange(0, gt.shape[2] - self.crop_res[1] + 1)
        offset_y = random.randrange(0, gt.shape[1] - self.crop_res[0] + 1)
        gt = gt[:, offset_y:(offset_y+self.crop_res[0]), offset_x:offset_x+self.crop_res[1]]
        gt_d = gt_d[:, offset_y:(offset_y+self.crop_res[0]), offset_x:offset_x+self.crop_res[1]]
        image = image[:, offset_y:(offset_y+self.crop_res[0]), offset_x:offset_x+self.crop_res[1]]
        mask = mask[:, offset_y:(offset_y+self.crop_res[0]), offset_x:offset_x+self.crop_res[1]]

        if self.crop_top != 0 or self.crop_bottom != 0:
            mask = mask[:, self.crop_top:-self.crop_bottom, :]
            gt = gt[:, self.crop_top:-self.crop_bottom, :]
            gt_d = gt_d[:, self.crop_top:-self.crop_bottom, :]

        if self.half_res:
            mask = transform.resize(mask, (mask.shape[0], mask.shape[1] / 2, mask.shape[2] / 2),)
            mask[mask == 2.0] = 1.0
            mask[mask != 1.0] = 0.0

            gt = \
                transform.resize(gt,
                                 (gt.shape[0], gt.shape[1] / 2, gt.shape[2] / 2)) - \
                0.5/resolutionProjector # not a hundred percent sure why this is necessary (why projector and not IRCam
            #TODO: find out why this 0.5/resolutionProjector is limited to half_res
            gt_d = \
                transform.resize(gt_d,
                                 (gt_d.shape[0], gt_d.shape[1] / 2, gt_d.shape[2] / 2))


        #print(idx)
        if False:
            fig = plt.figure()
            fig.add_subplot(3, 1, 1)
            plt.imshow(image[0, :, :])

            fig.add_subplot(3, 1, 2)
            plt.imshow(mask[0, :, :], vmin=0, vmax=1)

            fig.add_subplot(3, 1, 3)
            plt.imshow(gt[0, :, :])
            plt.show()

        image[0, :, :] = image[0, :, :] + np.random.rand(image.shape[1], image.shape[2]).astype(np.float32) * self.noise

        sample = {'image': image, 'mask': mask, 'gt': gt,
                  'gt_d': gt_d, 'offset': np.array([offset_x, offset_y], np.float32)}
        return sample