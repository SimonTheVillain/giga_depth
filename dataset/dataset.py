import torch
import torch.nn as nn
import torch.utils.data as data
from skimage import io, transform
import numpy as np
import numpy.matlib
from pathlib import Path
import cv2

import matplotlib.pyplot as plt



class StructureCoreDatasetRendered(data.Dataset):

    def __init__(self, root_dir, from_ind, to_ind, half_res=False):
        self.root_dir = root_dir

        self.from_ind = from_ind
        self.to_ind = to_ind
        self.half_res = half_res
        pass
        #if partition =='train':

    def __len__(self):
        return self.to_ind - self.from_ind

    @staticmethod
    def to_grey(image):
        return 0.2125 * image[:, :, 0] + 0.7154 * image[:, :, 1] + 0.0721 * image[:, :, 2]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = idx + self.from_ind
        image = io.imread(Path(self.root_dir)/Path(str(idx) + "_r.exr"))
        image = self.to_grey(image)

        vertical = np.asmatrix(np.array(range(0, image.shape[0])) / image.shape[0])
        vertical = np.transpose(np.matlib.repeat(vertical, image.shape[1], 0))
        image = np.array([image, vertical])
        image = image.astype(np.float32)

        #image = np.array([image])
        #print(image.shape)
        #image = np.expand_dims(np.array([image]), axis=0)
        #print(image.shape)

        #todo:split mask up in gt and mask
        mask = io.imread(Path(self.root_dir)/Path(str(idx) + "_gt_r.exr"))
        groundtruth = mask[:, :, 0]
        groundtruth = np.array([groundtruth])
        mask1 = mask[:, :, 1] == 0
        #mask = np.array([mask])

        w = io.imread(Path(self.root_dir)/Path(str(idx) + "_r_w.exr"))
        w = self.to_grey(w)
        wo = io.imread(Path(self.root_dir)/Path(str(idx) + "_r_wo.exr"))
        wo = self.to_grey(wo)
        mask2 = (w-wo > 0.09)# 0.05
        mask = np.logical_and(mask1, mask2)
        mask = np.array([mask]).astype(np.float32)

        if self.half_res:
            mask = transform.resize(mask, (mask.shape[0], mask.shape[1] / 2, mask.shape[2] / 2),)
            mask[mask != 1.0] = 0.0

            groundtruth = \
                transform.resize(groundtruth,
                                 (groundtruth.shape[0], groundtruth.shape[1] / 2, groundtruth.shape[2] / 2))



        #print(idx)
        #fig = plt.figure()
        #fig.add_subplot(3, 1, 1)
        #plt.imshow(image[0, :, :])

        #fig.add_subplot(3, 1, 2)
        #plt.imshow(mask[0, :, :])

        #fig.add_subplot(3, 1, 3)
        #plt.imshow(groundtruth[0, :, :])
        #plt.show()

        sample = {'image': image, 'mask': mask, 'gt': groundtruth}
        return sample


        pass
