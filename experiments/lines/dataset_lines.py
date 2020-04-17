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



class DatasetLines(data.Dataset):

    def __init__(self, root_dir, from_ind, to_ind):
        self.root_dir = root_dir
        self.from_ind = from_ind
        self.to_ind = to_ind
        pass
        #if partition =='train':

    def __len__(self):
        return self.to_ind - self.from_ind

    def __getitem__(self, idx):
        i = idx + self.from_ind
        image = np.load(self.root_dir + "/" + str(i) + "_r.npy")
        r = int(image.shape[0] / 2)
        image_padded = np.zeros((image.shape[0], image.shape[1] + 2*r), dtype=np.float32)
        image_padded[:, r: r + image.shape[1]] = image
        image_padded = np.array([image_padded], dtype=np.float32)
        gt = np.load(self.root_dir + "/" + str(i) + "_gt_r.npy")
        mask = gt[:, :, 1] == 2.0
        mask = np.array([mask], dtype=np.float32)
        gt = gt[:, :, 2]#the groundtruth is in the R channel. (the preparation program used opencv)

        x_offset = np.asmatrix(range(0, image.shape[1])).astype(np.float32)# * (1.0 / float(image.shape[2]))
        gt = gt * 1.0 + x_offset * (1.0 / 1216.0)
        gt = np.array([gt]).astype(np.float32)
        sample = {'image': image_padded, 'mask': mask, 'gt': gt}
        return sample