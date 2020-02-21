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



class DatasetCombined(data.Dataset):

    def __init__(self, dataset_rendered, dataset_real):
        self.dataset1 = dataset_rendered
        self.dataset2 = dataset_real

        pass
        #if partition =='train':

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        size1 = len(self.dataset1)
        if idx < size1:
            image = self.dataset1[idx].image
            domain = 0.0
        else:
            image = self.dataset2[idx-size1]
            domain = 1.0

        sample = {'image' : image, 'domain' : domain}
        return sample


        pass
