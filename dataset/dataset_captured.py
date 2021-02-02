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



class DatasetCaptured(data.Dataset):

    def __init__(self, root_dir, from_ind, to_ind, crop_res=(896, 1216)):
        self.root_dir = root_dir
        self.crop_res = (int(crop_res[0]), int(crop_res[1]))
        self.from_ind = from_ind
        self.to_ind = to_ind
        pass

    def __len__(self):
        return self.to_ind - self.from_ind


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = idx + self.from_ind

        ir_path = self.root_dir + "/single_shots/ir/" + str(idx) + '.png'
        ir = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
        ir = ir.astype(float) * (1.0/1000.0) * 0.5

        #in the combined image, the capture of the left camera is on the right half
        # the right camera is closer to the camera
        ir_r, ir_l = np.split(ir, 2, 1)
        image = ir_l

        image = image.astype(np.float32)

        #get random crop
        offset_x = 0
        offset_y = 0
        image = image[offset_y:(offset_y+self.crop_res[0]), offset_x:offset_x+self.crop_res[1]]
        image = np.expand_dims(image, axis=0)
        return image
