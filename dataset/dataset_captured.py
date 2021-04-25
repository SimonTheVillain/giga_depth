import torch
import torch.nn as nn
import torch.utils.data as data
from skimage import io, transform
import numpy as np
from pathlib import Path
import cv2
import os
import random

import matplotlib.pyplot as plt



class DatasetCaptured(data.Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir

        files = os.listdir(f"{root_dir}/ir")
        #print(files)
        keys = []
        for file in files:
            if os.path.isfile(f"{root_dir}/ir/{file}"):
                keys.append(file.split(".")[0])

        self.keys = list(set(keys))
        pass

    def __len__(self):
        return len(self.keys) * 2


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        lr = idx % 2
        idx = int(idx / 2)
        ir_path = self.root_dir + "/single_shots/ir/" + str(idx) + '.png'
        ir = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
        ir = ir.astype(float) * (1.0/1000.0) * 0.5

        #in the combined image, the capture of the left camera is on the right half
        # the right camera is closer to the camera
        ir_r, ir_l = np.split(ir, 2, 1)
        image = ir_l
        if lr == 1:
            image = ir_r

        image = image.astype(np.float32)

        #get random crop
        offset_x = 0
        offset_y = 0
        #image = image[offset_y:(offset_y+self.crop_res[0]), offset_x:offset_x+self.crop_res[1]]
        image = np.expand_dims(image, axis=0)
        return image
