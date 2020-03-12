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



class DatasetCapturedStereo(data.Dataset):

    def __init__(self, root_dir, from_ind, to_ind, crop_res=(896, 1216)):
        self.root_dir = root_dir
        self.crop_res = (int(crop_res[0]), int(crop_res[1]))
        self.from_ind = from_ind
        self.to_ind = to_ind
        pass

    def __len__(self):
        return self.to_ind - self.from_ind


    def __getitem__(self, idx):
        pass # todo: implement this
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = idx + self.from_ind

        ir_path = self.root_dir + "/single_shots/ir/" + str(idx) + '.png'
        ir = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
        ir = ir.astype(float) * (1.0/1000.0) * 0.5
        ir_l, ir_r = np.split(ir, 2, 1)
        image = ir_r

        #get the y coordinate
        vertical = np.asmatrix(np.array(range(0, image.shape[0])) / image.shape[0])
        vertical = np.transpose(np.matlib.repeat(vertical, image.shape[1], 0))
        vertical = np.array(vertical).astype(np.float32)
        image = np.asarray(image) #, vertical])
        image = image.astype(np.float32)

        #get random crop
        offset_x = random.randrange(0, image.shape[2] - self.crop_res[1] + 1)
        offset_y = random.randrange(0, image.shape[1] - self.crop_res[0] + 1)
        image_right = image[offset_y:(offset_y+self.crop_res[0]), offset_x:offset_x+self.crop_res[1]]
        vertical = vertical[offset_y:(offset_y+self.crop_res[0]), offset_x:offset_x+self.crop_res[1]]

        #same with right image
        image = np.array(ir_l)
        image = image.asastype(np.float32)
        image_left = image[ offset_y:(offset_y+self.crop_res[0]), offset_x:offset_x+self.crop_res[1]]

        sample = {'image_left': image_left, 'image_right': image_right, 'vertical': vertical}
        return sample