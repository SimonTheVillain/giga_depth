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

class DatasetRenderedStereo(data.Dataset):

    def __init__(self, root_dir, from_ind, to_ind, crop_res=(896, 1216)):
        self.root_dir = root_dir
        self.crop_res = (int(crop_res[0]), int(crop_res[1]))
        self.from_ind = from_ind
        self.to_ind = to_ind


    def __len__(self):
        return self.to_ind - self.from_ind

    @staticmethod
    def to_grey(image):
        return 0.2125 * image[:, :, 0] + 0.7154 * image[:, :, 1] + 0.0721 * image[:, :, 2]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = idx + self.from_ind
        path = Path(self.root_dir) / Path(str(idx) + "_r.exr")#putting this into a additional variable as a debug measure
        image = io.imread(path)
        image = np.asarray(self.to_grey(image))
        vertical = np.asmatrix(np.array(range(0, image.shape[0])) / image.shape[0])
        vertical = np.transpose(np.matlib.repeat(vertical, image.shape[1], 0)).astype(np.float32)

        #v_offset = random.uniform(-0.5, 0.5)
        v_offset = 0
        vertical = vertical + v_offset#random.uniform(-0.01, 0.01)
        image = image.astype(np.float32)

        offset_x = random.randrange(0, image.shape[1] - self.crop_res[1] + 1)
        offset_y = random.randrange(0, image.shape[0] - self.crop_res[0] + 1)
        image_right = image[offset_y:(offset_y+self.crop_res[0]), offset_x:offset_x+self.crop_res[1]]
        vertical = vertical[offset_y:(offset_y+self.crop_res[0]), offset_x:offset_x+self.crop_res[1]]


        path = Path(self.root_dir) / Path(str(idx) + "_l.exr")#putting this into a additional variable as a debug measure
        image = io.imread(path)
        image = np.asarray(self.to_grey(image))
        image = image.astype(np.float32)
        image_left = image[offset_y:(offset_y+self.crop_res[0]), offset_x:offset_x+self.crop_res[1]]
        sample = {'image_left': image_left, 'image_right': image_right, 'vertical': np.asarray(vertical)}
        return sample
