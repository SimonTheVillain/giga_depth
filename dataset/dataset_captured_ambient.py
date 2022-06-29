import torch.utils.data as data
import numpy as np
import cv2
import os
import re
from common.common import *
from dataset.dataset_rendered_shapenet import DatasetRenderedShapenet
from pathlib import Path
import random

class DatasetCapturedAmbient(data.Dataset):
    def __init__(self, sequence_dir, phase='train'):
        self.sequence_dir = sequence_dir
        dirs = os.listdir(sequence_dir)
        dirs.sort()
        dirs_val = dirs[::25]
        dirs_test = list(set(dirs).difference(set(dirs_val)))

        if phase == "val":
            self.dirs = dirs_val
        elif phase == 'train':
            self.dirs = dirs_test

    def __len__(self):
        return len(self.dirs) * 4 * 2

    def __getitem__(self, idx):
        lr = idx % 2
        idx = idx // 2
        frm = idx % 4
        idx = idx // 4

        ir = cv2.imread(f"{self.sequence_dir}/{self.dirs[idx]}/ir{frm}.png", cv2.IMREAD_UNCHANGED)
        amb = cv2.imread(f"{self.sequence_dir}/{self.dirs[idx]}/ambient{frm}.png", cv2.IMREAD_UNCHANGED)
        if lr == 1:
            ir = ir[:, ir.shape[1] // 2:]
            amb = amb[:, amb.shape[1] // 2:]
        else:
            ir = ir[:, :ir.shape[1] // 2]
            amb = amb[:, :amb.shape[1] // 2]

        ir = ir.astype(np.float32) / (256 ** 2)
        amb = amb.astype(np.float32) / (256 ** 2)

        if bool(random.getrandbits(1)):
            ir = np.flip(ir, 1)
            amb = np.flip(amb, 1)

        if bool(random.getrandbits(1)):
            ir = np.flip(ir, 0)
            amb = np.flip(amb, 0)

        ir = np.expand_dims(ir, 0).copy()
        amb = np.expand_dims(amb, 0).copy()

        ir = ir.astype(np.float32)
        amb = amb.astype(np.float32)
        return ir, amb