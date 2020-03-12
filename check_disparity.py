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


image = io.imread(Path("/media/simon/TOSHIBA EXT/5_gt_r.exr"))

print(image.shape)
fig = plt.figure()
plt.imshow(image[:, :, 0])
plt.show()