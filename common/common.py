import numpy as np
import scipy.signal
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

def LCN(image, window_size=9):
    eps = 0.001
    width = image.shape[1]
    height = image.shape[0]
    im_padded = np.pad(image, int(window_size / 2), mode="edge")
    kernel = np.ones((window_size, 1), dtype=np.float32) * (1.0 / window_size)
    mean = scipy.signal.convolve2d(im_padded, kernel, mode="valid")
    kernel = np.ones((1, window_size), dtype=np.float32) * (1.0 / window_size)
    mean = scipy.signal.convolve2d(mean, kernel, mode="valid")
    #cv2.imshow("mean", mean)
    sq = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    for i in range(window_size):
        for j in range(window_size):
            sq[:, :] += (im_padded[i:i+height, j:j+width] - mean) ** 2

    sigma = np.sqrt(sq)
    I = (image - mean) / (sigma + eps)

    return I * 0.5 + 0.5

def LCN_tensors(x, window_size=9):
    eps = 0.001
    dtype = x.dtype
    device = x.device
    width = x.shape[3]
    height = x.shape[2]
    x_padded = F.pad(x, [int(window_size / 2)]*4, mode="replicate")
    kernel = torch.ones((1, 1, window_size, 1), dtype=dtype, device=device) * (1.0 / float(window_size))
    mean = F.conv2d(x_padded, kernel)
    kernel = torch.ones((1, 1, 1, window_size), dtype=dtype, device=device) * (1.0 / float(window_size))
    mean = F.conv2d(mean, kernel)
    sq = torch.zeros((x.shape[0], 1, height, width), dtype=dtype, device=device)
    for i in range(window_size):
        for j in range(window_size):
            sq[:, :] += (x_padded[:, :, i:i+height, j:j+width] - mean) ** 2

    sigma = torch.sqrt(sq)
    I = (x - mean) / (sigma + eps)

    return I * 0.5 + 0.5