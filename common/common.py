import numpy as np
import scipy.signal
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


def downsample(image):
    image = image.reshape(int(image.shape[0]/2), 2, int(image.shape[1]/2), 2)
    image = np.mean(image, axis=3)
    image = np.mean(image, axis=1)
    return image

def downsampleDepth(d):
    d = d.reshape(int(d.shape[0]/2), 2, int(d.shape[1]/2), 2)
    d = np.min(d, axis=3)
    d = np.min(d, axis=1)
    return d

def downsampleDisp(d):
    d = d.reshape(int(d.shape[0]/2), 2, int(d.shape[1]/2), 2)
    d = np.max(d, axis=3)
    d = np.max(d, axis=1)
    return d

def dilatation(src, r):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    dilation_dst = cv2.dilate(src, element)
    return dilation_dst


#TODO: this version is super inefficient! Remove!!!!
def LCN_np(image, window_size=9):
    eps = 0.001
    width = image.shape[1]
    height = image.shape[0]
    im_padded = np.pad(image, int(window_size / 2), mode="edge")
    kernel = np.ones((window_size, 1), dtype=np.float32) * (1.0 / window_size)
    mean = scipy.signal.convolve2d(im_padded, kernel, mode="valid")
    kernel = np.ones((1, window_size), dtype=np.float32) * (1.0 / window_size)
    mean = scipy.signal.convolve2d(mean, kernel, mode="valid")
    cv2.imshow("mean_np", mean)
    sq = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    for i in range(window_size):
        for j in range(window_size):
            sq[:, :] += (im_padded[i:i+height, j:j+width] - mean) ** 2

    sigma = np.sqrt(sq/(window_size**2))
    cv2.imshow("std_np", sigma)
    I = (image - mean) / (sigma + eps)

    return I * 0.5 + 0.5

#TODO: this version is super inefficient! REMOVE!!!!!
'''
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

    sigma = torch.sqrt(sq * 1.0 / (window_size*window_size))
    I = (x - mean) / (sigma + eps)

    return I
'''
def LCN_tensors(x, window_size=9):
    eps = 0.001
    dtype = x.dtype
    device = x.device

    w = torch.ones((x.shape[1], x.shape[1], window_size, window_size), device=device, dtype=dtype) / \
        (window_size * window_size)
    mean_local = F.conv2d(input=x, weight=w, padding=window_size // 2)

    mean_square_local = F.conv2d(input=x ** 2, weight=w, padding=window_size // 2)
    # std_local = (mean_square_local - mean_local ** 2) * (kSize ** 2) / (kSize ** 2 - 1)
    std_local = torch.sqrt(torch.clamp(mean_square_local - mean_local ** 2, min=eps**2))  # fix by simon!!!!! (why kSize ** 2 - >1< ???)

    return (x - mean_local) / (std_local + eps), mean_local, std_local


#TODO: make it a bit more elegant. kernel size instead of radius
# no divisions in forward pass, but incorporate with weights
# ALSO IT SEEMS TO BE WRONG!!!!!!!gi
class LCN(torch.nn.Module):
  '''
  Local Contract Normalization
  '''
  def __init__(self, radius=4, epsilon=0.001):
    #assert False, "This module is superterrible and actually wrong!"
    super(LCN, self).__init__()
    self.box_conv = torch.nn.Sequential(
        torch.nn.ReflectionPad2d(radius),
        torch.nn.Conv2d(1, 1, kernel_size=(2*radius+1, 1), bias=False),
        torch.nn.Conv2d(1, 1, kernel_size=(1, 2*radius+1), bias=False)
    )
    self.box_conv[1].weight.requires_grad=False
    self.box_conv[1].weight.fill_(1.)#TODO: (1./(2*radius+1))
    self.box_conv[2].weight.requires_grad=False
    self.box_conv[2].weight.fill_(1.)#TODO: (1./(2*radius+1))

    self.epsilon = epsilon
    self.radius = radius

  def forward(self, x):
    boxs = self.box_conv(x)
    n = (2*self.radius+1)**2
    avgs = boxs / n # todo: remove
    x_sq = self.box_conv(x**2)
    #TODO: isn't this wrong?
    stds = torch.sqrt(torch.clamp((x_sq - 2 * boxs * avgs + n * avgs**2) / n, min=0.0))#TODO: remove /n

    return (x - avgs) / (stds + self.epsilon)


