import skimage
import skimage.io
import cv2
import numpy as np

test = cv2.imread("/home/simon/Downloads/Recordings/aov_image_0019.exr",)
#test = skimage.io.imread("/home/simon/Downloads/Recordings/aov_image_0019.exr")
print(test.shape)
print(np.min(test))
print(np.max(test))