import cv2
from mpl_toolkits.mplot3d import Axes3D

import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy import signal
from skimage.feature import peak_local_max

import matplotlib.pyplot as plt
import numpy as np


result_scale = 6
intermediate_scale = 10
#create the dot pattern by convolution
size = 40
sigma = 40
x, y = np.meshgrid(range(-size, size+1), range(-size, size+1))
kernel = np.exp(-(0.5/sigma)*(np.multiply(x, x) + np.multiply(y, y)))
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(x, y, kernel)
#plt.title('z as 3d height map')
#plt.show()


image = cv2.imread("pattern_averaged_corrected_16.png", cv2.IMREAD_UNCHANGED)

#cv2.imshow("image", image * 10)
#cv2.waitKey()
#https://www.researchgate.net/post/Image_Processing_How_to_find_local_sub-pixel_maxima_in_image
z = image.astype(np.float)#image[100:150, 100:150]
initial_shape = z.shape
z = cv2.resize(z, (z.shape[1] * intermediate_scale, z.shape[0] * intermediate_scale), interpolation=cv2.INTER_CUBIC) #interpolation=cv2.INTER_LANCZOS4)
x, y = np.meshgrid(range(z.shape[1]), range(z.shape[0]))
correction = (np.power(x - z.shape[1]/2, 2) + np.power(y - z.shape[0], 2)) * 0.000000005 + 1.0
z = np.multiply(z, correction)

# show hight map in 3d
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(x, y, z)
#plt.title('z as 3d height map')
#plt.show()

threshold = 400
coordinates = peak_local_max(z, min_distance=5)
coordinates2 = list()
for i in range(coordinates.shape[0]):
    if z[coordinates[i, 0], coordinates[i, 1]] > threshold:
        coordinates2.append(coordinates[i, :])
coordinates = np.array(coordinates2)
print(coordinates.shape)
# show hight map in 2d
plt.figure()
plt.title('z as 2d heat map')
p = plt.imshow(z)
plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
plt.colorbar(p)

plt.show()

z_new = np.zeros(z.shape)
z_new[coordinates[:, 0], coordinates[:, 1]] = 1
print("before convolution")
#TODO: faster convolution
#https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.signal.fftconvolve.html
#z2 = signal.convolve2d(z_new, kernel, 'same')
z2 = signal.fftconvolve(z_new, kernel, mode='same')
print("after convolution")
p = plt.imshow(z2)
plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
plt.colorbar(p)
z2 = cv2.resize(z2, (initial_shape[1] * result_scale, initial_shape[0] * result_scale), interpolation=cv2.INTER_LINEAR)
z3 = np.zeros((z2.shape[1], z2.shape[1]))
half = int((z3.shape[0] - z2.shape[0]) / 2)
z3[half:half + z2.shape[0], :] = z2
cv2.imwrite("pattern_enhanced_8.png", z3*255)


plt.show()

