import cv2
from mpl_toolkits.mplot3d import Axes3D

import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from skimage.feature import peak_local_max

import matplotlib.pyplot as plt
import numpy as np




image = cv2.imread("pattern_corrected_16.png", cv2.IMREAD_UNCHANGED)

#cv2.imshow("image", image * 10)
#cv2.waitKey()
#https://www.researchgate.net/post/Image_Processing_How_to_find_local_sub-pixel_maxima_in_image
z = image#image[100:150, 100:150]
z = cv2.resize(z, (z.shape[1] * 10, z.shape[0] * 10), interpolation=cv2.INTER_LANCZOS4)
x, y = np.meshgrid(range(z.shape[1]), range(z.shape[0]))

# show hight map in 3d
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(x, y, z)
#plt.title('z as 3d height map')
#plt.show()

threshold = 1000
coordinates = peak_local_max(z, min_distance=10)
coordinates2 = list()
for i in range(coordinates.shape[0]):
    if z[coordinates[i, 0], coordinates[i, 1]] > threshold:
        coordinates2.append(coordinates[i, :])

coordinates = np.array(coordinates2)
# show hight map in 2d
plt.figure()
plt.title('z as 2d heat map')
p = plt.imshow(z)
plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
plt.colorbar(p)
plt.show()