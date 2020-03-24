import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib

half_res = False
fxr = 1115.44
cxr = 604
fxl = 1115.44
cxl = 604
cxr = cxl = 608  # 1216/2 (lets put it right in the center since we are working on
fxp = 1115.44
cxp = 640  # put the center right in the center
b1 = 0.0634
b2 = 0.07501
epsilon = 0.01  # very small pixel offsets should be forbidden
if half_res:
    fxr = fxr * 0.5
    cxr = cxr * 0.5
    fxl = fxl * 0.5
    cxl = cxl * 0.5

ind = 3

path = "D:/dataset/"

image = cv2.imread(path + str(ind) + "_r.exr", cv2.IMREAD_UNCHANGED)

#cv2.imshow("image", image)
#cv2.waitKey()

gt = cv2.imread(path + str(ind) + "_gt_r.exr", cv2.IMREAD_UNCHANGED)

xr = np.asmatrix(np.array(range(0, gt.shape[1]))).astype(np.float32)
xr = np.matlib.repeat(xr, gt.shape[0], 0)

z_gt = gt[:, :, 0]
xp = gt[:, :, 2]
xp = xp * 1280.0 + xr * (1280.0/1216)
z_ = (xp - cxp) * fxr - (xr - cxr) * fxp
#z_ = -z_
z = np.divide(b1 * fxp * fxr,  z_)



fig = plt.figure()
count_y = 3
count_x = 2
fig.add_subplot(count_y, count_x, 1)
plt.imshow(image, vmin=0, vmax=0.2, cmap='gist_gray')
#plt.imshow(debug_right[0, 0, :, :].detach().cpu(), vmin=0, vmax=1)
plt.title("Intensity")
fig.add_subplot(count_y, count_x, 2)
plt.imshow(z_gt)#, vmin=0, vmax=10)
plt.title("gt")
fig.add_subplot(count_y, count_x, 3)
plt.imshow(z, vmin=0, vmax=20)
plt.title("derived")
fig.add_subplot(count_y, count_x, 4)
plt.imshow(gt[:, :, 2])#, vmin=0, vmax=)
plt.title("data")

fig.add_subplot(count_y, count_x, 5)
plt.imshow(gt[:, :, 1])#, vmin=0, vmax=)
plt.title("mask")

fig.add_subplot(count_y, count_x, 6)
plt.imshow(xr)#, vmin=0, vmax=)
plt.title("xr")
plt.show()
