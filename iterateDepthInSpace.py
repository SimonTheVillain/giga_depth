import h5py
import numpy as np
import cv2

for i in range(150):
    print(i)
    f = h5py.File(f'/media/simon/T7/datasets/DepthInSpace/rendered_kinect/{i:08d}/frames.hdf5', 'r')
    l = list(f.keys())
    R = np.array(f.get("R")) # ["R"]
    ambient = np.array(f["ambient"])
    disp = np.array(f["disp"])
    grad = np.array(f["grad"])
    im = np.array(f["im"])
    #sgm_disp = np.array(f["sgm_disp"]) # semiglobal matching disp?
    t = np.array(f["t"])
    print(im.shape)
    cv2.imshow("im", im[0,0,:,:])
    cv2.imshow("im2", im[1,0,:,:])
    cv2.imshow("im3", im[2,0,:,:])
    cv2.imshow("im4", im[3,0,:,:])
    cv2.imshow("disp",disp[0,0,:,:]/100)
    cv2.waitKey()