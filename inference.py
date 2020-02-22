import cv2
import torch
import torch.nn as nn
import numpy as np
import numpy.matlib


dataset_path = "/media/simon/SSD/datasets/structure_core"
model_path = "/home/simon/pycharm/GigaDepth/trained_models/model_flat_half_no_mask_4.pt"
count = 46
device = torch.device('cpu')
model = torch.load(model_path, map_location=device)
model.eval()
model.to(device)


for i in range(0, count):
    print("image {}".format(i))
    for offset in [0]:#np.arange(-0.1, 0.1, 0.01):
        print(offset)
        scale = 1.0
        ir_path = dataset_path + "/single_shots/ir/" + str(i) + '.png'
        rgb_path = dataset_path + '/single_shots/rgb/' + str(i) + '.png'
        depth_path = dataset_path + '/single_shots/depth/' + str(i) + '.png'
        ir = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path)
        ir = ir.astype(float) * (1.0/1000.0) * 0.5
        ir_l, ir_r = np.split(ir, 2, 1)
        dim = (int(ir_r.shape[1] * scale), int(ir_r.shape[0] * scale))
        ir_r = cv2.resize(ir_r, dim)
        vertical = np.asmatrix(np.array(range(0, ir_r.shape[0])) / ir_r.shape[0])
        vertical = np.transpose(np.matlib.repeat(vertical, ir_r.shape[1], 0)) + offset
        image = np.array([[ir_r, vertical]]).astype(float)
        image = image.astype(np.float32)
        output = model(torch.tensor(image).to(device))


        output_mat = output.cpu().detach().numpy()
        cv2.imshow("rgb", rgb)
        cv2.imshow("ir_r", ir_r)
        cv2.imshow("ir_l", ir_l)
        cv2.imshow("depth", depth)
        cv2.imshow("mask", output_mat[0, 1, :, :])
        cv2.imshow("output", output_mat[0, 0, :, :])
        cv2.waitKey(100)






