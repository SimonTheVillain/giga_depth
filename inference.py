import cv2
import torch
import torch.nn as nn
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import os


dataset_path = "/media/simon/SSD/datasets/structure_core"
dataset_rendered_path = ""
if os.name == 'nt':
    dataset_path = "D:/datasets/structure_core"  # todo: path for windows
    dataset_rendered_path = "D:/dataset_filtered"







model_path = "trained_models/model_2_lr_0001.pt"
count = 800
device = torch.device('cpu')
model = torch.load(model_path, map_location=device)
model.eval()
model.to(device)
useRendered = False

with torch.no_grad():
    for i in range(0, count):
        print("image {}".format(i))
        for offset in [0]:#np.arange(-0.1, 0.1, 0.01):
            print(offset)
            scale = 1.0
            if useRendered:
                ir_path = dataset_path + "/" + str(i) + '_r.png'

            else:
                ir_path = dataset_path + "/single_shots/ir/" + str(i) + '.png'
                rgb_path = dataset_path + '/single_shots/rgb/' + str(i) + '.png'
                depth_path = dataset_path + '/single_shots/depth/' + str(i) + '.png'
                ir = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
                depth = cv2.imread(depth_path)
                rgb = cv2.imread(rgb_path)
                ir = ir.astype(float) * (1.0/1000.0) * 0.5
                ir_l, ir_r = np.split(ir, 2, 1)
                dim = (int(ir_r.shape[1] * scale), int(ir_r.shape[0] * scale))
                ir_r = cv2.resize(ir_r, dim)

            vertical = np.asmatrix(np.array(range(0, ir_r.shape[0])) / ir_r.shape[0])
            vertical = np.transpose(np.matlib.repeat(vertical, ir_r.shape[1], 0)) + offset
            image = np.array([[ir_r, vertical]]).astype(float)
            image = image.astype(np.float32)
            output, latent = model(torch.tensor(image).to(device))
            plt.imshow(output[0, 0, :, :].cpu().detach().numpy(), vmin=0, vmax=1)

            plt.show()


            output_mat = output.cpu().detach().numpy()
            cv2.imshow("rgb", rgb)
            cv2.imshow("ir_r", ir_r)
            cv2.imshow("depth", depth)
            cv2.imshow("mask", output_mat[0, 1, :, :])
            cv2.imshow("output", output_mat[0, 0, :, :])
            cv2.waitKey()#100






