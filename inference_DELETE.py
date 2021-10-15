import cv2
import torch
import torch.nn as nn
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import os
import open3d as o3d

def calc_x_pos(class_inds, regressions, class_count, neighbourhood_regression):
    #print(class_inds.shape)
    #print(regressions.shape)
    regressions = torch.gather(regressions, dim=1, index=class_inds)
    if neighbourhood_regression:
        regressions = (regressions * (1.0/3.0) - 1.0/3.0) * (1.0 / class_count)
    else:
        regressions = regressions * (1.0 / class_count)
    x = class_inds * (1.0 / class_count) + regressions
    return x
def calc_depth_right(right_x_pos, half_res=False, real_data=True):#todo: offset!
    device = right_x_pos.device
    projector_width = 1280.0
    fxr = 1115.44
    cxr = 604.0
    fxl = 1115.44
    cxl = 604.0
    cxr = cxl = 608.0  # 1216/2 (lets put it right in the center since we are working on
    fxp = 1115.44
    cxp = 640.0  # put the center right in the center
    b1 = 0.0634#baseline projector to right camera
    b2 = 0.07501#baselne between both cameras
    epsilon = 0.01  # very small pixel offsets should be forbidden
    if real_data:
        cxp = 650.0#596.5
        cxr = 604.0
    if half_res:
        fxr = fxr * 0.5
        cxr = cxr * 0.5
        fxl = fxl * 0.5
        cxl = cxl * 0.5

    xp = right_x_pos[:, [0], :, :] * projector_width  # float(right.shape[3])
    if half_res:
        pass
        #xp = xp - 0.5 #why the downsampling shifted everything by 0.5 pixel?

    xr = np.asmatrix(np.array(range(0, right_x_pos.shape[3]))).astype(np.float32)
    xr = torch.tensor(np.matlib.repeat(xr, right_x_pos.shape[2], 0), device=device)
    xr = xr.unsqueeze(0).unsqueeze(0).repeat((right_x_pos.shape[0], 1, 1, 1))
    z_ = (xp - cxp) * fxr - (xr - cxr) * fxp

    z = torch.div(b1 * fxp * fxr, z_)
    return z

def display_pcl(z, offset ,half_res=True):
    fx = 1115.44
    cxr = 604.0
    cyr = 896.0 * 0.5
    if half_res:
        fx = fx * 0.5
        cxr = cxr * 0.5
        cyr = cyr * 0.5
    print(z.shape)
    pts = []
    for i in range(0, z.shape[2]):
        for j in range(0, z.shape[3]):
            y = i + offset - cyr
            x = j - cxr
            depth = z[0, 0, i, j]
            if 0 < depth < 5:
                pts.append([x*depth/fx, y*depth/fx, depth])
    xyz = np.array(pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])

def main():
    #xyz = np.zeros((10, 3))
    #xyz[5:, : ] = 1
    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(xyz)
    #o3d.geometry.create_mesh_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
    #o3d.visualization.draw_geometries([pcd])#, zoom=0.3412,
                                  #front=[0.4257, -0.2125, -0.8795],
                                  #lookat=[2.6172, 2.0475, 1.532],
                                  #up=[-0.0694, -0.9768, 0.2024])
    use_rendered = True
    show_pcl = False
    slice_offset = 100
    slice_size = 142 # 9 and 10
    #slice_size = 112 + 27*2 # number 8
    if use_rendered:
        dataset_path = "/media/simon/ssd_data/data/dataset_reduced_0_08"
        count = 10000
    else:
        dataset_path = "/media/simon/ssd_data/data/datasets/structure_core"
        count = 800
        slice_offset += 8
        if os.name == 'nt':
            dataset_path = "D:/datasets/structure_core"  # todo: path for windows
            dataset_rendered_path = "D:/dataset_filtered"

    model_path = "trained_models/CR_9hs_chckpt_2.pt"
    device = torch.device(torch.cuda.current_device())
    model = torch.load(model_path, map_location=device)
    model.eval()
    model.to(device)


    with torch.no_grad():
        for i in range(0, count):
            print("image {}".format(i))
            for offset in [0]:  # np.arange(-0.1, 0.1, 0.01):
                print(offset)
                scale = 1.0
                if use_rendered:
                    ir_path = dataset_path + "/" + str(i) + '_r.exr'
                    ir_r = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
                    ir_r = cv2.cvtColor(ir_r, cv2.COLOR_BGR2GRAY)
                    ir_r = ir_r[slice_offset:slice_offset + slice_size, :]

                else:
                    ir_path = dataset_path + "/single_shots/ir/" + str(i) + '.png'
                    rgb_path = dataset_path + '/single_shots/rgb/' + str(i) + '.png'
                    depth_path = dataset_path + '/single_shots/depth/' + str(i) + '.png'
                    ir = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
                    depth_gt = cv2.imread(depth_path)
                    rgb = cv2.imread(rgb_path)

                    # get a 142 pixel high strip from the image
                    ir = ir[slice_offset:slice_offset + slice_size, :]
                    depth_gt = depth_gt[slice_offset:slice_offset + slice_size, :]
                    ir = ir.astype(float) * (1.0 / 1000.0) * 0.5
                    ir_l, ir_r = np.split(ir, 2, 1)
                    dim = (int(ir_r.shape[1] * scale), int(ir_r.shape[0] * scale))
                    ir_r = cv2.resize(ir_r, dim)

                #vertical = np.asmatrix(np.array(range(0, ir_r.shape[0])) / ir_r.shape[0])
                #vertical = np.transpose(np.matlib.repeat(vertical, ir_r.shape[1], 0)) + offset
                image = np.array([[ir_r]]).astype(float)
                image = image.astype(np.float32)
                classes, regressions, mask, x_latent = model(torch.tensor(image[:, [0], :, :]).to(device))
                classes = torch.argmax(classes, dim=1).unsqueeze(dim=1)
                output = calc_x_pos(class_inds=classes, regressions=regressions,
                                    class_count=128, neighbourhood_regression=False)

                depth = calc_depth_right(output, True, real_data=not use_rendered)
                if use_rendered and show_pcl:
                    #pass
                    display_pcl(depth.cpu().detach().numpy(), slice_offset, True)

                fig, axs = plt.subplots(4)
                axs[0].imshow(image[0, 0, :, :])
                axs[0].set_title("input")
                axs[1].imshow(output[0, 0, :, :].cpu().detach().numpy(), vmin=0, vmax=1)
                axs[1].set_title("x_pos")

                axs[2].imshow(depth[0, 0, :, :].cpu().detach().numpy(), vmin=0, vmax=5)
                axs[2].set_title("depth")
                axs[3].plot(output[0, 0, 27, :].cpu().detach().numpy())
                axs[3].set_title("x_pos (1-line)")

                plt.show()

                output_mat = output.cpu().detach().numpy()
                #cv2.imshow("rgb", rgb)
                cv2.imshow("ir_r", ir_r)
                #cv2.imshow("depth_gt", depth_gt)
                #cv2.imshow("mask", output_mat[0, 1, :, :])
                cv2.imshow("output", output_mat[0, 0, :, :])
                cv2.waitKey()  # 100


if __name__ == '__main__':
    main()
