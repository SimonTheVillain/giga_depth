import torch
import os
#Work in the parent directory
os.chdir("../")
from model.composite_model import CompositeModel
import cv2
import numpy as np
import timeit


run_artificial = True

device = "cuda:0"
regressor_model_pth = "trained_models/full_67_lcn_j4_light_regressor.pt"
backbone_model_pth = "trained_models/full_67_lcn_j4_light_backbone.pt"

if run_artificial:
    path_src = "/media/simon/T7/datasets/structure_core_unity_test"
    path_results = "/media/simon/T7/datasets/structure_core_unity_test_results/GigaDepth67" #GigaDepthNoLCN"
else:
    path_src = "/media/simon/T7/datasets/structure_core_photoneo_test"
    path_results = "/media/simon/T7/datasets/structure_core_photoneo_test_results/GigaDepth67"


def apply_recursively(model, input_root, output_root, current_sub ):
    pad = 0.1
    current_src = os.path.join(input_root, current_sub)
    current_tgt = os.path.join(output_root, current_sub)
    if os.path.isdir(current_src) and not os.path.exists(current_tgt):
        os.mkdir(current_tgt)
    for path in os.listdir(current_src):
        current = os.path.join(current_src, path)
        if os.path.isdir(current):
            print(os.path.join(current_sub, path))
            apply_recursively(model, input_root, output_root, os.path.join(current_sub, path))
            continue
        if (path[-3:] == 'jpg') or (path[-3:] == 'png' and path[:3] == "ir0"):
            src = current
            ir = cv2.imread(current, cv2.IMREAD_UNCHANGED)

            if path[-3:] == "png": #format is like "im0_left.png"
                tag = path[2:-4]
                ir = ir[:, 1216:]
                ir = ir.astype(np.float32) / 65536.0
            else: #format is like 11_left.jpg
                tag = path[0:-4]
                ir = ir.astype(np.float32) / 256.0

            cv2.imshow("ir", ir)
            cv2.waitKey(1)
            ir = torch.tensor(ir).to(device)
            ir = ir.unsqueeze(0).unsqueeze(0)
            dst = os.path.join(current_tgt, path)

            if len(ir.shape) == 5:#ir.shapeir.shape[4] == 3:
                ir = (ir[..., 0] + ir[..., 1] + ir[..., 2]) / 3
                src_res = (1401, 1001)
                src_cxy = (700, 500)
                tgt_res = (1216, 896)
                tgt_cxy = (604, 457)
                # the focal length is shared between src and target frame
                focal = 1.1154399414062500e+03

                rr = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1], tgt_res[0], tgt_res[1])
                ir = ir[:, :, rr[1]:rr[1]+rr[3], rr[0]:rr[0]+rr[2]]

            with torch.no_grad():
                start = timeit.default_timer()
                x, msk = model(ir)
                x = x[0, 0, :, :]
                #x = (x - pad) / (1.0 - 2.0 * pad)
                x = x * x.shape[1]
                x_0 = torch.arange(0, x.shape[1]).unsqueeze(0).to(device)
                x -= x_0
                x *= -1.0
                x = x.detach().cpu().numpy()
                stop = timeit.default_timer()
                print('Time: ', stop - start)

                msk = msk.detach().cpu().squeeze().numpy()

                target_path = os.path.join(current_tgt, f"disp{tag}.exr")
                cv2.imwrite(target_path, x)

                target_path = os.path.join(current_tgt, f"msk{tag}.png")
                cv2.imwrite(target_path, msk)
                cv2.imshow("x", x/100)#/640)
                cv2.imshow("msk", msk)
                cv2.waitKey()


backbone = torch.load(backbone_model_pth, map_location=device)
backbone.eval()

regressor = torch.load(regressor_model_pth, map_location=device)
regressor.eval()

model = CompositeModel(backbone, regressor)

model.to(device)
apply_recursively(model, path_src, path_results, "")
