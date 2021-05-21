import torch
from model.composite_model import CompositeModel
import cv2
import os
import numpy as np

#Work in the parent directory
os.chdir("../")

device = "cuda:0"
regressor_model_pth = "trained_models/shapenet_small_63_regressor.pt"
backbone_model_pth = "trained_models/shapenet_small_63_backbone.pt"
path_src = "/media/simon/ext_ssd/datasets/shapenet_rendered_test_compressed"
path_results = "/media/simon/ext_ssd/datasets/shapenet_rendered_test_results/GigaDepth"


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
        if path[-3:] == 'png' and path[:3] == "im0":
            src = current
            ir = cv2.imread(current, cv2.IMREAD_UNCHANGED)
            ir = ir.astype(np.float32) / 65536.0

            cv2.imshow("ir", ir)
            cv2.waitKey(1)
            ir = torch.tensor(ir).cuda()
            ir = ir.unsqueeze(0).unsqueeze(0)
            dst = os.path.join(current_tgt, path)
            with torch.no_grad():
                x, msk = model(ir)
                x = x[0, 0, :, :]
                x = (x - pad) / (1.0 - 2.0 * pad)
                x = x * x.shape[1]
                x_0 = torch.arange(0, x.shape[1]).unsqueeze(0).cuda()
                x -= x_0
                x *= -1.0
                x = x.detach().cpu().numpy()
                target_path = os.path.join(current_tgt, f"disp{path[2:-4]}.exr")
                cv2.imwrite(target_path, x)


backbone = torch.load(backbone_model_pth, map_location=device)
backbone.eval()

regressor = torch.load(regressor_model_pth, map_location=device)
regressor.eval()

model = CompositeModel(backbone, regressor)

model.cuda()
apply_recursively(model, path_src, path_results, "")
