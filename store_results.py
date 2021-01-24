import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from dataset.dataset_rendered_2 import DatasetRendered2
from experiments.lines.model_lines_CR8_n import *

import numpy as np



class CompositeModel(nn.Module):
    def __init__(self, backbone, regressor):
        super(CompositeModel, self).__init__()
        self.backbone = backbone
        self.regressor = regressor

    def forward(self, x, x_gt=None, mask_gt=None):
        x = self.backbone(x)
        return self.regressor(x, x_gt, mask_gt)


target_path = "/media/simon/ssd_data/data/datasets/structure_core_unity_slice_100_35_results"
dataset_path = "/media/simon/ssd_data/data/datasets/structure_core_unity_slice_100_35"
tgt_res = (1216, 896)#(1216, 896)

lines_only = True
is_npy = True

regressor = "trained_models/cr8_2021_32_std_5_regressor_chk.pt"
backbone = "trained_models/cr8_2021_32_std_5_backbone_chk.pt"

#regressor = "trained_models/cr8_2021_regressor_chk.pt"
#backbone = "trained_models/cr8_2021_backbone_chk.pt"

backbone = torch.load(backbone)
backbone.eval()

regressor = torch.load(regressor)
regressor.eval()

device = torch.cuda.current_device()

model = CompositeModel(backbone, regressor)
model.to(device)
model.eval()
dataset = DatasetRendered2(dataset_path, 0, 42000, tgt_res=tgt_res, is_npy=is_npy)

for i, data in enumerate(dataset):
    print(i)
    ir, x_gt, mask_gt = data
    with torch.no_grad():
        ir = torch.tensor(ir, device=device).unsqueeze(0)
        x, sigma_sq = model(ir)
        #print(x)
        x = x.cpu().numpy()
        np.save(f"{target_path}/{i}.npy", x)