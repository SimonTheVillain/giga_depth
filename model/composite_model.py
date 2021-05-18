import torch
import torch.nn as nn
from torch.cuda.amp import autocast


class CompositeModel(nn.Module):
    def __init__(self, backbone, regressor, half_precision=False):
        super(CompositeModel, self).__init__()
        self.half_precision = half_precision
        self.backbone = backbone
        self.regressor = regressor

        # TODO: remove this debug(or at least make it so it can run with other than 64 channels
        # another TODO: set affine parameters to false!
        # self.bn = nn.BatchNorm2d(64, affine=False)

    def forward(self, x, x_gt=None):

        if x_gt != None:
            if self.half_precision:
                with autocast():
                    x, debugs = self.backbone(x, True)
                x = x.type(torch.float32)
            else:
                x, debugs = self.backbone(x, True)

            results = self.regressor(x, x_gt)
            # todo: batch norm the whole backbone and merge two dicts:
            # https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python-taking-union-o
            # z = {**x, **y}
            for key, val in debugs.items():
                results[-1][key] = val
            return results
        else:

            if self.half_precision:
                with autocast():
                    x = self.backbone(x)
                x = x.type(torch.float32)
            else:
                x = self.backbone(x)
            return self.regressor(x, x_gt)