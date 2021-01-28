import torch
import torch.nn as nn
import torch.nn.functional as F
from model.cuda_cond_mul.cond_mul import CondMul

class Regressor(nn.Module):

    #default parameters are the same as for CR8_reg_cond_mul_2
    def __init__(self, classes=128, height=448, ch_in=128, ch_latent_c=[128, 128], ch_latent_r=[128, 4]):
        super(Regressor, self).__init__()
        self.classes = classes
        self.ch_latent_c = ch_latent_c
        self.ch_latent_r = ch_latent_r
        self.cl1 = nn.Conv2d(ch_in * height, ch_latent_c[0] * height, 1, groups=height)
        #todo: does batch norm also work for these grouped convolutions?
        #self.cl1_bn = nn.BatchNorm2d(ch_latent_c[0])
        self.cl2 = nn.Conv2d(ch_latent_c[0] * height, ch_latent_c[1] * height, 1, groups=height)
        self.cl3 = nn.Conv2d(ch_latent_c[1] * height, (classes + 1) * height, 1, groups=height)

        # only one output (regression!!
        # self.cond_mul = RefCondMulConv(classes, input_features=ch_latent, output_features=1)
        # self.cond_mul = RefCondMul(classes, input_features=ch_latent, output_features=1)

        self.reg1 = nn.Conv2d(ch_in * height, ch_latent_r[0] * height, 1, groups=height)
        #self.reg1_bn = nn.BatchNorm2d(ch_latent_r[0])
        self.reg2_cm = CondMul(classes * height, input_features=ch_latent_r[0], output_features=ch_latent_r[1])
        self.reg3_cm = CondMul(classes * height, input_features=ch_latent_r[1], output_features=1)

    def forward(self, x_in, x_gt=None, mask_gt=None):
        device = x_in.device
        batch_size = x_in.shape[0]
        width = x_in.shape[3]
        height = x_in.shape[2]
        int_type = torch.int32

        # Linewise things:
        #we go from (b, c, h, w) to (b, h, c, w) to (b, h*c, 1, w)
        x = x_in.permute((0, 2, 1, 3)).reshape((batch_size, -1, 1, width))
        x_r = F.leaky_relu(self.reg1(x))#first step of regression is line dependant and will be used later!
        x = F.leaky_relu(self.cl1(x))
        x = F.leaky_relu(self.cl2(x))
        x = self.cl3(x)
        # go back to (b, c, h, w)
        x = x.reshape((batch_size, height, -1, width)).permute((0, 2, 1, 3))
        #print(x.shape)
        # classes = F.softmax(x[:, 0:self.classes, :, :], dim=1)
        classes = x[:, 0:self.classes, :, :]  # cross entropy already has a softmax
        mask = F.leaky_relu(x[:, [-1], :, :])
        # reshaped latent features:
        inds = classes.argmax(dim=1).unsqueeze(1)
        offsets = torch.arange(0, height, device=device).unsqueeze(1).unsqueeze(0).unsqueeze(0) * self.classes
        inds_r = inds + offsets

        # now do the regressions:
        inds_r = inds_r.flatten().type(int_type)

        # from (b, h*c, 1, w) to (b, w, 1, h*c) to (b * w * h, c)
        x = x_r.transpose(1, 3).reshape((-1, self.ch_latent_r[0]))
        x = F.leaky_relu(self.reg2_cm(x, inds_r))
        r = self.reg3_cm(x, inds_r)

        # go from (b * w * h, c) to (b, w, h, c) to (b, 2, h, w)
        r = r.reshape((batch_size, width, height, -1)).permute(0, 3, 2, 1)

        x_real = (inds.type(torch.float32) + r) * (1.0 / float(self.classes))
        if x_gt is None:
            return x_real, mask
        else:
            inds_gt = (x_gt * self.classes).type(torch.int64)
            loss = F.cross_entropy(classes, inds_gt.squeeze(1).clamp(0, self.classes-1))
            class_losses = [torch.mean(loss * mask_gt)]

            inds_r = inds_gt.clamp(0, self.classes - 1) + offsets

            # going from (b, 1, h, w) to (b * h * w)
            inds_r = inds_r.flatten().type(int_type)

            # from (b, h*c, 1, w) to (b, h, c, w) to (b, h, w, c) to (b * h * w, c)
            x = x_r.reshape((batch_size, height, -1, width)).permute((0, 1, 3, 2)).reshape((-1, self.ch_latent_r[0]))
            x = F.leaky_relu(self.reg2_cm(x, inds_r))
            r = self.reg3_cm(x, inds_r)

            # go from (b*h*w, 1) to (b, 1, h, w)
            r = r.reshape((batch_size, 1, height, width))
            x = (inds_gt.type(torch.float32) + r) * (1.0 / float(self.classes))

            if torch.any(torch.isnan(r)):
                print("regressions: found nan")
            if torch.any(torch.isinf(r)):
                print("regressions: found inf")

            return x, mask, class_losses, x_real
