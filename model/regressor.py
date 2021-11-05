import torch
import torch.nn as nn
import torch.nn.functional as F
from model.cuda_cond_mul.cond_mul import CondMul
import numpy as np


class RegressorConv(nn.Module):
    def __init__(self, ch_in=64,
                 ch_latent=[64, 64, 128, 1024, 1024, 128],
                 ch_latent_msk=[64, 64],
                 slices=4,
                 vertical_fourier_encoding=8):
        super(RegressorConv, self).__init__()
        self.slices = nn.ModuleList()

        self.slices_msk = nn.ModuleList()

        #TODO: fourier encoding
        self.fourier_encoding = vertical_fourier_encoding

        ch_latent.insert(0, ch_in + self.fourier_encoding * 2)
        ch_latent_msk.insert(0, ch_in)
        for i in range(slices):
            slice = nn.Sequential()
            for j in range(len(ch_latent)-1):
                slice.add_module(f"conv{j}", nn.Conv2d(ch_latent[j], ch_latent[j+1], 1))
                slice.add_module(f"bn{j}", nn.BatchNorm2d(ch_latent[j+1]))
                slice.add_module(f"ReLU{j}", nn.LeakyReLU())
            #the output is a simple regression:
            slice.add_module("conv_out", nn.Conv2d(ch_latent[-1], 1, 1))
            self.slices.append(slice)

            slice = nn.Sequential()
            for j in range(len(ch_latent_msk) - 1):
                slice.add_module(f"conv{j}", nn.Conv2d(ch_latent_msk[j], ch_latent_msk[j+1], 1))
                slice.add_module(f"bn{j}", nn.BatchNorm2d(ch_latent_msk[j+1]))
                slice.add_module(f"ReLU{j}", nn.LeakyReLU())
            slice.add_module("conv_out", nn.Conv2d(ch_latent_msk[-1], 1, 1))
            self.slices_msk.append(slice)                                 

    def append_vertical_fourier_encoding(self, x):
        device = x.device
        features = torch.zeros(x.shape[0], self.fourier_encoding * 2, x.shape[2], x.shape[3], device=device)
        y = torch.arange(0, x.shape[2]).unsqueeze(1).unsqueeze(0).unsqueeze(0)
        for i in range(self.fourier_encoding):
            features[:, i * 2, :, :] = torch.sin(y * ((i + 1) * np.pi / x.shape[2]))
            features[:, i * 2 + 1, :, :] = torch.cos( y * ((i + 1) * np.pi / x.shape[2]))
        x = torch.cat((x, features), dim=1)
        return x

    def forward(self, x):
        sh = x.shape[2] // len(self.slices)
        regression = []
        mask = []
        for i in range(len(self.slices)):
            x_slice = x[:, :, sh * i:sh * (i + 1), :]
            x_fourier = self.append_vertical_fourier_encoding(x_slice)
            regression.append(self.slices[i](x_fourier))
            mask.append(self.slices_msk[i](x_slice))

        regression = torch.cat(regression, dim=2)
        mask = torch.cat(mask, dim=2)

        return regression, mask


class Classifier3Stage(nn.Module):
    def __init__(self, ch_in=128,
                 height=448,
                 classes=[16, 16, 16],
                 pad=[0, 8, 8],  # pad around classes
                 ch_latent=[[32, 32], [32, 32], [32, 32]],
                 c3_line_div=1,
                 close_far_separation=False):#todo: make these of variable lengths
        super(Classifier3Stage, self).__init__()
        self.classes = classes
        self.pad = pad
        self.close_far_separation = close_far_separation
        classes12 = classes[0] * classes[1]

        #todo: proper loop and encapsulation of the 3

        if close_far_separation:
            ch_latent[0].insert(0, int(ch_in/2))
            ch_latent[2].insert(0, int(ch_in/2))
        else:
            ch_latent[0].insert(0, ch_in)
            ch_latent[2].insert(0, ch_in)

        ch_latent[1].insert(0, ch_in)

        ch_latent[0].append(classes[0] + 2 * pad[0])
        ch_latent[1].append(classes[1] + 2 * pad[1])
        ch_latent[2].append(classes[2] + 2 * pad[2])
        self.c1 = nn.ModuleList()
        self.c2 = nn.ModuleList()
        self.c3 = nn.ModuleList()
        for i in range(0, len(ch_latent[0]) - 1):
            self.c1.append(nn.Conv2d(height * ch_latent[0][i], height * ch_latent[0][i + 1], 1, groups=height))

        for i in range(0, len(ch_latent[1]) - 1):
            self.c2.append(CondMul(height * classes[0], ch_latent[1][i], ch_latent[1][i + 1]))

        self.c3_line_div = c3_line_div
        heights = [int(height / c3_line_div)] * (len(ch_latent[2]) - 1)
        heights[-1] = height
        for i in range(0, len(ch_latent[2]) - 1):
            self.c3.append(CondMul(heights[i] * classes12, ch_latent[0][i], ch_latent[1][i + 1]))

    def get_mean_weights(self):
        mean_weights = {}
        accu = 0
        for c in self.c1:
            accu += c.weight.abs().mean()
        mean_weights["mean_weight_c1"] = accu
        accu = 0
        for c in self.c2:
            accu += c.w.abs().mean()
        mean_weights["mean_weight_c2"] = accu
        accu = 0
        for c in self.c3:
            accu += c.w.abs().mean()
        mean_weights["mean_weight_c3"] = accu
        return mean_weights


    def get_close_segment(self, x_in):
        if self.close_far_separation:
            # the second half of channelss dedicated to close features
            x = x_in[:, int(x_in.shape[1] / 2):, :, :]
        else:
            x = x_in
        # convert from (b, c, h, w) to (b, h, w, c) to (b * h * w, c)
        return x.permute((0, 2, 3, 1)).reshape((-1, x.shape[1])).contiguous()

    def get_far_segment(self, x_in):
        if self.close_far_separation:
            # the first half of channels is dedicated to high-level features
            x = x_in[:, :int(x_in.shape[1] / 2), :, :]
        else:
            x = x_in
        # convert from (b, c, h, w) to (b, h, c, w) to (b, h*c, 1, w)
        x = x.permute((0, 2, 1, 3)).reshape((x_in.shape[0], -1, 1, x_in.shape[3]))
        return x

    def forward(self, x_in, inds_gt=None):
        bs = x_in.shape[0]  # batch size
        height = x_in.shape[2]
        width = x_in.shape[3]
        device = x_in.device

        offsets = torch.arange(0, height, device=device, dtype=torch.int32).reshape((1, 1, height, 1))

        classes12 = self.classes[0] * self.classes[1]
        classes123 = classes12 * self.classes[2]
        classes23 = self.classes[1] * self.classes[2]

        # STEP 1:
        x = self.get_far_segment(x_in)
        # convert from (b, c, h, w) to (b, h, c, w) to (b, h*c, 1, w)
        #x = x.permute((0, 2, 1, 3)).reshape((bs, -1, 1, width))
        for i in range(0, len(self.c1)):
            x = self.c1[i](x)
            if i < len(self.c1) - 1:
                x = F.leaky_relu(x)
        #x = F.leaky_relu(self.c1[0](x))
        #x = F.leaky_relu(self.c2[0](x))
        #x = self.c3[0](x)

        #convert from (b, h*c, 1, w) to (b, h, c, w) to (b, c, h, w)
        x = x.reshape(bs, height, -1, width).permute((0, 2, 1, 3))
        x1 = x
        inds1 = x.argmax(dim=1).unsqueeze(1)
        inds1_l = inds1.type(torch.int32) + self.classes[0] * offsets# add offset for each line!

        # STEP 2:
        # convert from (b, 1, h, w) to (b * h * w)
        inds1_l = inds1_l.flatten()
        # convert from (b, c, h, w) to (b, h, w, c) to (b * h * w, c)
        x_l = x_in.permute((0, 2, 3, 1)).reshape((-1, x_in.shape[1])).contiguous()
        x = x_l
        for i in range(0, len(self.c2)):
            x = self.c2[i](x, inds1_l)
            if i < len(self.c2) - 1:
                x = F.leaky_relu(x)
        #x = F.leaky_relu(self.c1[1](x_l, inds1_l))
        #x = F.leaky_relu(self.c2[1](x, inds1_l))
        #x = self.c3[1](x, inds1_l)

        # (b * h * w, c) to (b * h * w, 1) to (b, 1, h, w)
        inds2 = x.argmax(dim=1).reshape((bs, 1, height, width))

        inds12 = inds1 * self.classes[1] + (inds2 - self.pad[1])
        inds12_l = inds12.clamp(0, classes12 - 1) + classes12 * offsets
        inds12_l_scaled_lines = inds12.clamp(0, classes12 - 1) + classes12 * (offsets // self.c3_line_div)

        # (b, 1, h, w) to (b * h * w)
        inds12_l = inds12_l.type(torch.int32).flatten()
        inds12_l_scaled_lines = inds12_l_scaled_lines.type(torch.int32).flatten()



        # STEP 3:
        x = self.get_close_segment(x_in)
        for i in range(0, len(self.c3)):
            if i < len(self.c3) - 1:
                x = F.leaky_relu(self.c3[i](x, inds12_l_scaled_lines))
            else:
                x = self.c3[i](x, inds12_l)

        #x = F.leaky_relu(self.c1[2](x_l, inds12_l))
        #x = F.leaky_relu(self.c2[2](x, inds12_l))
        #x = self.c3[2](x, inds12_l)

        # (b * h * w, c) to (b * h * w, 1) to (b, 1, h, w)
        inds3 = x.argmax(dim=1).reshape((bs, 1, height, width))

        inds123_real = inds12 * self.classes[2] + (inds3 - self.pad[2])
        inds123_real = inds123_real.clamp(0, classes123 - 1) # due to padding the clamping might be necessary
        if inds_gt is None:
            return inds123_real
        else:
            losses = []
            inds_gt = inds_gt.clamp(0, classes123 - 1)
            inds1_gt = inds_gt // classes23
            loss = F.cross_entropy(x1, inds1_gt.squeeze(1).type(torch.int64), reduction='none')# .mean()
            losses.append(loss)
            inds1_gt = inds1_gt
            # also select the neighbouring superclasses
            loss_sum = 0
            for i in [-1, 0, 1]:
                # calculate the index of this class/ its neighbours
                inds1_l = inds1_gt + i
                inds1_l = inds1_l.clamp(0, self.classes[0] - 1)
                # calculate the local groundtruth index
                inds2_gt = inds_gt // self.classes[2] - inds1_l * self.classes[1]
                inds2_gt = inds2_gt + self.pad[1]

                # the mask masks out where this would not yield any valid samples
                mask = torch.logical_and(inds2_gt >= 0, inds2_gt < (self.classes[1] + 2 * self.pad[1]))
                inds2_gt = inds2_gt.clamp(0, self.classes[1] + 2 * self.pad[1] - 1).squeeze(1).type(torch.int64)

                inds1_l = (inds1_l + self.classes[0] * offsets).flatten().type(torch.int32)
                x = x_l
                for j in range(0, len(self.c2)):
                    x = self.c2[j](x, inds1_l)
                    if i < len(self.c2) - 1:
                        x = F.leaky_relu(x)
                #x = F.leaky_relu(self.c1[1](x_l, inds1_l))
                #x = F.leaky_relu(self.c2[1](x, inds1_l))
                #x = self.c3[1](x, inds1_l)
                # from (b * h * w, c) to (b, h, w, c) to (b, c, h, w)
                x = x.reshape((bs, height, width, -1)).permute((0, 3, 1, 2))
                loss = F.cross_entropy(x, inds2_gt, reduction='none') * mask
                loss_sum += loss# .mean()
                #torch.cuda.synchronize()
            losses.append(loss_sum)

            inds12_gt = inds_gt // self.classes[2]
            inds12_gt = inds12_gt
            loss_sum = 0
            for i in [-1, 0, 1]:
                # calculate the index of this class/ its neighbours
                inds12_l = inds12_gt + i
                inds12_l = inds12_l.clamp(0, classes12 - 1)
                # calculate the local groundtruth index
                inds3_gt = inds_gt - inds12_l * self.classes[2]
                inds3_gt = inds3_gt + self.pad[2]

                # the mask masks out where this does not yield any valid samples
                mask = torch.logical_and(inds3_gt >= 0, inds3_gt < (self.classes[2] + 2 * self.pad[2]))
                inds3_gt = inds3_gt.clamp(0, self.classes[2] + 2 * self.pad[2] - 1).squeeze(1).type(torch.int64)

                inds12_l = (inds12_l + classes12 * offsets).type(torch.int32).flatten()
                inds12_l_scaled_lines = inds12.clamp(0, classes12 - 1) + classes12 * (offsets // self.c3_line_div)
                inds12_l_scaled_lines = inds12_l_scaled_lines.type(torch.int32).flatten()

                x = self.get_close_segment(x_in)
                for i in range(0, len(self.c3)):
                    if i < len(self.c3) - 1:
                        x = F.leaky_relu(self.c3[i](x, inds12_l_scaled_lines))
                    else:
                        x = self.c3[i](x, inds12_l)
                #x = F.leaky_relu(self.c1[2](x_l, inds12_l))
                #x = F.leaky_relu(self.c2[2](x, inds12_l))
                #x = self.c3[2](x, inds12_l)
                # from (b * h * w, c) to (b, h, w, c) to (b, c, h, w)
                x = x.reshape((bs, height, width, -1)).permute((0, 3, 1, 2))
                loss = F.cross_entropy(x, inds3_gt, reduction='none') * mask
                loss_sum += loss# .mean()
                #torch.cuda.synchronize()
            losses.append(loss_sum)
            return inds123_real, losses


# same as #5 but without batch normalization
class Reg_3stage(nn.Module):

    # default parameters are the same as for CR8_reg_cond_mul_2
    def __init__(self, ch_in=128,
                 height=448,
                 ch_latent=[128, 128, 128],  #these are of variable length
                 superclasses=8,
                 ch_latent_r=[128, 32],  #TODO: check if the first (or even both) of these layers is unnecessary.
                 ch_latent_msk=[32, 16],  #todo:make these of variable length?
                 classes=[16, 16, 16],
                 pad=[0, 8, 8],
                 ch_latent_c=[[32, 32], [32, 32], [32, 32]],  #these are of variable length
                 regress_neighbours=0,
                 reg_line_div=1,
                 c3_line_div=1,
                 close_far_separation=False, # split input up in high and low half and feed low->c1&c2 and high->c2/c3/r
                 sigma_mode="line",
                 vertical_slices=4,
                 pad_proj=0.1): # "conv", "line", "class"
        super(Reg_3stage, self).__init__()
        if c3_line_div == 1 and len(ch_latent) != 0:
            print("You can't share weights between lines the classification c3 if there is  a per line backbone( "
                  "h_latent).")
            c3_line_div = 1
        classes123 = classes[0] * classes[1] * classes[2]
        self.classes = classes
        self.height = height
        self.superclasses = superclasses
        self.class_factor = int(classes123 / superclasses)
        self.regress_neighbours = regress_neighbours
        self.reg_line_div = reg_line_div
        self.vertical_slices = vertical_slices
        self.pad_proj = pad_proj

        # the first latent layer for classification is shared
        self.bb = nn.ModuleList()
        assert len(ch_latent) == 0 #something was wrong the last time i used ch_latent! Take a look before using it!
        ch_latent.insert(0, ch_in)
        for i in range(0, len(ch_latent) - 1):
            self.bb.append(nn.Conv2d(height * ch_latent[i], height * ch_latent[i+1], 1, groups=height))
        #self.bb1 = nn.Conv2d(height * ch_in, height * ch_latent[0], 1, groups=height)
        #self.bb2 = nn.Conv2d(height * ch_latent[0], height * ch_latent[1], 1, groups=height)
        #self.bb3 = nn.Conv2d(height * ch_latent[1], height * ch_latent[2], 1, groups=height)

        self.c = Classifier3Stage(ch_in=ch_latent[-1],
                                  height=height,
                                  classes=classes,
                                  pad=pad,  # pad around classes
                                  ch_latent=ch_latent_c,
                                  c3_line_div=c3_line_div,
                                  close_far_separation=close_far_separation)

        # only one output (regression!!
        # self.cond_mul = RefCondMulConv(classes, input_features=ch_latent, output_features=1)
        # self.cond_mul = RefCondMul(classes, input_features=ch_latent, output_features=1)

        #todo: is it really the best using the raw input here. maybe we use the per line backbone?
        #self.r1 = nn.Conv2d(height * ch_in, height * ch_latent_r[0], 1, groups=height)
        #self.r2 = CondMul(height * superclasses, ch_latent_r[0], ch_latent_r[1])
        #self.r3 = CondMul(height * classes123, ch_latent_r[1], 1)
        if close_far_separation:
            self.r2 = CondMul(int(height / reg_line_div) * superclasses, int(ch_in/2), ch_latent_r[0])
        else:
            self.r2 = CondMul(int(height / reg_line_div) * superclasses, ch_in, ch_latent_r[0])
        self.r3 = CondMul(height * classes123, ch_latent_r[0], 1)

        # kernels for masks:
        #todo: is it really the best using the raw input here. maybe we use the per line backbone?
        ch_latent_msk.insert(0, ch_in)
        ch_latent_msk.append(1)
        self.sigma_mode = sigma_mode
        if sigma_mode == "line":
            self.msk = nn.ModuleList()
            for i in range(0, len(ch_latent_msk) - 1):
                self.msk.append(nn.Conv2d(height * ch_latent_msk[i], height * ch_latent_msk[i + 1], 1, groups=height))
        if sigma_mode == "conv":
            if vertical_slices != 1:
                self.msk = nn.ModuleList()
                for j in range(vertical_slices):
                    self.msk.append(nn.ModuleList())
                    for i in range(0, len(ch_latent_msk) - 1):
                        self.msk[j].append(nn.Conv2d(ch_latent_msk[i], ch_latent_msk[i + 1], 1))
            else:
                self.msk = nn.ModuleList()
                for i in range(0, len(ch_latent_msk) - 1):
                    self.msk.append(nn.Conv2d(ch_latent_msk[i], ch_latent_msk[i + 1], 1))
        if sigma_mode == "class":
            self.msk = nn.ModuleList()
            for i in range(0, len(ch_latent_msk) - 1):
                self.msk.append(CondMul(height * classes123, ch_latent_msk[i], ch_latent_msk[i + 1]))


    def forward(self, x_in, x_gt=None):

        #scale the groundtruth that -pad_proj to 1.0 + pad_proj
        #are scaled to between 0.0 and 1.0
        if x_gt is not None:
            x_gt = x_gt * (1.0 - 2.0 * self.pad_proj) + self.pad_proj

        height = x_in.shape[2]
        width = x_in.shape[3]
        classes123 = self.classes[0] * self.classes[1] * self.classes[2]
        bs = x_in.shape[0]
        int_type = torch.int32
        device = x_in.device
        x_for_r = self.c.get_close_segment(x_in)

        # reshape from (b, c, h, w) to (b, h, c, w) to (b, h * c, 1, w)
        x_in = x_in.permute((0, 2, 1, 3)).reshape((bs, -1, 1, width)) # todo: maybe don't overwrite x_in here!
        # the first stage is to adapt to features to something that has meaning on this line!
        x = x_in
        for node in self.bb:
            x = F.leaky_relu(node(x))
        x_l = x
        #convert from (b, h * c, 1, w) to (b, h, c, w) to (b, c, h, w)
        x_l = x_l.reshape((bs, height, -1, width)).permute((0, 2, 1, 3))

        if self.sigma_mode == "line":
            # calculate the mask/confidence on these lines
            x = x_in
            for node in self.msk:
                x = F.leaky_relu(node(x))
            mask = x

            # reshape from (b, h * c, 1, w) (c=1) to (b, h, c, w) to (b, c, h, w)
            # or in short from (b, h, 1, w) to (b, 1, h, w)
            mask = mask.reshape((bs, 1, height, -1))
        if self.sigma_mode == "conv":
            # todo: clean up the code around this so that we don't need to reshape here
            # convert from (b, h * c, 1, w) to (b, h, c, w) to (b, c, h, w)
            x = x_in.reshape(bs, height, -1, width).permute((0, 2, 1, 3))
            if self.vertical_slices != 1:
                # todo: clean up the code in here
                msks = [] # slices of the mask
                slice_height = x.shape[2] // self.vertical_slices
                for i in range(self.vertical_slices):
                    x_slice = x[:, :, i * slice_height: (i + 1) * slice_height, :]
                    for node in self.msk[i]:
                        x_slice = F.leaky_relu(node(x_slice))
                    msks.append(x_slice)
                x = torch.cat(msks, dim=2)
            else:
                for node in self.msk:
                    x = F.leaky_relu(node(x))
            mask = x

        # create vector with index offsets along the vertical dimension (1, 1, h, 0)
        line_offsets = torch.arange(0, height, device=device).unsqueeze(1).unsqueeze(0).unsqueeze(0)
        if x_gt is None:

            # the input for the classifier (as well as the output) should come in (b, c, h, w)
            inds = self.c(x_l)
            inds_super = inds // self.class_factor + self.superclasses * (line_offsets // self.reg_line_div)
            inds_l = inds + line_offsets * classes123
            inds_super = inds_super.flatten().type(torch.int32)
            inds_l = inds_l.flatten().type(torch.int32)


            #x = F.leaky_relu(self.r1(x_in))
            # from (b, h * c, 1, w) to (b, h, c, w) to (b * h * w, c)
            #x_l = x_in.reshape((bs, height, -1, width)).permute((0, 1, 3, 2))
            #x_l = x_l.reshape((bs * height * width, -1)).contiguous()
            x_l = x_for_r
            x = F.leaky_relu(self.r2(x_l, inds_super))
            r = self.r3(x, inds_l).flatten()

            x = (inds.flatten().type(torch.float32) + r) * (1.0 / float(classes123))
            x_real = x.reshape((bs, 1, height, width))

            if self.sigma_mode == "class":
                # from (b, h * c, 1, w) to (b, h, c, w) to (b * h * w, c)
                x_l = x_in.reshape((bs, height, -1, width)).permute((0, 1, 3, 2))
                x_l = x_l.reshape((bs * height * width, -1)).contiguous()
                x = x_l
                for node in self.msk:
                    x = F.leaky_relu(node(x, inds_l))

                # from (b * h * w, 1) to (b, 1, h, w)
                mask = x.reshape((bs, 1, height, width))

            #undo the scaling that is made at the beginning
            x_real = (x_real - self.pad_proj) / (1.0 - 2.0 * self.pad_proj)
            return x_real, mask
        else:

            inds_gt = (x_gt * classes123).type(torch.int32).clamp(0, classes123 - 1)
            inds, class_losses = self.c(x_l, inds_gt)

            #x = F.leaky_relu(self.r1(x_in))
            # from (b, h * c, 1, w) to (b, h, c, w) to (b * h * w, c)
            #x_l = x_in.reshape((bs, height, -1, width)).permute((0, 1, 3, 2))
            #x_l = x_l.reshape((bs * height * width, -1)).contiguous()
            x_l = x_for_r

            # calculate the regression only x
            x_reg_combined = torch.zeros((bs, 1 + 2 * self.regress_neighbours, height, width),
                                         device=device)
            for offset in range(-self.regress_neighbours, self.regress_neighbours+1):
                inds_gt = (inds_gt + offset).clamp(0, classes123 - 1)
                inds_super = inds_gt // self.class_factor
                inds_super = inds_super + self.superclasses * (line_offsets // self.reg_line_div)
                inds_super = inds_super.flatten().type(torch.int32)
                inds_l = (inds_gt + classes123 * line_offsets).flatten().type(torch.int32)

                # STEP 2
                x = F.leaky_relu(self.r2(x_l, inds_super))
                # STEP 3 + reshape
                # from (b * h * w, 1) to (b, 1, h, w)
                r = self.r3(x, inds_l).reshape((bs, 1, height, width))
                #r = self.r2(x_l, inds_gt).flatten()#todo:remove this reactivate the two lines above
                x_reg = (inds_gt.type(torch.float32) + r) * (1.0 / float(classes123))
                x_reg = x_reg.reshape((bs, 1, height, width))
                x_reg_combined[:, [offset+self.regress_neighbours], :, :] = x_reg
            # calculate the real x
            inds_super = inds // self.class_factor
            inds_super = inds_super + self.superclasses * (line_offsets // self.reg_line_div)
            inds_super = inds_super.flatten().type(torch.int32)
            inds_l = (inds + classes123 * line_offsets).flatten().type(torch.int32)
            # STEP 1
            x = F.leaky_relu(self.r2(x_l, inds_super))
            # STEP 3 + reshape
            # from (b * h * w, 1) to (b, 1, h, w)
            r = self.r3(x, inds_l).reshape((bs, 1, height, width))

            x = (inds.type(torch.float32) + r) * (1.0 / float(classes123))
            x_real = x.reshape((bs, 1, height, width))
            #torch.cuda.synchronize()

            #lets check if the weights
            debugs = self.c.get_mean_weights()
            debug_r = self.r2.w.abs().mean() + self.r3.w.abs().mean()
            debugs["mean_w_reg"] = debug_r

            if self.sigma_mode == "class":
                # from (b, h * c, 1, w) to (b, h, c, w) to (b * h * w, c)
                x_l = x_in.reshape((bs, height, -1, width)).permute((0, 1, 3, 2))
                x_l = x_l.reshape((bs * height * width, -1)).contiguous()
                x = x_l
                for node in self.msk:
                    x = F.leaky_relu(node(x, inds_l))

                # from (b * h * w, 1) to (b, 1, h, w)
                mask = x.reshape((bs, 1, height, width))

            #undo the scaling that is made at the beginning of this function
            x_real = (x_real - self.pad_proj) / (1.0 - 2.0 * self.pad_proj)
            x_reg_combined = (x_reg_combined - self.pad_proj) / (1.0 - 2.0 * self.pad_proj)
            return x_reg_combined, mask, class_losses, x_real, debugs
