import torch
import torch.nn as nn
import torch.nn.functional as F
from model.cuda_cond_mul.cond_mul import CondMul


class Regressor(nn.Module):

    # default parameters are the same as for CR8_reg_cond_mul_2
    def __init__(self, classes=128, height=448, ch_in=128, ch_latent_c=[128, 128], ch_latent_r=[128, 4]):
        super(Regressor, self).__init__()
        self.classes = classes
        self.ch_latent_c = ch_latent_c
        self.ch_latent_r = ch_latent_r
        self.cl1 = nn.Conv2d(ch_in * height, ch_latent_c[0] * height, 1, groups=height)
        # todo: does batch norm also work for these grouped convolutions?
        # self.cl1_bn = nn.BatchNorm2d(ch_latent_c[0])
        self.cl2 = nn.Conv2d(ch_latent_c[0] * height, ch_latent_c[1] * height, 1, groups=height)
        self.cl3 = nn.Conv2d(ch_latent_c[1] * height, (classes + 1) * height, 1, groups=height)

        # only one output (regression!!
        # self.cond_mul = RefCondMulConv(classes, input_features=ch_latent, output_features=1)
        # self.cond_mul = RefCondMul(classes, input_features=ch_latent, output_features=1)

        self.reg1 = nn.Conv2d(ch_in * height, ch_latent_r[0] * height, 1, groups=height)
        # self.reg1_bn = nn.BatchNorm2d(ch_latent_r[0])
        self.reg2_cm = CondMul(classes * height, input_features=ch_latent_r[0], output_features=ch_latent_r[1])
        self.reg3_cm = CondMul(classes * height, input_features=ch_latent_r[1], output_features=1)

    def forward(self, x_in, x_gt=None, mask_gt=None):
        device = x_in.device
        batch_size = x_in.shape[0]
        width = x_in.shape[3]
        height = x_in.shape[2]
        int_type = torch.int32

        # Linewise things:
        # we go from (b, c, h, w) to (b, h, c, w) to (b, h*c, 1, w)
        x = x_in.permute((0, 2, 1, 3)).reshape((batch_size, -1, 1, width))
        x_r = F.leaky_relu(self.reg1(x))  # first step of regression is line dependant and will be used later!
        x = F.leaky_relu(self.cl1(x))
        x = F.leaky_relu(self.cl2(x))
        x = self.cl3(x)
        # go back to (b, c, h, w)
        x = x.reshape((batch_size, height, -1, width)).permute((0, 2, 1, 3))
        # print(x.shape)
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
            loss = F.cross_entropy(classes, inds_gt.squeeze(1).clamp(0, self.classes - 1))
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


class Regressor2(nn.Module):

    # default parameters are the same as for CR8_reg_cond_mul_2
    def __init__(self, classes=256, superclasses=16, height=448, ch_in=128, ch_latent_c=[128, 128, 128],
                 ch_latent_r=[256, 8]):
        super(Regressor2, self).__init__()
        self.classes = classes
        self.superclasses = superclasses
        self.class_factor = int(classes / superclasses)
        self.ch_latent_c = ch_latent_c
        self.ch_latent_r = ch_latent_r
        self.cl1 = nn.Conv2d(ch_in * height, ch_latent_c[0] * height, 1, groups=height)
        # todo: does batch norm also work for these grouped convolutions?
        # self.cl1_bn = nn.BatchNorm2d(ch_latent_c[0])
        self.cl2 = nn.Conv2d(ch_latent_c[0] * height, ch_latent_c[1] * height, 1, groups=height)
        self.cl3 = nn.Conv2d(ch_latent_c[1] * height, ch_latent_c[2] * height, 1, groups=height)
        self.cl4 = nn.Conv2d(ch_latent_c[2] * height, (classes + 1) * height, 1, groups=height)

        # only one output (regression!!
        self.reg1 = nn.Conv2d(ch_in * height, ch_latent_r[0] * height, 1, groups=height)
        # self.reg1_bn = nn.BatchNorm2d(ch_latent_r[0])
        self.reg2_cm = CondMul(superclasses * height, input_features=ch_latent_r[0], output_features=ch_latent_r[1])
        self.reg3_cm = CondMul(classes * height, input_features=ch_latent_r[1], output_features=1)

    def forward(self, x_in, x_gt=None, mask_gt=None):
        device = x_in.device
        batch_size = x_in.shape[0]
        width = x_in.shape[3]
        height = x_in.shape[2]
        int_type = torch.int32

        # Linewise things:
        # we go from (b, c, h, w) to (b, h, c, w) to (b, h*c, 1, w)
        x = x_in.permute((0, 2, 1, 3)).reshape((batch_size, -1, 1, width))
        x_r = F.leaky_relu(self.reg1(x))  # first step of regression is line dependant and will be used later!
        x = F.leaky_relu(self.cl1(x))
        x = F.leaky_relu(self.cl2(x))
        x = F.leaky_relu(self.cl3(x))
        x = self.cl4(x)
        # go back to (b, c, h, w)
        x = x.reshape((batch_size, height, -1, width)).permute((0, 2, 1, 3))
        # print(x.shape)
        # classes = F.softmax(x[:, 0:self.classes, :, :], dim=1)
        classes = x[:, 0:self.classes, :, :]  # cross entropy already has a softmax
        mask = F.leaky_relu(x[:, [-1], :, :])
        # reshaped latent features:
        inds = classes.argmax(dim=1).unsqueeze(1)
        offsets = torch.arange(0, height, device=device).unsqueeze(1).unsqueeze(0).unsqueeze(0) * self.classes
        offsets_sc = torch.arange(0, height, device=device).unsqueeze(1).unsqueeze(0).unsqueeze(0) * self.superclasses
        inds_r = inds + offsets
        inds_sc = inds // self.class_factor + offsets_sc
        # now do the regressions:
        inds_r = inds_r.flatten().type(int_type)
        inds_sc = inds_sc.flatten().type(int_type)

        # from (b, h*c, 1, w) to (b, w, 1, h*c) to (b * w * h, c)
        x = x_r.transpose(1, 3).reshape((-1, self.ch_latent_r[0]))
        x = F.leaky_relu(self.reg2_cm(x, inds_sc))
        r = self.reg3_cm(x, inds_r)

        # go from (b * w * h, c) to (b, w, h, c) to (b, 2, h, w)
        r = r.reshape((batch_size, width, height, -1)).permute(0, 3, 2, 1)

        x_real = (inds.type(torch.float32) + r) * (1.0 / float(self.classes))
        if x_gt is None:
            return x_real, mask
        else:
            inds_gt = (x_gt * self.classes).type(torch.int64)
            loss = F.cross_entropy(classes, inds_gt.squeeze(1).clamp(0, self.classes - 1))
            class_losses = [torch.mean(loss * mask_gt)]

            inds_r = inds_gt.clamp(0, self.classes - 1) + offsets
            inds_sc = (inds_gt // self.class_factor).clamp(0, self.superclasses - 1) + offsets_sc

            # going from (b, 1, h, w) to (b * h * w)
            inds_r = inds_r.flatten().type(int_type)
            inds_sc = inds_sc.flatten().type(int_type)

            # from (b, h*c, 1, w) to (b, h, c, w) to (b, h, w, c) to (b * h * w, c)
            x = x_r.reshape((batch_size, height, -1, width)).permute((0, 1, 3, 2)).reshape((-1, self.ch_latent_r[0]))
            x = F.leaky_relu(self.reg2_cm(x, inds_sc))
            r = self.reg3_cm(x, inds_r)

            # go from (b*h*w, 1) to (b, 1, h, w)
            r = r.reshape((batch_size, 1, height, width))
            x = (inds_gt.type(torch.float32) + r) * (1.0 / float(self.classes))

            if torch.any(torch.isnan(r)):
                print("regressions: found nan")
            if torch.any(torch.isinf(r)):
                print("regressions: found inf")

            return x, mask, class_losses, x_real


class Classifier3Stage(nn.Module):
    def __init__(self, ch_in=128,
                 height=448,
                 classes=[16, 16, 16],
                 pad=[0, 8, 8],  # pad around classes
                 ch_latent=[[32, 32], [32, 32], [32, 32]]):#todo: make these of variable lengths
        super(Classifier3Stage, self).__init__()
        self.classes = classes
        self.pad = pad
        classes12 = classes[0] * classes[1]
        self.c1 = nn.ModuleList([nn.Conv2d(height * ch_in, height * ch_latent[0][0], 1, groups=height),
                                 CondMul(height * classes[0], ch_in, ch_latent[1][0]),
                                 CondMul(height * classes12, ch_in, ch_latent[2][0])])
        self.c2 = nn.ModuleList([nn.Conv2d(height * ch_latent[0][0], height * ch_latent[0][1], 1, groups=height),
                                 CondMul(height * classes[0], ch_latent[1][0], ch_latent[1][1]),
                                 CondMul(height * classes12, ch_latent[2][0], ch_latent[2][1])])
        self.c3 = nn.ModuleList([nn.Conv2d(height * ch_latent[0][1], height * classes[0], 1),
                                 CondMul(height * classes[0], ch_latent[1][1], classes[1] + 2 * pad[1]),
                                 CondMul(height * classes12, ch_latent[2][1], classes[2] + 2 * pad[2])])

    def forward(self, x_in, inds_gt=None, mask_gt=None):
        bs = x_in.shape[0]  # batch size
        height = x_in.shape[2]
        width = x_in.shape[3]
        device = x_in.device

        offsets = torch.arange(0, height, device=device, dtype=torch.int32).reshape((1, 1, height, 1))

        classes12 = self.classes[0] * self.classes[1]
        classes123 = classes12 * self.classes[2]
        classes23 = self.classes[1] * self.classes[2]

        # STEP 1:
        # convert from (b, c, h, w) to (b, h, c, w) to (b, h*c, 1, w)
        x = x_in.permute((0, 2, 1, 3)).reshape((bs, -1, 1, width))
        x = F.leaky_relu(self.c1[0](x))
        x = F.leaky_relu(self.c2[0](x))
        x = self.c3[0](x)

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
        x = F.leaky_relu(self.c1[1](x_l, inds1_l))
        x = F.leaky_relu(self.c2[1](x, inds1_l))
        x = self.c3[1](x, inds1_l)

        # (b * h * w, c) to (b * h * w, 1) to (b, 1, h, w)
        inds2 = x.argmax(dim=1).reshape((bs, 1, height, width))

        inds12 = inds1 * self.classes[1] + (inds2 - self.pad[1])
        inds12_l = (inds12.clamp(0, classes12 - 1) + classes12 * offsets).type(torch.int32)

        # (b, 1, h, w) to (b * h * w)
        inds12_l = inds12_l.flatten()

        # STEP 3:
        x = F.leaky_relu(self.c1[2](x_l, inds12_l))
        x = F.leaky_relu(self.c2[2](x, inds12_l))
        x = self.c3[2](x, inds12_l)

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
            loss = F.cross_entropy(x1, inds1_gt.squeeze(1).type(torch.int64)).mean()
            losses.append(loss)
            inds1_gt = inds1_gt
            # also select the neighbouring superclasses
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
                x = F.leaky_relu(self.c1[1](x_l, inds1_l))
                x = F.leaky_relu(self.c2[1](x, inds1_l))
                x = self.c3[1](x, inds1_l)
                # from (b * h * w, c) to (b, h, w, c) to (b, c, h, w)
                x = x.reshape((bs, height, width, -1)).permute((0, 3, 1, 2))
                loss = F.cross_entropy(x, inds2_gt) * mask
                losses.append(loss.mean())
                torch.cuda.synchronize()

            inds12_gt = inds_gt // self.classes[2]
            inds12_gt = inds12_gt
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

                inds12_l = (inds12_l + classes12 * offsets).flatten().type(torch.int32)
                x = F.leaky_relu(self.c1[2](x_l, inds12_l))
                x = F.leaky_relu(self.c2[2](x, inds12_l))
                x = self.c3[2](x, inds12_l)
                # from (b * h * w, c) to (b, h, w, c) to (b, c, h, w)
                x = x.reshape((bs, height, width, -1)).permute((0, 3, 1, 2))
                loss = F.cross_entropy(x, inds3_gt) * mask
                losses.append(loss.mean())
                torch.cuda.synchronize()

            return inds123_real, losses


# same as #5 but without batch normalization
class Reg_3stage(nn.Module):

    # default parameters are the same as for CR8_reg_cond_mul_2
    def __init__(self, ch_in=128,
                 height=448,
                 ch_latent=[128, 128, 128],#todo: make this of variable length
                 superclasses=8,
                 ch_latent_r=[128, 32],
                 ch_latent_msk=[32, 16],
                 classes=[16, 16, 16],
                 pad=[0, 8, 8],
                 ch_latent_c=[[32, 32], [32, 32], [32, 32]],#todo: make these of variable length
                 regress_neighbours=0):
        super(Reg_3stage, self).__init__()
        classes123 = classes[0] * classes[1] * classes[2]
        self.classes = classes
        self.height = height
        self.superclasses = superclasses
        self.class_factor = int(classes123 / superclasses)
        self.regress_neighbours = regress_neighbours
        # the first latent layer for classification is shared
        self.bb = nn.ModuleList()
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
                                  ch_latent=ch_latent_c)

        # only one output (regression!!
        # self.cond_mul = RefCondMulConv(classes, input_features=ch_latent, output_features=1)
        # self.cond_mul = RefCondMul(classes, input_features=ch_latent, output_features=1)

        #todo: is it really the best using the raw input here. maybe we use the per line backbone?
        self.r1 = nn.Conv2d(height * ch_in, height * ch_latent_r[0], 1, groups=height)
        self.r2 = CondMul(height * superclasses, ch_latent_r[0], ch_latent_r[1])
        self.r3 = CondMul(height * classes123, ch_latent_r[1], 1)

        # kernels for masks:
        self.msk = nn.ModuleList()
        #todo: is it really the best using the raw input here. maybe we use the per line backbone?
        ch_latent_msk.insert(0, ch_in)
        ch_latent_msk.append(1)
        for i in range(0, len(ch_latent_msk) - 1):
            self.msk.append(nn.Conv2d(height * ch_latent_msk[i], height * ch_latent_msk[i + 1], 1, groups=height))

        #self.msk1 = nn.Conv2d(height * ch_in, height * ch_latent_msk[0], 1, groups=height)
        #self.msk2 = nn.Conv2d(height * ch_latent_msk[0], height * ch_latent_msk[1], 1, groups=height)
        #self.msk3 = nn.Conv2d(height * ch_latent_msk[1], height, 1, groups=height)

    def forward(self, x_in, x_gt=None, mask_gt=None):
        height = x_in.shape[2]
        width = x_in.shape[3]
        classes123 = self.classes[0] * self.classes[1] * self.classes[2]
        bs = x_in.shape[0]
        int_type = torch.int32
        device = x_in.device

        # reshape from (b, c, h, w) to (b, h, c, w) to (b, h * c, 1, w)
        x_in = x_in.permute((0, 2, 1, 3)).reshape((bs, -1, 1, width))
        # the first stage is to adapt to features to something that has meaning on this line!
        x = x_in
        for node in self.bb:
            x = F.leaky_relu(node(x))
        x_l = x
        #convert from (b, h * c, 1, w) to (b, h, c, w) to (b, c, h, w)
        x_l = x_l.reshape((bs, height, -1, width)).permute((0, 2, 1, 3))

        # calculate the mask/confidence on these lines
        x = x_in
        for node in self.msk:
            x = F.leaky_relu(node(x))
        mask = x

        # reshape from (b, h * c, 1, w) (c=1) to (b, h, c, w) to (b, c, h, w)
        # or in short from (b, h, 1, w) to (b, 1, h, w)
        mask = mask.reshape((bs, 1, height, -1))

        # create vector with index offsets along the vertical dimension (1, 1, h, 0)
        ind_offsets = torch.arange(0, height, device=device).unsqueeze(1).unsqueeze(0).unsqueeze(0)
        if x_gt is None:

            # the input for the classifier (as well as the output) should come in (b, c, h, w)
            inds = self.c(x_l)
            inds_super = inds // self.class_factor + ind_offsets * self.superclasses
            inds_l = inds + ind_offsets * classes123
            inds_super = inds_super.flatten().type(torch.int32)
            inds_l = inds_l.flatten().type(torch.int32)


            x = F.leaky_relu(self.r1(x_in))
            # from (b, h * c, 1, w) to (b, h, c, w) to (b * h * w, c)
            x_l = x.reshape((bs, height, -1, width)).permute((0, 1, 3, 2))
            x_l = x_l.reshape((bs * height * width, -1)).contiguous()
            x = F.leaky_relu(self.r2(x_l, inds_super))
            r = self.r3(x, inds_l).flatten()

            x = (inds.flatten().type(torch.float32) + r) * (1.0 / float(classes123))
            x = x.reshape((bs, 1, height, width))
            return x, mask
        else:

            inds_gt = (x_gt * classes123).type(torch.int32).clamp(0, classes123 - 1)
            inds, class_losses = self.c(x_l, inds_gt, mask_gt)

            # todo: change this for multiline!
            x = F.leaky_relu(self.r1(x_in))
            # from (b, h * c, 1, w) to (b, h, c, w) to (b * h * w, c)
            x_l = x.reshape((bs, height, -1, width)).permute((0, 1, 3, 2))
            x_l = x_l.reshape((bs * height * width, -1)).contiguous()

            # calculate the regression only x
            x_reg_combined = torch.zeros((bs, 1 + 2 * self.regress_neighbours, height, width),
                                         device=device)
            for offset in range(-self.regress_neighbours, self.regress_neighbours+1):
                inds_gt = (inds_gt + offset).clamp(0, classes123 - 1)
                inds_super = inds_gt // self.class_factor
                inds_super = (inds_super + self.superclasses * ind_offsets).flatten().type(torch.int32)
                inds_l = (inds_gt + classes123 * ind_offsets).flatten().type(torch.int32)

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
            inds_super = (inds_super + self.superclasses * ind_offsets).flatten().type(torch.int32)
            inds_l = (inds + classes123 * ind_offsets).flatten().type(torch.int32)
            # STEP 1
            x = F.leaky_relu(self.r2(x_l, inds_super))
            # STEP 3 + reshape
            # from (b * h * w, 1) to (b, 1, h, w)
            r = self.r3(x, inds_l).reshape((bs, 1, height, width))

            x = (inds.type(torch.float32) + r) * (1.0 / float(classes123))
            x_real = x.reshape((bs, 1, height, width))
            torch.cuda.synchronize()

            return x_reg_combined, mask, class_losses, x_real
