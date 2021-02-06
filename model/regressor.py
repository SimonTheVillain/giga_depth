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


class Regressor2(nn.Module):

    #default parameters are the same as for CR8_reg_cond_mul_2
    def __init__(self, classes=256, superclasses=16, height=448, ch_in=128, ch_latent_c=[128, 128, 128], ch_latent_r=[256, 8]):
        super(Regressor2, self).__init__()
        self.classes = classes
        self.superclasses = superclasses
        self.class_factor = int(classes/superclasses)
        self.ch_latent_c = ch_latent_c
        self.ch_latent_r = ch_latent_r
        self.cl1 = nn.Conv2d(ch_in * height, ch_latent_c[0] * height, 1, groups=height)
        #todo: does batch norm also work for these grouped convolutions?
        #self.cl1_bn = nn.BatchNorm2d(ch_latent_c[0])
        self.cl2 = nn.Conv2d(ch_latent_c[0] * height, ch_latent_c[1] * height, 1, groups=height)
        self.cl3 = nn.Conv2d(ch_latent_c[1] * height, ch_latent_c[2] * height, 1, groups=height)
        self.cl4 = nn.Conv2d(ch_latent_c[2] * height, (classes + 1) * height, 1, groups=height)

        # only one output (regression!!
        self.reg1 = nn.Conv2d(ch_in * height, ch_latent_r[0] * height, 1, groups=height)
        #self.reg1_bn = nn.BatchNorm2d(ch_latent_r[0])
        self.reg2_cm = CondMul(superclasses * height, input_features=ch_latent_r[0], output_features=ch_latent_r[1])
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
        x = F.leaky_relu(self.cl3(x))
        x = self.cl4(x)
        # go back to (b, c, h, w)
        x = x.reshape((batch_size, height, -1, width)).permute((0, 2, 1, 3))
        #print(x.shape)
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
            loss = F.cross_entropy(classes, inds_gt.squeeze(1).clamp(0, self.classes-1))
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
                 ch_latent=[[32, 32], [32, 32], [32, 32]]):
        super(Classifier3Stage, self).__init__()
        self.classes = classes
        self.pad = pad
        classes12 = classes[0] * classes[1]
        self.c1 = nn.ModuleList([nn.Conv2d(ch_in * height, ch_latent[0][0] * height, 1, groups=height),
                                 CondMul(classes[0] * height, ch_in, ch_latent[1][0]),
                                 CondMul(classes12 * height, ch_in, ch_latent[2][0])])
        self.c2 = nn.ModuleList([nn.Conv2d(ch_latent[0][0], ch_latent[0][1], 1, groups=height),
                                 CondMul(classes[0] * height, ch_latent[1][0], ch_latent[1][1]),
                                 CondMul(classes12 * height, ch_latent[2][0], ch_latent[2][1])])
        self.c3 = nn.ModuleList([nn.Conv2d(ch_latent[0][1], classes[0], 1),
                                 CondMul(classes[0] * height, ch_latent[1][1], classes[1] + 2 * pad[1]),
                                 CondMul(classes12 * height, ch_latent[2][1], classes[2] + 2 * pad[2])])

    def forward(self, x_in, inds_gt=None, mask_gt=None):
        batches = x_in.shape[0]
        height = x_in.shape[2]
        width = x_in.shape[3]

        classes12 = self.classes[0] * self.classes[1]
        classes123= classes12 * self.classes[2]
        classes23 = self.classes[1] * self.classes[2]
        x = F.leaky_relu(self.c1[0](x_in))
        x = F.leaky_relu(self.c2[0](x))
        x = self.c3[0](x)
        x1 = x
        inds1 = x.argmax(dim=1)
        inds1_l = inds1.flatten().type(torch.int32)
        # convert from (b, c, h, w) to (b, h, w, c) to (b * h * w, c)
        x_l = x_in.permute((0, 2, 3, 1)).reshape((-1, x_in.shape[1])).contiguous()
        x = F.leaky_relu(self.c1[1](x_l, inds1_l))
        x = F.leaky_relu(self.c2[1](x, inds1_l))
        x = self.c3[1](x, inds1_l)
        inds2 = x.argmax(dim=1)
        inds12_l = inds1_l * self.classes[1] + (inds2.flatten() - self.pad[1])
        inds12_l = inds12_l.clamp(0, classes12 - 1).type(torch.int32)

        x = F.leaky_relu(self.c1[2](x_l, inds12_l))
        x = F.leaky_relu(self.c2[2](x, inds12_l))
        x = self.c3[2](x, inds12_l)
        inds3 = x.argmax(dim=1)
        inds123_real = inds12_l * self.classes[2] + (inds3.flatten() - self.pad[2])
        inds123_real = inds123_real.reshape((batches, 1, height, width)).clamp(0, classes123 - 1)
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
                #calculate the index of this class/ its neighbours
                inds1_l = inds1_gt + i
                inds1_l = inds1_l.clamp(0, self.classes[0] - 1)
                #calculate the local groundtruth index
                inds2_gt = inds_gt // self.classes[2] - inds1_l * self.classes[1]
                inds2_gt = inds2_gt + self.pad[1] #todo: really plus?

                #the mask masks out where this would not yield any valid samples
                mask = torch.logical_and(inds2_gt >= 0, inds2_gt < (self.classes[1] + 2 * self.pad[1]))
                inds2_gt = inds2_gt.clamp(0, self.classes[1] + 2 * self.pad[1] - 1).squeeze(1).type(torch.int64)

                inds1_l = inds1_l.flatten().type(torch.int32)
                x = F.leaky_relu(self.c1[1](x_l, inds1_l))
                x = F.leaky_relu(self.c2[1](x, inds1_l))
                x = self.c3[1](x, inds1_l)
                #from (b * h * w, c) to (b, h, w, c) to (b, c, h, w)
                x = x.reshape((batches, height, width, -1)).permute((0, 3, 1, 2))
                loss = F.cross_entropy(x, inds2_gt) * mask
                losses.append(loss.mean())

            inds12_gt = inds_gt // self.classes[2]
            inds12_gt = inds12_gt
            for i in [-1, 0, 1]:
                #calculate the index of this class/ its neighbours
                inds12_l = inds12_gt + i
                inds12_l = inds12_l.clamp(0, classes12 - 1)
                #calculate the local groundtruth index
                inds3_gt = inds_gt  - inds12_l * self.classes[2]
                inds3_gt = inds3_gt + self.pad[2] #todo: really plus?

                #the mask masks out where this does not yield any valid samples
                mask = torch.logical_and(inds3_gt >= 0, inds3_gt < (self.classes[2] + 2 * self.pad[2]))
                inds3_gt = inds3_gt.clamp(0, self.classes[2] + 2 * self.pad[2] - 1).squeeze(1).type(torch.int64)

                inds12_l = inds12_l.flatten().type(torch.int32)
                x = F.leaky_relu(self.c1[2](x_l, inds12_l))
                x = F.leaky_relu(self.c2[2](x, inds12_l))
                x = self.c3[2](x, inds12_l)
                # from (b * h * w, c) to (b, h, w, c) to (b, c, h, w)
                x = x.reshape((batches, height, width, -1)).permute((0, 3, 1, 2))
                loss = F.cross_entropy(x, inds3_gt) * mask
                losses.append(loss.mean())

            return inds123_real, losses


#same as #5 but without batch normalization
class Reg_3stage(nn.Module):

    #default parameters are the same as for CR8_reg_cond_mul_2
    def __init__(self, ch_in=128,
                 ch_latent=[128, 128, 128],
                 superclasses=8,
                 ch_latent_r=[128, 32],
                 ch_latent_msk=[32, 16],
                 classes=[16, 16, 16],
                 pad=[0, 8, 8],
                 ch_latent_c=[[32, 32], [32, 32], [32, 32]]):
        super(Reg_3stage, self).__init__()
        classes123 = classes[0] * classes[1] * classes[2]
        self.classes = classes
        self.superclasses = superclasses
        self.class_factor = int(classes123/superclasses)
        # the first latent layer for classification is shared
        self.bb1 = nn.Conv2d(ch_in, ch_latent[0], 1)
        self.bb2 = nn.Conv2d(ch_latent[0], ch_latent[1], 1)
        self.bb3 = nn.Conv2d(ch_latent[1], ch_latent[2], 1)

        self.c = Classification3Stage(ch_in=ch_latent[2],
                                      classes=classes,
                                      pad=pad,  # pad around classes
                                      ch_latent=ch_latent_c)

        # only one output (regression!!
        # self.cond_mul = RefCondMulConv(classes, input_features=ch_latent, output_features=1)
        # self.cond_mul = RefCondMul(classes, input_features=ch_latent, output_features=1)

        self.r1 = nn.Conv2d(ch_in, ch_latent_r[0], 1)
        self.r2 = CondMul(superclasses, ch_latent_r[0], ch_latent_r[1])
        self.r3 = CondMul(classes123, ch_latent_r[1], 1)

        # kernels for masks:
        self.msk1 = nn.Conv2d(ch_in, ch_latent_msk[0], 1)
        self.msk2 = nn.Conv2d(ch_latent_msk[0], ch_latent_msk[1], 1)
        self.msk3 = nn.Conv2d(ch_latent_msk[1], 1, 1)



    def forward(self, x_in, x_gt=None, mask_gt=None):
        height = x_in.shape[2]
        width = x_in.shape[3]
        classes123 = self.classes[0] * self.classes[1] * self.classes[2]
        batch_size = x_in.shape[0]
        int_type = torch.int32

        #the first stage is to adapt to features to something that has meaning on this line!
        x = F.leaky_relu(self.bb1(x_in))
        x = F.leaky_relu(self.bb2(x))
        x_l = F.leaky_relu(self.bb3(x))

        #calculate the mask/confidence on these lines
        x = F.leaky_relu(self.msk1(x_in))
        x = F.leaky_relu(self.msk2(x))
        mask = F.leaky_relu(self.msk3(x))


        if x_gt is None:
            inds = self.c(x_l).flatten().type(torch.int32)
            inds_super = inds // self.class_factor

            x = F.leaky_relu(self.r1(x_in))
            # todo: change this for multiline!
            # from (b, c, h, w) to (b, h, w, c) to (b * h * w, c)
            x_l = x.permute((0, 2, 3, 1)).reshape((-1, x.shape[1])).contiguous()
            x = F.leaky_relu(self.r2(x_l, inds_super))
            r = self.r3(x, inds).flatten()

            x = (inds.type(torch.float32) + r) * (1.0 / float(classes123))
            x = x.reshape((batch_size, 1, height, width))
            return x, mask
        else:

            inds_gt = (x_gt * classes123).type(torch.int32).clamp(0, classes123 - 1)
            inds, class_losses = self.c(x_l, inds_gt, mask_gt)

            # todo: change this for multiline!
            x = F.leaky_relu(self.r1(x_in))
            # from (b, c, h, w) to (b, h, w, c) to (b * h * w, c)
            x_l = x.permute((0, 2, 3, 1)).reshape((-1, x.shape[1]))

            #calculate the regression only x
            inds_gt = inds_gt.clamp(0, classes123 - 1).flatten()
            inds_super = inds_gt // self.class_factor
            x = F.leaky_relu(self.r2(x_l, inds_super))
            r = self.r3(x, inds_gt).flatten()

            x_reg = (inds_gt.type(torch.float32) + r) * (1.0 / float(classes123))
            x_reg = x_reg.reshape((batch_size, 1, height, width))


            #calculate the real x
            inds = inds.flatten().type(torch.int32)
            inds_super = inds // self.class_factor
            x = F.leaky_relu(self.r2(x_l, inds_super))
            r = self.r3(x, inds_gt).flatten()

            x = (inds.type(torch.float32) + r) * (1.0 / float(classes123))
            x_real = x.reshape((batch_size, 1, height, width))

            return x_reg, mask, class_losses, x_real
