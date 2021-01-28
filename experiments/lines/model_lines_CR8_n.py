import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_shrink, ResidualBlock_vshrink
from model.cuda_cond_mul.cond_mul import CondMul
from model.cuda_cond_mul.reference_cond_mul import RefCondMul, RefCondMulConv


#don't trust any of these calculations in the layers we had before...
# (conv_ch_up_1 would have a stride of 2)
class Model_Lines_CR8_n(nn.Module):

    def __init__(self, classes):
        self.classes = classes
        super(Model_Lines_CR8_n, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_start = nn.Conv2d(1, 32, 3, padding=0) #1
        self.resi_block1 = ResidualBlock_shrink(32, 3, 0, depadding=3) #3
        #self.conv1 = nn.Conv2d(66, 63, 3, padding=1, padding_mode='same')
        self.resi_block2 = ResidualBlock_shrink(32, 3, 0, depadding=3) #3
        self.conv_ch_up_1 = nn.Conv2d(32, 64, 5, padding=0) #2 here stride 2 subsampling
        self.resi_block3 = ResidualBlock_shrink(64, 3, 0, depadding=3) #3
        self.conv_end_1 = nn.Conv2d(64, 128, 7, padding=0, padding_mode='replicate') #3 or here
        # if this does not work... add one more 1x1 convolutional layer here
        self.conv_end_2 = nn.Conv2d(128, 512, 1, padding=0, padding_mode='replicate')
        self.conv_end_3 = nn.Conv2d(512, classes*2+1, 1, padding=0, padding_mode='replicate')

    def forward(self, x):
        ### LAYER 0
        x = F.leaky_relu(self.conv_start(x))

        x = self.resi_block1(x)
        x = self.resi_block2(x)
        x = F.leaky_relu(self.conv_ch_up_1(x))
        x = self.resi_block3(x)
        x_latent = x
        x = F.leaky_relu(self.conv_end_1(x))
        x = F.leaky_relu(self.conv_end_2(x))
        x = F.leaky_relu(self.conv_end_3(x))
        classes = F.softmax(x[:, 0:self.classes, :, :], dim=1)
        regressions = x[:, self.classes:(2 * self.classes), :, :]
        mask = x[:, [-1], :, :]
        return classes, regressions, mask, x_latent

# CR8 backbone!!!
class CR8_bb(nn.Module):

    def __init__(self):
        super(CR8_bb, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_start = nn.Conv2d(1, 32, 3, padding=(0, 1)) #1
        self.resi_block1 = ResidualBlock_vshrink(32, 3, padding=(0, 1), depadding=3) # + 3 = 4
        self.resi_block2 = ResidualBlock_vshrink(32, 3, padding=(0, 1), depadding=3) # + 3 = 7
        self.conv_ch_up_1 = nn.Conv2d(32, 64, 5, padding=(0, 2), stride=(2, 2)) # + 2 = 9
        # subsampled beginning here!
        self.resi_block3 = ResidualBlock_vshrink(64, 3, padding=(0, 1), depadding=3) # + 2 * 3 = 15
        self.conv_end_1 = nn.Conv2d(64, 128, 3, padding=(0, 1)) # + 2 * 1 = 17
        # if this does not work... add one more 1x1 convolutional layer here


    def forward(self, x):
        ### LAYER 0
        #print(x.shape)
        x = F.leaky_relu(self.conv_start(x))

        #print(x.shape)
        x = self.resi_block1(x)

        #print(x.shape)
        x = self.resi_block2(x)
        #print(x.shape)
        x = F.leaky_relu(self.conv_ch_up_1(x))
        #print(x.shape)
        x = self.resi_block3(x)
        #print(x.shape)
        x = F.leaky_relu(self.conv_end_1(x))
        #print(x.shape)
        return x

# CR8 regressor!!!
class CR8_reg(nn.Module):

    def __init__(self, classes, ch_in=128, ch_latent=512):
        self.classes = classes
        super(CR8_reg, self).__init__()
        self.conv_end_2 = nn.Conv2d(ch_in, ch_latent, 1, padding=0)
        self.bn_2 = nn.BatchNorm2d(ch_latent)
        self.conv_end_3 = nn.Conv2d(ch_latent, classes * 2 + 1, 1, padding=0, padding_mode='replicate')

    def forward(self, x, x_gt=None, mask_gt=None):
        ### LAYER 0

        #print(x.shape)
        x = F.leaky_relu(self.conv_end_2(x))
        x = self.bn_2(x)
        x = self.conv_end_3(x)

        classes = x[:, 0:self.classes, :, :]#the softmax is built into F.cross_entropy
        regressions_all = x[:, self.classes:(2 * self.classes), :, :]
        mask = F.leaky_relu(x[:, [-1], :, :])

        inds = classes.argmax(dim=1).unsqueeze(1)

        if x_gt is None:
            regressions = torch.gather(regressions_all, dim=1, index=inds)
            x = (inds.type(torch.float32) + regressions) * (1.0 / float(self.classes))
            return x, mask
        else:
            inds_gt = (x_gt * self.classes).type(torch.int64)
            loss = F.cross_entropy(classes, inds_gt.squeeze(1))
            class_losses = [torch.mean(loss * mask_gt)]

            regressions = torch.gather(regressions_all, dim=1, index=inds_gt)
            x = (inds_gt.type(torch.float32) + regressions) * (1.0 / float(self.classes))

            if torch.any(torch.isnan(regressions)):
                print("regressions: found nan")
            if torch.any(torch.isinf(regressions)):
                print("regressions: found inf")

            regressions = torch.gather(regressions_all, dim=1, index=inds)
            regressions = regressions.clamp(-1, 2) # hope to kill the infs with this
            x_real = (inds.type(torch.float32) + regressions) * (1.0 / float(self.classes))

            if torch.any(torch.isnan(regressions)):
                print("regressions_real: found nan")
            if torch.any(torch.isinf(regressions)):
                print("regressions_real: found inf")
            return x, mask, class_losses, x_real


# CR8 regressor!!!
class CR8_reg_cond_mul(nn.Module):

    def __init__(self, classes, ch_in=128, ch_latent=128):
        super(CR8_reg_cond_mul, self).__init__()
        self.classes = classes
        self.conv_1 = nn.Conv2d(ch_in, ch_latent, 1, padding=0)
        self.bn_1 = nn.BatchNorm2d(ch_latent)
        #only one output (regression!!
        #self.cond_mul = RefCondMulConv(classes, input_features=ch_latent, output_features=1)
        #self.cond_mul = RefCondMul(classes, input_features=ch_latent, output_features=1)
        self.cond_mul = CondMul(classes, input_features=ch_latent, output_features=1)
        self.conv_2 = nn.Conv2d(ch_latent, classes + 1, 1, padding=0, padding_mode='replicate')

    def forward(self, x, x_gt=None, mask_gt=None):
        batch_size = x.shape[0]
        int_type = torch.int32

        #print(x.shape)
        x_latent = F.leaky_relu(self.bn_1(self.conv_1(x)))
        x = self.conv_2(x_latent)

        #classes = F.softmax(x[:, 0:self.classes, :, :], dim=1)
        classes = x[:, 0:self.classes, :, :] #cross entropy already has a softmax
        mask = F.leaky_relu(x[:, [-1], :, :])

        #reshaped latent features:
        # from (b, c, h, w) to (b, w, h, c) to (b * w * h, c)
        x_l = x_latent.transpose(1, 3).reshape((-1, x_latent.shape[1]))
        inds = classes.argmax(dim=1).unsqueeze(1)

        if x_gt is None:

            regression = self.cond_mul(x_l, inds.flatten().type(int_type))
            #assumint h=1 we get the shape back to (b, c, h, w)
            regression = regression.reshape((batch_size, 1, 1, -1))
            x = (inds.type(torch.float32) + regression) * (1.0 / float(self.classes))
            return x, mask
        else:
            inds_gt = (x_gt * self.classes).type(torch.int64)
            loss = F.cross_entropy(classes, inds_gt.squeeze(1))
            class_losses = [torch.mean(loss * mask_gt)]

            regression = self.cond_mul(x_l, inds_gt.clamp(0, self.classes-1).flatten().type(int_type))
            # assumint h=1 we get the shape back to (b, c, h, w)
            regression = regression.reshape((batch_size, 1, 1, -1))

            x = (inds_gt.type(torch.float32) + regression) * (1.0 / float(self.classes))

            if torch.any(torch.isnan(regression)):
                print("regressions: found nan")
            if torch.any(torch.isinf(regression)):
                print("regressions: found inf")

            regression = self.cond_mul(x_l, inds.flatten().type(int_type))
            #assumint h=1 we get the shape back to (b, c, h, w)
            regression = regression.reshape((batch_size, 1, 1, -1))
            x_real = (inds.type(torch.float32) + regression) * (1.0 / float(self.classes))

            if torch.any(torch.isnan(regression)):
                print("regressions_real: found nan")
            if torch.any(torch.isinf(regression)):
                print("regressions_real: found inf")
            return x, mask, class_losses, x_real

# CR8 regressor!!!
class CR8_reg_cond_mul_2(nn.Module):

    def __init__(self, classes, ch_in=128, ch_latent=128):
        super(CR8_reg_cond_mul_2, self).__init__()
        self.classes = classes
        self.cl1 = nn.Conv2d(ch_in, ch_latent, 1)
        self.cl1_bn = nn.BatchNorm2d(ch_latent)
        self.cl2 = nn.Conv2d(ch_in, ch_latent, 1)
        self.cl3 = nn.Conv2d(ch_latent, classes + 1, 1)

        #only one output (regression!!
        #self.cond_mul = RefCondMulConv(classes, input_features=ch_latent, output_features=1)
        #self.cond_mul = RefCondMul(classes, input_features=ch_latent, output_features=1)

        self.reg1 = nn.Conv2d(ch_in, ch_latent, 1)
        self.reg1_bn = nn.BatchNorm2d(ch_latent)
        self.reg2_cm = CondMul(classes, input_features=ch_latent, output_features=32)
        self.reg3_cm = CondMul(classes, input_features=32, output_features=1)

    def forward(self, x_in, x_gt=None, mask_gt=None):
        batch_size = x_in.shape[0]
        int_type = torch.int32

        #print(x.shape)
        x = F.leaky_relu(self.cl1_bn(self.cl1(x_in)))
        x = F.leaky_relu(self.cl2(x))
        x = self.cl3(x)

        #classes = F.softmax(x[:, 0:self.classes, :, :], dim=1)
        classes = x[:, 0:self.classes, :, :] #cross entropy already has a softmax
        mask = F.leaky_relu(x[:, [-1], :, :])

        #reshaped latent features:
        inds = classes.argmax(dim=1).unsqueeze(1)

        #now do the regressions:
        inds_r = inds.flatten().type(int_type)
        x = F.leaky_relu(self.reg1_bn(self.reg1(x_in)))
        # from (b, c, h, w) to (b, w, h, c) to (b * w * h, c)
        x = x.transpose(1, 3).reshape((-1, x.shape[1])).contiguous()
        x = F.leaky_relu(self.reg2_cm(x, inds_r))
        regression = self.reg3_cm(x, inds_r)

        # assuming h=1 we get the shape back to (b, c, h, w)
        regression = regression.reshape((batch_size, 1, 1, -1))

        x_real = (inds.type(torch.float32) + regression) * (1.0 / float(self.classes))
        if x_gt is None:
            return x_real, mask
        else:
            inds_gt = (x_gt * self.classes).type(torch.int64)
            loss = F.cross_entropy(classes, inds_gt.squeeze(1))
            class_losses = [torch.mean(loss * mask_gt)]

            inds_r = inds_gt.clamp(0, self.classes - 1).flatten().type(int_type)
            x = F.leaky_relu(self.reg1_bn(self.reg1(x_in)))
            # from (b, c, h, w) to (b, w, h, c) to (b * w * h, c)
            x = x.transpose(1, 3).reshape((-1, x.shape[1]))
            x = F.leaky_relu(self.reg2_cm(x, inds_r))
            regression = self.reg3_cm(x, inds_r)
            regression = regression.reshape((batch_size, 1, 1, -1))

            x = (inds_gt.type(torch.float32) + regression) * (1.0 / float(self.classes))

            if torch.any(torch.isnan(regression)):
                print("regressions: found nan")
            if torch.any(torch.isinf(regression)):
                print("regressions: found inf")

            return x, mask, class_losses, x_real

        # CR8 regressor!!!
class CR8_reg_cond_mul_3(nn.Module):

    #default parameters are the same as for CR8_reg_cond_mul_2
    def __init__(self, classes, ch_in=128, ch_latent_c=[128, 128], ch_latent_r=[128, 32]):
        super(CR8_reg_cond_mul_3, self).__init__()
        self.classes = classes
        self.cl1 = nn.Conv2d(ch_in, ch_latent_c[0], 1)
        self.cl1_bn = nn.BatchNorm2d(ch_latent_c[0])
        self.cl2 = nn.Conv2d(ch_latent_c[0], ch_latent_c[1], 1)
        self.cl3 = nn.Conv2d(ch_latent_c[1], classes + 1, 1)

        # only one output (regression!!
        # self.cond_mul = RefCondMulConv(classes, input_features=ch_latent, output_features=1)
        # self.cond_mul = RefCondMul(classes, input_features=ch_latent, output_features=1)

        self.reg1 = nn.Conv2d(ch_in, ch_latent_r[0], 1)
        self.reg1_bn = nn.BatchNorm2d(ch_latent_r[0])
        self.reg2_cm = CondMul(classes, input_features=ch_latent_r[0], output_features=ch_latent_r[1])
        self.reg3_cm = CondMul(classes, input_features=ch_latent_r[1], output_features=1)

    def forward(self, x_in, x_gt=None, mask_gt=None):
        batch_size = x_in.shape[0]
        int_type = torch.int32

        # print(x.shape)
        x = F.leaky_relu(self.cl1_bn(self.cl1(x_in)))
        x = F.leaky_relu(self.cl2(x))
        x = self.cl3(x)

        # classes = F.softmax(x[:, 0:self.classes, :, :], dim=1)
        classes = x[:, 0:self.classes, :, :]  # cross entropy already has a softmax
        mask = F.leaky_relu(x[:, [-1], :, :])

        # reshaped latent features:
        inds = classes.argmax(dim=1).unsqueeze(1)

        # now do the regressions:
        inds_r = inds.flatten().type(int_type)
        x = F.leaky_relu(self.reg1_bn(self.reg1(x_in)))
        # from (b, c, h, w) to (b, w, h, c) to (b * w * h, c)
        x = x.transpose(1, 3).reshape((-1, x.shape[1]))
        x = F.leaky_relu(self.reg2_cm(x, inds_r))
        regression = self.reg3_cm(x, inds_r)

        # assuming h=1 we get the shape back to (b, c, h, w)
        regression = regression.reshape((batch_size, 1, 1, -1))

        x_real = (inds.type(torch.float32) + regression) * (1.0 / float(self.classes))
        if x_gt is None:
            return x_real, mask
        else:
            inds_gt = (x_gt * self.classes).type(torch.int64)
            loss = F.cross_entropy(classes, inds_gt.squeeze(1))
            class_losses = [torch.mean(loss * mask_gt)]

            inds_r = inds_gt.clamp(0, self.classes - 1).flatten().type(int_type)
            x = F.leaky_relu(self.reg1_bn(self.reg1(x_in)))
            # from (b, c, h, w) to (b, w, h, c) to (b * w * h, c)
            x = x.transpose(1, 3).reshape((-1, x.shape[1]))
            x = F.leaky_relu(self.reg2_cm(x, inds_r))
            regression = self.reg3_cm(x, inds_r)
            regression = regression.reshape((batch_size, 1, 1, -1))

            x = (inds_gt.type(torch.float32) + regression) * (1.0 / float(self.classes))

            if torch.any(torch.isnan(regression)):
                print("regressions: found nan")
            if torch.any(torch.isinf(regression)):
                print("regressions: found inf")

            return x, mask, class_losses, x_real

# CR8 regressor!!!
class CR8_reg_2_stage(nn.Module):

    def __init__(self, classes=[16, 16], ch_in=128, ch_latent=128):

        super(CR8_reg_2_stage, self).__init__()
        self.classes = classes
        #everything we need for the first classification stage (TODO:per line):
        self.c1_1 = nn.Conv2d(ch_in, ch_latent, 1, padding=0)
        self.c1_2 = nn.Conv2d(ch_latent, ch_latent, 1, padding=0)
        self.c1_3 = nn.Conv2d(ch_latent, classes[0] + 1, 1, padding=0, padding_mode='replicate')

        #used or the second classification stage:
        self.c2_1 = CondMul(classes[0], input_features=ch_in, output_features=32)
        self.c2_2 = CondMul(classes[0], input_features=32, output_features=32) #because 3
        self.c2_3 = CondMul(classes[0], input_features=32, output_features=classes[1])

        #now
        self.r1_1 =CondMul(classes[0] * classes[1], input_features=ch_in, output_features=32)
        self.r1_2 =CondMul(classes[0] * classes[1], input_features=32, output_features=16)
        self.r1_3 = CondMul(classes[0] * classes[1], input_features=16, output_features=1)

    def forward(self, x_in, x_gt=None, mask_gt=None):
        batch_size = x_in.shape[0]
        classes = self.classes[0] * self.classes[1]

        # print(x.shape)
        x1 = F.leaky_relu(self.c1_1(x_in))
        x1 = F.leaky_relu(self.c1_2(x1))
        x1 = self.c1_3(x1)
        mask = F.leaky_relu(x1[:, [-1], :, :])

        inds1 = x1[:, 0:self.classes[0], :, :].argmax(dim=1).flatten().type(torch.int32)

        # from (b, c, h, w) to (b, w, h, c) to (b * w * h, c)
        x3 = x2 = x_in.transpose(1, 3).reshape((-1, x_in.shape[1]))
        x2 = F.leaky_relu(self.c2_1(x2, inds1))
        x2 = F.leaky_relu(self.c2_2(x2, inds1))
        x2 = self.c2_3(x2, inds1)

        inds2 = x2.argmax(dim=1).flatten().type(torch.int32)

        inds12 = inds1 * self.classes[0] + inds2
        x3 = F.leaky_relu(self.r1_1(x3, inds12))
        x3 = F.leaky_relu(self.r1_2(x3, inds12))
        r = self.r1_3(x3, inds12).flatten()

        x_real = (inds12.type(torch.float32) + r) * (1.0 / float(classes))
        #to (b, c, h, w)
        x_real = x_real.reshape([x_in.shape[0], 1, x_in.shape[2], x_in.shape[3]])

        if x_gt is None:
            return x_real, mask
        else:
            inds1_gt = (x_gt * self.classes[0]).type(torch.int64)
            inds12_gt = (x_gt * self.classes[0] * self.classes[1]).type(torch.int64)
            inds2_gt = inds12_gt % self.classes[1]

            loss1 = F.cross_entropy(x1[:, 0:self.classes[0], :, :], inds1_gt.squeeze(1))
            inds1_gt = inds1_gt.clamp(0, self.classes[0] - 1).type(torch.int64)
            inds2_gt = inds2_gt.clamp(0, self.classes[1] - 1).type(torch.int64)
            inds12_gt = inds12_gt.clamp(0, classes - 1).type(torch.int32)

            x3 = x2 = x_in.transpose(1, 3).reshape((-1, x_in.shape[1]))
            inds1_gt = inds1_gt.flatten().type(torch.int32)
            x2 = F.leaky_relu(self.c2_1(x2, inds1_gt))
            x2 = F.leaky_relu(self.c2_2(x2, inds1_gt))
            x2 = self.c2_3(x2, inds1_gt)

            #from (b*w*h, c) to (b, h, w, c)
            x2 = x2.reshape([batch_size, x_in.shape[2], x_in.shape[3], x2.shape[1]])
            #to (b, c, h, w)
            x2 = x2.permute(0, 3, 1, 2)
            loss2 = F.cross_entropy(x2, inds2_gt.squeeze(1))

            inds12_gt = inds12_gt.flatten()
            x3 = F.leaky_relu(self.r1_1(x3, inds12_gt))
            x3 = F.leaky_relu(self.r1_2(x3, inds12_gt))
            r = self.r1_3(x3, inds12).flatten()

            x = (inds12_gt.type(torch.float32) + r) * (1.0 / float(classes))
            # to (b, c, h, w)
            x = x.reshape([x_in.shape[0], 1, x_in.shape[2], x_in.shape[3]])

            class_losses = [torch.mean(loss1), torch.mean(loss2)] #* mask_gt
            return x, mask, class_losses, x_real


            regression = self.cond_mul(x_l, inds_gt.clamp(0, self.classes - 1).flatten().type(int_type))
            # assumint h=1 we get the shape back to (b, c, h, w)
            regression = regression.reshape((batch_size, 1, 1, -1))

            x = (inds_gt.type(torch.float32) + regression) * (1.0 / float(self.classes))

            if torch.any(torch.isnan(regression)):
                print("regressions: found nan")
            if torch.any(torch.isinf(regression)):
                print("regressions: found inf")

            regression = self.cond_mul(x_l, inds.flatten().type(int_type))
            # assumint h=1 we get the shape back to (b, c, h, w)
            regression = regression.reshape((batch_size, 1, 1, -1))
            x_real = (inds.type(torch.float32) + regression) * (1.0 / float(self.classes))

            if torch.any(torch.isnan(regression)):
                print("regressions_real: found nan")
            if torch.any(torch.isinf(regression)):
                print("regressions_real: found inf")
            return x, mask, class_losses, x_real

# CR8 backbone!!!
class CR8_bb_no_residual(nn.Module):

    def __init__(self):
        super(CR8_bb_no_residual, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_start = nn.Conv2d(1, 32, 3, padding=(0, 1)) #1
        self.conv_1 = nn.Conv2d(32, 32, 3, padding=(0, 1)) # + 1 = 2
        self.conv_2 = nn.Conv2d(32, 32, 3, padding=(0, 1)) # + 1 = 3
        self.conv_3 = nn.Conv2d(32, 32, 3, padding=(0, 1)) # + 1 = 4
        self.conv_4 = nn.Conv2d(32, 32, 3, padding=(0, 1)) # + 1 = 5
        self.conv_5 = nn.Conv2d(32, 32, 3, padding=(0, 1)) # + 1 = 6
        self.conv_6 = nn.Conv2d(32, 32, 3, padding=(0, 1)) # + 1 = 7
        self.conv_ch_up_1 = nn.Conv2d(32, 64, 5, padding=(0, 2), stride=(2, 2)) # + 2 = 9
        #subsampled from here!
        self.conv_7 = nn.Conv2d(64, 64, 3, padding=(0, 1)) # + 1 * 2 = 11
        self.conv_8 = nn.Conv2d(64, 64, 3, padding=(0, 1)) # + 1 * 2 = 13
        self.conv_9 = nn.Conv2d(64, 64, 3, padding=(0, 1)) # + 1 * 2 = 15
        self.conv_out = nn.Conv2d(64, 128, 3, padding=(0, 1)) # + 1 * 2 = 17


    def forward(self, x):
        x = F.leaky_relu(self.conv_start(x))
        x = F.leaky_relu(self.conv_1(x))
        x = F.leaky_relu(self.conv_2(x))
        x = F.leaky_relu(self.conv_3(x))
        x = F.leaky_relu(self.conv_4(x))
        x = F.leaky_relu(self.conv_5(x))
        x = F.leaky_relu(self.conv_6(x))
        x = F.leaky_relu(self.conv_ch_up_1(x))
        x = F.leaky_relu(self.conv_7(x))
        x = F.leaky_relu(self.conv_8(x))
        x = F.leaky_relu(self.conv_9(x))
        x = F.leaky_relu(self.conv_out(x))

        return x
# CR8 backbone!!!
class CR8_bb_no_residual_light(nn.Module):

    def __init__(self):
        super(CR8_bb_no_residual_light, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_start = nn.Conv2d(1, 16, 3, padding=(0, 1)) #1
        self.conv_1 = nn.Conv2d(16, 32, 3, padding=(0, 1)) # + 1 = 2
        self.conv_2 = nn.Conv2d(32, 32, 3, padding=(0, 1)) # + 1 = 3
        self.conv_3_down = nn.Conv2d(32, 64, 5, padding=(0, 2), stride=(2, 2)) # + 2 = 5
        # subsampled from here!
        self.conv_4 = nn.Conv2d(64, 64, 3, padding=(0, 1)) # + 1 * 2 = 7
        self.conv_5 = nn.Conv2d(64, 64, 3, padding=(0, 1)) # + 1 * 2 = 9
        self.conv_6 = nn.Conv2d(64, 64, 3, padding=(0, 1)) # + 1 * 2 = 11
        self.conv_7 = nn.Conv2d(64, 64, 3, padding=(0, 1)) # + 1 * 2 = 13
        self.conv_8 = nn.Conv2d(64, 64, 3, padding=(0, 1)) # + 1 * 2 = 15
        self.conv_9 = nn.Conv2d(64, 128, 3, padding=(0, 1)) # + 1 * 2 = 17


    def forward(self, x):
        x = F.leaky_relu(self.conv_start(x))
        x = F.leaky_relu(self.conv_1(x))
        x = F.leaky_relu(self.conv_2(x))
        x = F.leaky_relu(self.conv_3_down(x))
        x = F.leaky_relu(self.conv_4(x))
        x = F.leaky_relu(self.conv_5(x))
        x = F.leaky_relu(self.conv_6(x))
        x = F.leaky_relu(self.conv_7(x))
        x = F.leaky_relu(self.conv_8(x))
        x = F.leaky_relu(self.conv_9(x))
        return x



# CR8 backbone!!!
class CR8_mask_var(nn.Module):

    def __init__(self):
        super(CR8_mask_var, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_start = nn.Conv2d(1, 32, 3, padding=(0, 1)) #1
        self.resi_block1 = ResidualBlock_vshrink(32, 3, padding=(0, 1), depadding=3) # + 3 = 4
        self.resi_block2 = ResidualBlock_vshrink(32, 3, padding=(0, 1), depadding=3) # + 3 = 7
        self.conv_ch_up_1 = nn.Conv2d(32, 64, 5, padding=(0, 2), stride=(2, 2)) # + 2 = 9
        # subsampled beginning here!
        self.resi_block3 = ResidualBlock_vshrink(64, 3, padding=(0, 1), depadding=3) # + 2 * 3 = 15
        self.conv_end_1 = nn.Conv2d(64, 2, 3, padding=(0, 1)) # + 2 * 1 = 17
        # if this does not work... add one more 1x1 convolutional layer here


    def forward(self, x):
        ### LAYER 0
        #print(x.shape)
        x = F.leaky_relu(self.conv_start(x))

        #print(x.shape)
        x = self.resi_block1(x)

        #print(x.shape)
        x = self.resi_block2(x)
        #print(x.shape)
        x = F.leaky_relu(self.conv_ch_up_1(x))
        #print(x.shape)
        x = self.resi_block3(x)
        #print(x.shape)
        x = F.leaky_relu(self.conv_end_1(x))
        #print(x.shape)
        mask = x[:, 0, :, :]
        sigma = x[:, 1, :, :]
        return mask, sigma