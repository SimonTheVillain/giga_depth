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

class CR8_reg_cond_mul_4(nn.Module):

    #default parameters are the same as for CR8_reg_cond_mul_2
    def __init__(self, classes, ch_in=128, ch_latent_c=[128, 128], ch_latent_r=[128, 32]):
        super(CR8_reg_cond_mul_4, self).__init__()
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
        self.reg2_cm = CondMul(classes, input_features=ch_latent_r[0] + ch_latent_c[0], output_features=ch_latent_r[1])
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
        x = torch.cat((x, F.leaky_relu(self.cl1_bn(self.cl1(x_in)))), 1)
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
            x = torch.cat((x, F.leaky_relu(self.cl1_bn(self.cl1(x_in)))), 1)
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

class CR8_reg_cond_mul_5(nn.Module):

    #default parameters are the same as for CR8_reg_cond_mul_2
    def __init__(self, classes=128, superclasses=8, ch_in=128, ch_latent_c=[128, 128], ch_latent_r=[128, 32]):
        super(CR8_reg_cond_mul_5, self).__init__()
        self.classes = classes
        self.superclasses = superclasses
        self.class_factor = int(classes/superclasses)
        self.cl1 = nn.Conv2d(ch_in, ch_latent_c[0], 1)
        self.cl1_bn = nn.BatchNorm2d(ch_latent_c[0])
        self.cl2 = nn.Conv2d(ch_latent_c[0], ch_latent_c[1], 1)
        self.cl3 = nn.Conv2d(ch_latent_c[1], classes + 1, 1)

        # only one output (regression!!
        # self.cond_mul = RefCondMulConv(classes, input_features=ch_latent, output_features=1)
        # self.cond_mul = RefCondMul(classes, input_features=ch_latent, output_features=1)

        self.reg1 = nn.Conv2d(ch_in, ch_latent_r[0], 1)
        self.reg1_bn = nn.BatchNorm2d(ch_latent_r[0])
        self.reg2_cm = CondMul(superclasses, input_features=ch_latent_r[0] + ch_latent_c[0], output_features=ch_latent_r[1])
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
        inds_super = inds_r / self.class_factor
        x = F.leaky_relu(self.reg1_bn(self.reg1(x_in)))
        x = torch.cat((x, F.leaky_relu(self.cl1_bn(self.cl1(x_in)))), 1)
        # from (b, c, h, w) to (b, w, h, c) to (b * w * h, c)
        x = x.transpose(1, 3).reshape((-1, x.shape[1]))
        x = F.leaky_relu(self.reg2_cm(x.contiguous(), inds_super))
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
            inds_super = (inds_r/self.class_factor).type(int_type)
            x = F.leaky_relu(self.reg1_bn(self.reg1(x_in)))
            x = torch.cat((x, F.leaky_relu(self.cl1_bn(self.cl1(x_in)))), 1)
            # from (b, c, h, w) to (b, w, h, c) to (b * w * h, c)
            x = x.transpose(1, 3).reshape((-1, x.shape[1]))
            x = F.leaky_relu(self.reg2_cm(x, inds_super))
            regression = self.reg3_cm(x, inds_r)
            regression = regression.reshape((batch_size, 1, 1, -1))

            x = (inds_gt.type(torch.float32) + regression) * (1.0 / float(self.classes))

            if torch.any(torch.isnan(regression)):
                print("regressions: found nan")
            if torch.any(torch.isinf(regression)):
                print("regressions: found inf")

            return x, mask, class_losses, x_real

#same as #5 but without batch normalization
class CR8_reg_cond_mul_6(nn.Module):

    #default parameters are the same as for CR8_reg_cond_mul_2
    def __init__(self, classes=128, superclasses=8, ch_in=128, ch_latent_c=[128, 128], ch_latent_r=[128, 32], concat=True):
        super(CR8_reg_cond_mul_6, self).__init__()
        self.classes = classes
        self.concat = concat
        self.superclasses = superclasses
        self.class_factor = int(classes/superclasses)
        self.cl1 = nn.Conv2d(ch_in, ch_latent_c[0], 1)
        self.cl2 = nn.Conv2d(ch_latent_c[0], ch_latent_c[1], 1)
        self.cl3 = nn.Conv2d(ch_latent_c[1], classes + 1, 1)

        # only one output (regression!!
        # self.cond_mul = RefCondMulConv(classes, input_features=ch_latent, output_features=1)
        # self.cond_mul = RefCondMul(classes, input_features=ch_latent, output_features=1)

        self.reg1 = nn.Conv2d(ch_in, ch_latent_r[0], 1)
        if concat:
            self.reg2_cm = CondMul(superclasses, input_features=ch_latent_r[0] + ch_latent_c[0], output_features=ch_latent_r[1])
        else:
            self.reg2_cm = CondMul(superclasses, input_features=ch_latent_r[0],
                                   output_features=ch_latent_r[1])

        self.reg3_cm = CondMul(classes, input_features=ch_latent_r[1], output_features=1)

    def forward(self, x_in, x_gt=None, mask_gt=None):
        batch_size = x_in.shape[0]
        int_type = torch.int32

        # print(x.shape)
        x = F.leaky_relu(self.cl1(x_in))
        x = F.leaky_relu(self.cl2(x))
        x = self.cl3(x)

        # classes = F.softmax(x[:, 0:self.classes, :, :], dim=1)
        classes = x[:, 0:self.classes, :, :]  # cross entropy already has a softmax
        mask = F.leaky_relu(x[:, [-1], :, :])

        # reshaped latent features:
        inds = classes.argmax(dim=1).unsqueeze(1)

        #print("before first regressions")
        # now do the regressions:
        inds_r = inds.flatten().clamp(0, self.classes - 1).type(int_type)
        inds_super = (inds_r // self.class_factor).clamp(0, self.superclasses - 1)
        x = F.leaky_relu(self.reg1(x_in))
        #print("after linewise!")
        if self.concat:
            x = torch.cat((x, F.leaky_relu(self.cl1(x_in))), 1)
        # from (b, c, h, w) to (b, w, h, c) to (b * w * h, c)
        x = x.transpose(1, 3).reshape((-1, x.shape[1]))

        x = F.leaky_relu(self.reg2_cm(x.contiguous(), inds_super))
        regression = self.reg3_cm(x, inds_r)
        #print("after first regressions")
        # assuming h=1 we get the shape back to (b, c, h, w)
        regression = regression.reshape((batch_size, 1, 1, -1))
        x_real = (inds.type(torch.float32) + regression) * (1.0 / float(self.classes))
        if x_gt is None:
            return x_real, mask
        else:
            inds_gt = (x_gt * self.classes).type(torch.int64).clamp(0, self.classes - 1)
            loss = F.cross_entropy(classes, inds_gt.squeeze(1))
            class_losses = [torch.mean(loss * mask_gt)]

            inds_r = inds_gt.flatten().type(int_type)
            #todo: it is amazing how this needs to be clamped. really there shouldn't be a need to
            inds_super = (inds_r//self.class_factor).type(int_type).clamp(0, self.superclasses - 1)
            x = F.leaky_relu(self.reg1(x_in))
            if self.concat:
                x = torch.cat((x, F.leaky_relu(self.cl1(x_in))), 1)
            # from (b, c, h, w) to (b, w, h, c) to (b * w * h, c)
            x = x.transpose(1, 3).reshape((-1, x.shape[1]))
            x = F.leaky_relu(self.reg2_cm(x, inds_super))
            regression = self.reg3_cm(x, inds_r)

            regression = regression.reshape((batch_size, 1, 1, -1))

            x = (inds_gt.type(torch.float32) + regression) * (1.0 / float(self.classes))

            if torch.any(torch.isnan(regression)):
                print("regressions: found nan")
            if torch.any(torch.isinf(regression)):
                print("regressions: found inf")
            #print("forward done")
            return x, mask, class_losses, x_real




#same as #5 but without batch normalization
class CR8_reg_2stage(nn.Module):

    #default parameters are the same as for CR8_reg_cond_mul_2
    def __init__(self, classes=[32, 32], superclasses=8, ch_in=128,
                 ch_latent_c=[128, 128],
                 ch_latent_r=[128, 32],
                 ch_latent_msk=[32, 16],
                 concat=False):
        super(CR8_reg_2stage, self).__init__()
        overall_classes = classes[0] * classes[1]
        self.classes = classes
        self.concat = concat
        self.superclasses = superclasses
        self.class_factor = int(overall_classes/superclasses)
        # the first latent layer for classification is shared
        self.cl1 = nn.Conv2d(ch_in, ch_latent_c[0], 1)

        self.cl2_1 = nn.Conv2d(ch_latent_c[0], ch_latent_c[1], 1)
        self.cl3_1 = nn.Conv2d(ch_latent_c[1], classes[0], 1)

        self.cl2_2 = CondMul(classes[0], ch_latent_c[0], 32)
        self.cl3_2 = CondMul(classes[0], 32, classes[1])



        # only one output (regression!!
        # self.cond_mul = RefCondMulConv(classes, input_features=ch_latent, output_features=1)
        # self.cond_mul = RefCondMul(classes, input_features=ch_latent, output_features=1)

        self.reg1 = nn.Conv2d(ch_in, ch_latent_r[0], 1)
        if concat:
            self.reg2_cm = CondMul(superclasses, input_features=ch_latent_r[0] + ch_latent_c[0], output_features=ch_latent_r[1])
        else:
            self.reg2_cm = CondMul(superclasses, input_features=ch_latent_r[0],
                                   output_features=ch_latent_r[1])

        self.reg3_cm = CondMul(overall_classes, input_features=ch_latent_r[1], output_features=1)

        # kernels for masks:
        self.msk1 = nn.Conv2d(ch_in, ch_latent_msk[0], 1)
        self.msk2 = nn.Conv2d(ch_latent_msk[0], ch_latent_msk[1], 1)
        self.msk3 = nn.Conv2d(ch_latent_msk[1], 1, 1)



    def forward(self, x_in, x_gt=None, mask_gt=None):
        batch_size = x_in.shape[0]
        int_type = torch.int32
        overall_classes = self.classes[0] * self.classes[1]

        x = F.leaky_relu(self.msk1(x_in))
        x = F.leaky_relu(self.msk2(x))
        mask = F.leaky_relu(self.msk3(x))

        # print(x.shape)
        x = F.leaky_relu(self.cl1(x_in))
        # x_latent for the second step in classification
        x_l = x.transpose(1, 3).reshape((-1, x.shape[1]))
        x = F.leaky_relu(self.cl2_1(x))
        x = self.cl3_1(x)
        cl1 = x # for later/ the cross entropy loss

        inds1 = x.argmax(dim=1).flatten().type(int_type)
        x = F.leaky_relu(self.cl2_2(x_l.contiguous(), inds1))
        x = self.cl3_2(x, inds1)
        inds2 = x.argmax(dim=1).flatten().type(int_type)
        inds = inds1 * self.classes[1] + inds2
        #second


        #print("before first regressions")
        # now do the regressions:
        #todo: remove clamp here! it should not be necessary!!!
        inds_r = inds.clamp(0, overall_classes - 1).type(int_type)
        inds_super = (inds_r // self.class_factor).clamp(0, self.superclasses - 1)
        x = F.leaky_relu(self.reg1(x_in))
        #print("after linewise!")
        if self.concat:
            x = torch.cat((x, F.leaky_relu(self.cl1(x_in))), 1)
        # from (b, c, h, w) to (b, w, h, c) to (b * w * h, c)
        x = x.transpose(1, 3).reshape((-1, x.shape[1]))

        x = F.leaky_relu(self.reg2_cm(x.contiguous(), inds_super))
        regression = self.reg3_cm(x, inds_r)
        #print("after first regressions")
        # assuming h=1 we get the shape back to (b, c, h, w)
        regression = regression.reshape((batch_size, 1, 1, -1))
        inds = inds.reshape(regression.shape)
        x_real = (inds.type(torch.float32) + regression) * (1.0 / float(overall_classes))
        if x_gt is None:
            return x_real, mask
        else:
            inds_gt = (x_gt * overall_classes).type(torch.int64).clamp(0, overall_classes - 1)
            inds1_gt = inds_gt // self.classes[1]
            inds2_gt = torch.remainder(inds_gt, self.classes[1])
            loss1 = F.cross_entropy(cl1, inds1_gt.squeeze(1))

            inds1_gt = inds1_gt.flatten().type(int_type)
            x = F.leaky_relu(self.cl2_2(x_l, inds1_gt))
            x = self.cl3_2(x, inds1_gt)
            x = x.reshape((batch_size, -1, x.shape[1])).permute((0, 2, 1))
            x = x.reshape((batch_size, self.classes[1], 1, -1))
            loss2 = F.cross_entropy(x, inds2_gt.squeeze(1))

            class_losses = [torch.mean(loss1 * mask_gt), torch.mean(loss2 * mask_gt)]

            inds_r = inds_gt.flatten().type(int_type)
            #todo: it is amazing how this needs to be clamped. really there shouldn't be a need to
            inds_super = (inds_r//self.class_factor).type(int_type).clamp(0, self.superclasses - 1)
            x = F.leaky_relu(self.reg1(x_in))
            if self.concat:
                x = torch.cat((x, F.leaky_relu(self.cl1(x_in))), 1)
            # from (b, c, h, w) to (b, w, h, c) to (b * w * h, c)
            x = x.transpose(1, 3).reshape((-1, x.shape[1]))
            x = F.leaky_relu(self.reg2_cm(x, inds_super))
            regression = self.reg3_cm(x, inds_r)

            regression = regression.reshape((batch_size, 1, 1, -1))

            x = (inds_gt.type(torch.float32) + regression) * (1.0 / float(overall_classes))

            if torch.any(torch.isnan(regression)):
                print("regressions: found nan")
            if torch.any(torch.isinf(regression)):
                print("regressions: found inf")
            #print("forward done")
            return x, mask, class_losses, x_real


class Classification3Stage(nn.Module):
    def __init__(self, ch_in=128,
                 classes=[16, 16, 16],
                 pad=[0, 8, 8],  # pad around classes
                 ch_latent=[[32, 32], [32, 32], [32, 32]]):
        super(Classification3Stage, self).__init__()
        self.classes = classes
        self.pad = pad
        classes12 = classes[0] * classes[1]
        self.c1 = nn.ModuleList([nn.Conv2d(ch_in, ch_latent[0][0], 1),
                                 CondMul(classes[0], ch_in, ch_latent[1][0]),
                                 CondMul(classes12, ch_in, ch_latent[2][0])])
        self.c2 = nn.ModuleList([nn.Conv2d(ch_latent[0][0], ch_latent[0][1], 1),
                                 CondMul(classes[0], ch_latent[1][0], ch_latent[1][1]),
                                 CondMul(classes12, ch_latent[2][0], ch_latent[2][1])])
        self.c3 = nn.ModuleList([nn.Conv2d(ch_latent[0][1], classes[0], 1),
                                 CondMul(classes[0], ch_latent[1][1], classes[1] + 2 * pad[1]),
                                 CondMul(classes12, ch_latent[2][1], classes[2] + 2 * pad[2])])

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
                inds2_gt = inds2_gt + self.pad[1]

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
                inds3_gt = inds3_gt + self.pad[2]

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
class CR8_reg_3stage(nn.Module):

    #default parameters are the same as for CR8_reg_cond_mul_2
    def __init__(self, ch_in=128,
                 ch_latent=[128, 128, 128],
                 superclasses=8,
                 ch_latent_r=[128, 32],
                 ch_latent_msk=[32, 16],
                 classes=[16, 16, 16],
                 pad=[0, 8, 8],
                 ch_latent_c=[[32, 32], [32, 32], [32, 32]],
                 regress_neighbours=0):
        super(CR8_reg_3stage, self).__init__()
        classes123 = classes[0] * classes[1] * classes[2]
        self.classes = classes
        self.superclasses = superclasses
        self.class_factor = int(classes123/superclasses)
        self.regress_neighbours = regress_neighbours
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
            #r = self.r2(x_l, inds).flatten()#todo:remove this reactivate the two lines above

            x = (inds.type(torch.float32) + r) * (1.0 / float(classes123))
            #x = (inds.type(torch.float32)) * (1.0 / float(classes123)) #todo remove debug
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
            x_reg_combined = torch.zeros((batch_size, 1 + 2 * self.regress_neighbours, height, width),
                                         device=x_l.device)
            for offset in range(-self.regress_neighbours, self.regress_neighbours+1):
                inds_gt = (inds_gt + offset).clamp(0, classes123 - 1).flatten()
                inds_super = inds_gt // self.class_factor
                x = F.leaky_relu(self.r2(x_l, inds_super))
                r = self.r3(x, inds_gt).flatten()
                #r = self.r2(x_l, inds_gt).flatten()#todo:remove this reactivate the two lines above

                x_reg = (inds_gt.type(torch.float32) + r) * (1.0 / float(classes123))
                x_reg = x_reg.reshape((batch_size, 1, height, width))
                x_reg_combined[:, [offset+self.regress_neighbours], :, :] = x_reg


            #calculate the real x
            inds = inds.flatten().type(torch.int32)
            inds_super = inds // self.class_factor
            x = F.leaky_relu(self.r2(x_l, inds_super))
            r = self.r3(x, inds).flatten()

            x = (inds.type(torch.float32) + r) * (1.0 / float(classes123))
            x_real = x.reshape((batch_size, 1, height, width))

            return x_reg_combined, mask, class_losses, x_real

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

class CR8_bb_short(nn.Module):

    def __init__(self, channels=[16, 32, 64], channels_sub=[64, 64, 64, 64]):
        super(CR8_bb_short, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_start = nn.Conv2d(1, channels[0], 3, padding=(0, 1))  # 1
        self.conv_1 = nn.Conv2d(channels[0], channels[1], 5, padding=(0, 2))  # + 2 = 3
        self.conv_3_down = nn.Conv2d(channels[1], channels[2], 5, padding=(0, 2), stride=(2, 2))  # + 2 = 5
        # subsampled from here!
        self.conv_4 = nn.Conv2d(channels[2], channels_sub[0], 5, padding=(0, 2))  # + 2 * 2 = 9
        self.conv_6 = nn.Conv2d(channels_sub[0], channels_sub[1], 5, padding=(0, 2))  # + 2 * 2 = 13
        self.conv_8 = nn.Conv2d(channels_sub[1], channels_sub[2], 3, padding=(0, 1))  # + 1 * 2 = 15
        self.conv_9 = nn.Conv2d(channels_sub[2], channels_sub[3], 3, padding=(0, 1))  # + 1 * 2 = 17

    def forward(self, x):
        x = F.leaky_relu(self.conv_start(x))
        x = F.leaky_relu(self.conv_1(x))
        x = F.leaky_relu(self.conv_3_down(x))
        x = F.leaky_relu(self.conv_4(x))
        x = F.leaky_relu(self.conv_6(x))
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