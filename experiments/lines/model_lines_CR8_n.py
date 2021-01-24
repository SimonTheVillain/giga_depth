import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_shrink, ResidualBlock_vshrink
from model.cuda_cond_mul.cond_mul import CondMul
from model.cuda_cond_mul.reference_cond_mul import RefCondMul


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

        classes = F.softmax(x[:, 0:self.classes, :, :], dim=1)
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

    def __init__(self, classes, ch_in=128, ch_latent=512):
        self.classes = classes
        super(CR8_reg_cond_mul, self).__init__()
        self.conv_1 = nn.Conv2d(ch_in, ch_latent, 1, padding=0)
        self.bn_1 = nn.BatchNorm2d(ch_latent)
        #only one output (regression!!
        self.cond_mul = CondMul(classes, input_features=ch_latent, output_features=1)
        self.conv_2 = nn.Conv2d(ch_latent, classes + 1, 1, padding=0, padding_mode='replicate')

    def forward(self, x, x_gt=None, mask_gt=None):
        batch_size = x.shape[0]
        int_type = torch.int32

        #print(x.shape)
        x_latent = F.leaky_relu(self.bn_1(self.conv_1(x)))
        x = self.conv_2(x_latent)

        classes = F.softmax(x[:, 0:self.classes, :, :], dim=1)
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

# CR8 backbone just a bit more narrow. To be combined with CR8_reg_light
class CR8_bb_light_01(nn.Module):

    def __init__(self):
        super(CR8_bb_light_01, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_start = nn.Conv2d(1, 16, 3, padding=(0, 1)) #1
        self.resi_block1 = ResidualBlock_vshrink(16, 3, padding=(0, 1), depadding=3) # + 3 = 4
        self.resi_block2 = ResidualBlock_vshrink(16, 3, padding=(0, 1), depadding=3) # + 3 = 7
        self.conv_ch_up_1 = nn.Conv2d(16, 64, 5, padding=(0, 2), stride=(2, 2)) # + 2 = 9
        # subsampled beginning here!
        self.resi_block3 = ResidualBlock_vshrink(64, 3, padding=(0, 1), depadding=3) # + 2 * 3 = 15
        self.conv_end_1 = nn.Conv2d(64, 128, 3, padding=(0, 1)) # + 2 * 1 = 17
        # if this does not work... add one more 1x1 convolutional layer here
        #self.conv_end_2 = nn.Conv2d(128, 512, 1, padding=0)

    def forward(self, x):
        x = F.leaky_relu(self.conv_start(x))
        x = self.resi_block1(x)
        x = self.resi_block2(x)
        x = F.leaky_relu(self.conv_ch_up_1(x))
        x = self.resi_block3(x)
        x = F.leaky_relu(self.conv_end_1(x))
        return x

# CR8 backbone!!!
class CR8_bb_light05(nn.Module):

    def __init__(self):
        super(CR8_bb_light05, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv_start = nn.Conv2d(1, 16, 3, padding=(0, 1))  # 1
        self.conv_ch_up_1 = nn.Conv2d(16, 32, 5, padding=(0, 2), stride=(2, 2))  # + 2 = 3

        self.resi_block1 = ResidualBlock_vshrink(32, 3, padding=(0, 1), depadding=3)  # + 2*3 = 9
        self.conv_ch_up_2 = nn.Conv2d(32, 64, 3, padding=(0, 1))  # + 2*1 = 11
        self.resi_block2 = ResidualBlock_vshrink(64, 3, padding=(0, 1), depadding=3)  # + 2*3 = 17
        self.conv_end_1 = nn.Conv2d(64, 128, 1, padding=(0, 0))  # + 2 * 0 = 17


    def forward(self, x):
        x = F.leaky_relu(self.conv_start(x))
        x = F.leaky_relu(self.conv_ch_up_1(x))
        x = self.resi_block1(x)
        x = F.leaky_relu(self.conv_ch_up_2(x))
        x = self.resi_block2(x)
        x = F.leaky_relu(self.conv_end_1(x))
        return x

#almost like CR9hs
class CR8_bb_light(nn.Module):

    def __init__(self, channels_out=128):
        super(CR8_bb_light, self).__init__()
        self.conv_start = nn.Conv2d(1, 16, 3, padding=(0, 1))  # 1
        self.conv_ds1 = nn.Conv2d(16, 32, 5, padding=(0, 2), stride=2)  # + 2 = 3
        self.conv_1 = nn.Conv2d(32, 32, 3, padding=(0, 1), stride=1)  # + 2 x 1 = 5
        self.conv_2 = nn.Conv2d(32, 32, 3, padding=(0, 1), stride=1)  # + 2 x 1 = 7
        self.conv_3 = nn.Conv2d(32, 32, 3, padding=(0, 1), stride=1)  # + 2 x 1 = 9
        self.conv_4 = nn.Conv2d(32, 64, 3, padding=(0, 1), stride=1)  # + 2 x 1 = 11
        self.conv_5 = nn.Conv2d(64, 64, 3, padding=(0, 1), stride=1)  # + 2 x 1 = 13
        self.conv_6 = nn.Conv2d(64, 64, 3, padding=(0, 1), stride=1)  # + 2 x 1 = 15
        self.conv_7 = nn.Conv2d(64, channels_out, 3, padding=(0, 1), stride=1, groups=1 + 0 * 64)  # + 2 x 1 = 17

    def forward(self, x):
        x = F.leaky_relu(self.conv_start(x))
        x = F.leaky_relu(self.conv_ds1(x)) # downsample here
        x = F.leaky_relu(self.conv_1(x)) #feed forward here
        d = 4
        x_ff = x[:, :, d:-d, :]
        x = F.leaky_relu(self.conv_2(x))
        x = F.leaky_relu(self.conv_3(x))
        x = F.leaky_relu(self.conv_4(x))
        x = self.conv_5(x)
        x[:, 0:x_ff.shape[1], :, :] += x_ff #skip connection here!
        x = F.leaky_relu(x)
        x = F.leaky_relu(self.conv_6(x))
        x = F.leaky_relu(self.conv_7(x))
        return x

class CR8_reg_light(nn.Module):

    def __init__(self, channels_in=128, classes=32):
        self.classes = classes
        super(CR8_reg_light, self).__init__()
        self.conv_end_2 = nn.Conv2d(channels_in, 512, 1, padding=0)
        self.conv_end_3 = nn.Conv2d(512, classes*2+1, 1, padding=0)

    def forward(self, x, x_gt=None, mask_gt=None):
        ### LAYER 0
        x = F.softmax(self.conv_end_2(x), dim=1)
        x = self.conv_end_3(x)

        classes = F.softmax(x[:, 0:self.classes, :, :], dim=1)
        regressions = x[:, self.classes:(2 * self.classes), :, :]
        #todo: use a softplus instead of a relu here!
        sigma = F.softplus(x[:, [-1], :, :])#formerly leaky_relu

        inds = classes.argmax(dim=1).unsqueeze(1)

        if x_gt is None:
            regressions = torch.gather(regressions, dim=1, index=inds)
            x = (inds.type(torch.float32) + regressions) * (1.0 / float(self.classes))
            return x, sigma
        else:
            inds_gt = (x_gt * self.classes).type(torch.int64)
            loss = F.cross_entropy(classes, inds_gt.squeeze(1))
            class_losses = [torch.mean(loss * mask_gt)]

            regressions = torch.gather(regressions, dim=1, index=inds_gt)
            x = (inds_gt.type(torch.float32) + regressions) * (1.0 / float(self.classes))

            regressions = torch.gather(regressions, dim=1, index=inds)
            x_real = (inds.type(torch.float32) + regressions) * (1.0 / float(self.classes))
            return x, sigma, class_losses, x_real

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