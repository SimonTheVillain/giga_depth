import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_shrink


# in case we subsample rather later:
# 640×480×4×(1024+513+ 128+4×(128*4 + 64×7 + 1)) would result in a bit below 7GB of video memory.
# in case we subsaple at the first pissible position:
# 640×480×4×(1024+513+ 128*5+4×(64×7 + 1)) would result in a bit below 5GB of video memory.
# (conv_ch_up_1 would have a stride of 2)
class Model_CR10_2_hsn(nn.Module):
    @staticmethod
    def padding():
        return 0  # 15 when striding at first opportunity 27 when at second

    def __init__(self, slices, classes, image_height, shallow=False):
        super(Model_CR10_2_hsn, self).__init__()
        self.slices = slices
        self.classes = classes
        self.r = self.padding()
        self.r_top = 0
        self.r_bottom = 0
        self.height = int(image_height)
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel

        self.shallow = shallow
        if self.shallow:
            self.conv_start = nn.Conv2d(1, 16, 3, padding=1)  # 1
            self.conv_ds1 = nn.Conv2d(16, 32, 5, padding=2, stride=2, groups=1 + 0 * 16)  # + 2 = 3
            self.conv_1 = nn.Conv2d(32, 32, 3, padding=1, stride=1)  # + 2 x 1 = 5
            self.conv_2 = nn.Conv2d(32, 64, 3, padding=1, stride=1, groups=1 + 0 * 32)  # + 2 x 1 = 7
            self.conv_3 = nn.Conv2d(64, 64, 5, padding=2, stride=1)  # + 2 x 2 = 11
            self.conv_4 = nn.Conv2d(64, 128, 5, padding=2, stride=1, groups=1 + 0 * 64)  # + 2 x 2 = 15

        else:
            self.conv_start = nn.Conv2d(1, 16, 3, padding=1)  # 1
            self.conv_ds1 = nn.Conv2d(16, 32, 5, padding=2, stride=2, groups=1 + 0 * 16)  # + 2 = 3
            self.conv_1 = nn.Conv2d(32, 32, 3, padding=1, stride=1)  # + 2 x 1 = 5
            self.conv_2 = nn.Conv2d(32, 32, 3, padding=1, stride=1, groups=1 + 0 * 32)  # + 2 x 1 = 7
            self.conv_3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)  # + 2 x 1 = 9
            self.conv_4 = nn.Conv2d(64, 64, 3, padding=1, stride=1, groups=1 + 0 * 64)  # + 2 x 1 = 11
            self.conv_5 = nn.Conv2d(64, 128, 3, padding=1, stride=1)  # + 2 x 1 = 13
            self.conv_6 = nn.Conv2d(128, 128, 3, padding=1, stride=1, groups=1 + 0 * 128)  # + 2 x 1 = 17

        self.conv_end_c = nn.ModuleList()

        for i in range(0, self.slices):
            self.conv_end_c.append(nn.Conv2d(128, self.classes, 1, padding=0, groups=1))

        # if this does not work... add one more 1x1 convolutional layer here
        half_height = int(image_height / 2)
        # the line-wise version of the class prediction:
        # self.conv_end_c = nn.Conv2d(128 * half_height, classes * half_height, 1,
        #                            padding=0, groups=half_height)
        #self.conv_end_r = nn.Conv2d((128 + 256 + self.classes) * half_height, classes * half_height, 1,
        #                            padding=0, groups=half_height)
        #self.reg_w_1 = torch.zeros((half_height * classes, 128, 1))
        #self.reg_b_1 = torch.zeros((half_height * classes, 1, 1))

        self.register_parameter(name='reg_w_1', param=nn.Parameter(torch.randn((half_height * classes, 128, 1))))
        self.register_parameter(name='reg_b_1', param=nn.Parameter(torch.randn((half_height * classes, 1, 1))))

        #self.reg_w_2 = torch.zeros((half_height * classes, 4, 1))
        #self.reg_b_2 = torch.zeros((half_height * classes, 1, 1))
        self.conv_end_m = nn.Conv2d(128, 1, 1,
                                    padding=0, groups=1)

    def forward(self, x, class_gt=None):
        device = x.device
        input = x
        # print(x.shape)

        x = F.leaky_relu(self.conv_start(x))
        x = F.leaky_relu(self.conv_ds1(x))
        x = F.leaky_relu(self.conv_1(x))
        x = F.leaky_relu(self.conv_2(x))
        x = F.leaky_relu(self.conv_3(x))
        x = F.leaky_relu(self.conv_4(x))
        if not self.shallow:
            x = F.leaky_relu(self.conv_5(x))
            x = F.leaky_relu(self.conv_6(x))

        
        x_latent = x.clone()
        #return x
        ### LAYER 0
        class_shape = (input.shape[0], self.classes, x.shape[2], x.shape[3])
        classes = torch.zeros(class_shape, device=device)
        step = int(self.height / self.slices)
        half_step = int(step / 2)
        half_height = int(self.height / 2)

        #classes we do slicewise
        for i in range(0, self.slices):
            s = x[:, :, (i * half_step):((i + 1) * half_step), :]
            s = F.leaky_relu(self.conv_end_c[i](s))
            classes[:, :, i * half_step:(i + 1) * half_step, :] = s
        classes = F.softmax(classes, dim=1)

        # mask generation is straight forward
        mask = F.leaky_relu(self.conv_end_m(x))

        #regression:
        if class_gt is None:
            inds = classes.argmax(dim=1).unsqueeze(1)
        else:
            inds = class_gt

        offset = torch.arange(0, half_height, device=device)
        offset = offset.unsqueeze(0).unsqueeze(0).unsqueeze(3)
        ind_shape = inds.shape
        inds = inds + offset * self.classes
        inds = inds.reshape(-1)

        b = self.reg_b_1.data.index_select(0, inds)
        w = self.reg_w_1.data.index_select(0, inds)
        b = b.reshape((ind_shape[0], ind_shape[2], ind_shape[3], b.shape[1], b.shape[2]))
        w = w.reshape((ind_shape[0], ind_shape[2], ind_shape[3], w.shape[1], w.shape[2]))

        x = x.permute((0, 2, 3, 1)).unsqueeze(3)
        x = torch.matmul(x, w) + b
        x = x.permute((0, 3, 1, 2, 4)).squeeze(4)

        regressions = x

        return classes, regressions, mask, x_latent


