import torch
import torch.nn as nn
import torch.nn.functional as F
from model.residual_block import ResidualBlock_shrink


#same as CR10_4 but with better per weight
class Model_CR10_5_hsn(nn.Module):
    @staticmethod
    def padding():
        return 0

    def __init__(self, classes, image_height, shallow=False):
        super(Model_CR10_5_hsn, self).__init__()
        self.classes = classes
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

            self.conv_start = nn.Conv2d(1, 16, 3, padding=1)  # 1
            self.conv_ds1 = nn.Conv2d(16, 32, 5, padding=2, stride=2, groups=1 + 0 * 16)  # + 2 = 3
            self.conv_1 = nn.Conv2d(32, 32, 3, padding=1, stride=1)  # + 2 x 1 = 5
            self.conv_2 = nn.Conv2d(32, 32, 3, padding=1, stride=1, groups=1 + 0 * 32)  # + 2 x 1 = 7
            self.conv_3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)  # + 2 x 1 = 9
            self.conv_4 = nn.Conv2d(64, 64, 3, padding=1, stride=1, groups=1 + 0 * 64)  # + 2 x 1 = 11
            self.conv_5 = nn.Conv2d(64, 64, 3, padding=1, stride=1)  # + 2 x 1 = 13
            self.conv_6 = nn.Conv2d(64, 128, 3, padding=1, stride=1, groups=1 + 0 * 128)  # + 2 x 1 = 17




        half_height = int(image_height / 2)
        # 1x1 convolution for classification with linewise weights
        self.conv_end_c = nn.Conv2d(128 * half_height, self.classes * half_height, 1, padding=0, groups=half_height)
        self.conv_end_r = nn.Conv2d(128 * half_height, self.classes * half_height, 1, padding=0, groups=half_height)

        #old slicewise code for classes
        #self.conv_end_c = nn.ModuleList()
        #for i in range(0, self.slices):
        #    self.conv_end_c.append(nn.Conv2d(128, self.classes, 1, padding=0, groups=1))

        # if this does not work... add one more 1x1 convolutional layer here
        # the line-wise version of the class prediction:
        # self.conv_end_c = nn.Conv2d(128 * half_height, classes * half_height, 1,
        #                            padding=0, groups=half_height)
        #self.conv_end_r = nn.Conv2d((128 + 256 + self.classes) * half_height, classes * half_height, 1,
        #                            padding=0, groups=half_height)
        #self.reg_w_1 = torch.zeros((half_height * classes, 128, 1))
        #self.reg_b_1 = torch.zeros((half_height * classes, 1, 1))


        #self.reg_w_2 = torch.zeros((half_height * classes, 4, 1))
        #self.reg_b_2 = torch.zeros((half_height * classes, 1, 1))
        self.conv_end_m = nn.Conv2d(128, 1, 1,
                                    padding=0, groups=1)

    def copy_backbone(self, other):
        if self.shallow != other.shallow:
            return

        self.conv_start.weight.data = other.conv_start.weight.data
        self.conv_start.bias.data   = other.conv_start.bias.data
        self.conv_ds1.weight.data   = other.conv_ds1.weight.data
        self.conv_ds1.bias.data     = other.conv_ds1.bias.data
        self.conv_1.weight.data     = other.conv_1.weight.data
        self.conv_1.bias.data       = other.conv_1.bias.data
        self.conv_2.weight.data     = other.conv_2.weight.data
        self.conv_2.bias.data       = other.conv_2.bias.data
        self.conv_3.weight.data     = other.conv_3.weight.data
        self.conv_3.bias.data       = other.conv_3.bias.data
        self.conv_4.weight.data     = other.conv_4.weight.data
        self.conv_4.bias.data       = other.conv_4.bias.data
        self.conv_5.weight.data     = other.conv_5.weight.data
        self.conv_5.bias.data       = other.conv_5.bias.data
        self.conv_6.weight.data     = other.conv_6.weight.data
        self.conv_6.bias.data       = other.conv_6.bias.data



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
        #class_shape = (input.shape[0], self.classes, x.shape[2], x.shape[3])
        #classes = torch.zeros(class_shape, device=device)
        #step = int(self.height / self.slices)
        #half_step = int(step / 2)
        half_height = int(self.height / 2)
        #classes we do slicewise
        #for i in range(0, self.slices):
        #    s = x[:, :, (i * half_step):((i + 1) * half_step), :]
        #    s = F.leaky_relu(self.conv_end_c[i](s))
        #    classes[:, :, i * half_step:(i + 1) * half_step, :] = s
        #classes = F.softmax(classes, dim=1)

        #classes slicewise


        x_1 = x.transpose(1, 2)
        x_1 = x_1.reshape((x_1.shape[0], 128 * half_height, 1, x_1.shape[3]))
        classes = F.leaky_relu(self.conv_end_c(x_1))
        classes = classes.reshape((classes.shape[0], half_height, self.classes, classes.shape[3]))
        classes = classes.transpose(1, 2)
        classes = F.softmax(classes, dim=1)

        regressions = self.conv_end_r(x_1)
        regressions = regressions.reshape((regressions.shape[0], half_height, self.classes, regressions.shape[3]))
        regressions = regressions.transpose(1, 2)




        # mask generation is straight forward
        mask = F.leaky_relu(self.conv_end_m(x))

        #regression:
        if class_gt is None:
            inds = classes.argmax(dim=1).unsqueeze(1)
        else:
            inds = class_gt

        regressions = regressions.gather(1, inds)


        return classes, regressions, mask, x_latent


