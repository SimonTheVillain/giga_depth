import torch
import torch.nn as nn
import torch.nn.functional as F
from model.cuda_cond_mul.cond_mul import CondMul
from model.cuda_cond_mul.reference_cond_mul import RefCondMul
from model.residual_block import ResidualBlock_shrink


#Regressor
class Regressor1Stage(nn.Module):

    def __init__(self, input_channels=64, height=448, width=608):
        super(Regressor1Stage, self).__init__()
        #per line weights for classes
        self.height = int(height)
        self.width = int(width)
        self.input_channels = input_channels
        self.stage_1_classes = 32
        #the first stage is supposed to output 32 classes (+ in future 32 variables that might help in the next steps)
        # TODO: to have more than 32 "efficient" outputs. cond_mul_cuda_kernel.cu needs code to support these cases.
        self.prestage_1 = nn.Conv2d(input_channels * self.height,
                                 input_channels * self.height,
                                 1, padding=0, groups=self.height)
        self.prestage_2 = nn.Conv2d(input_channels * self.height,
                                 input_channels * self.height,
                                 1, padding=0, groups=self.height)
        self.stage_1 = nn.Conv2d(input_channels * self.height,
                                 self.stage_1_classes * self.height,
                                 1, padding=0, groups=self.height)
        self.stage_regression = RefCondMul(self.stage_1_classes * self.height,
                                        input_channels, 2)



    def calc_x_pos(self, class_inds, regression):
        x = (class_inds + regression) * (1.0 / self.stage_1_classes)
        return x

    def calc_inds(self, height, inds, div_h, div_inds):
        device = inds.device
        offset = torch.arange(0, int(height//div_h), device=device)
        offset = offset.unsqueeze(1).repeat(1, div_h).flatten()
        offset = offset.unsqueeze(0).unsqueeze(0).unsqueeze(3)
        #ind_shape = inds.shape
        #inds_original = inds
        inds = (inds + offset * self.classes) // div_inds
        inds = inds.reshape(-1).type(torch.int32)
        return inds
    def forward(self, x, x_gt=None, mask_gt=None):
        batches = x.shape[0]
        device = x.device
        # go from (b, c, h, w) to (b, h, c, w)
        x_1 = x.transpose(1, 2)

        # go from (b, h, c, w) to (b, h * c, 1, w)
        x_1 = x_1.reshape((x_1.shape[0], self.input_channels * self.height, 1, x_1.shape[3]))
        #convolution with new weights for each "line"
        x_1 = F.leaky_relu(self.prestage_1(x_1))
        x_1 = F.leaky_relu(self.prestage_2(x_1))
        classes1 = F.leaky_relu(self.stage_1(x_1))

        # go from (b, h*c, 1, w) to (b, h, c, w)
        classes1 = classes1.reshape((classes1.shape[0], self.height, self.stage_1_classes, classes1.shape[3]))

        # get the classes //TODO: why unsqueeze/ adding dimension?
        inds1 = classes1.argmax(dim=2).unsqueeze(1)

        if x_gt is not None:
            inds1 = (x_gt * self.stage_1_classes).type(torch.int32)
            inds1 = inds1.clamp(0, self.stage_1_classes - 1)

            # the cross entropy takes the estimate in (b, C, h, w) the gt (inds) in (b, 1, h, w)
            classes1 = F.softmax(classes1, 1).permute((0, 2, 1, 3))
            # reduce inds1
            loss = F.cross_entropy(classes1, inds1.type(torch.int64).squeeze(1))
            #go back from (b, c, h, w) to (b, h, c, w)
            #classes1 = classes1.permute((0, 2, 1, 3))
            class_losses = [torch.mean(loss * mask_gt)]

        # go from (b, 1, h, w) to (b, h , 1, w)?
        #inds1 = inds1.transpose(1, 2)#TODO: is this necessary?

        # offsets for each line!
        offset = torch.arange(0, self.height, device=device)
        offset = offset.unsqueeze(0).unsqueeze(0).unsqueeze(3)

        inds1 += offset * self.stage_1_classes

        # go from (b, c, h, w) to (b, h, w, c)
        x_2 = x.permute([0, 2, 3, 1])
        # to (b * h * w, c)
        x_2 = x_2.reshape((x_2.shape[0] * x_2.shape[1] * x_2.shape[2], x_2.shape[3]))
        inds = inds1.reshape(-1).type(torch.int64)

        if torch.any(inds < 0) or torch.any(inds >= self.height * self.stage_1_classes):
            print("big mistake, indices are out of bounds!!!")
            return
        #print(id(inds))

        x_2 = F.leaky_relu(self.stage_regression(x_2.contiguous(), inds))

        # (b * h * w, 2) to (b, h, w, 2)
        x_2 = x_2.reshape((batches, self.height, self.width, 2))
        # (b, h, w, 2) to (b, 2, h, w)
        x_2 = x_2.permute([0, 3, 1, 2])

        #print(id(inds))

        inds = inds.reshape(batches, self.height, self.width) #todo: something not right here!!!
        #todo: subtract the offset from the inds here... otherwise the calculation down there would be wrong
        # the output lies between 0 and 1 to indicate the x position in the dot-pattern projector
        x = (inds.float() + x_2[:, 0, :, :]) * (1.0 / self.stage_1_classes)

        # one last relu for the masking
        mask = F.leaky_relu(x_2[:, 1, :, :])# TODO: no relu

        #TODO: find out what else we need here!
        if x_gt is None:
            return x, mask
        else:
            return x, mask, class_losses


