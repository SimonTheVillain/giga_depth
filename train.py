import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.optim as optim
from dataset.dataset_rendered_2 import DatasetRendered2, GetDataset
from model.regressor_2Stage import Regressor2Stage
from model.regressor_1Stage import Regressor1Stage
from model.regressor_1branch import Regressor1Branch
from model.regressor_branchless import RegressorBranchless
from model.backbone_6_64 import Backbone6_64
from experiments.lines.model_lines_CR8_n import *
from model.backbone import Backbone, BackboneSliced, BackboneSliced2
from model.regressor import Regressor, Regressor2, Reg_3stage
from torch.utils.data import DataLoader
import math
import argparse

from torch.utils.tensorboard import SummaryWriter


class CompositeModel(nn.Module):
    def __init__(self, backbone, regressor):
        super(CompositeModel, self).__init__()
        self.backbone = backbone
        self.regressor = regressor

    def forward(self, x, x_gt=None, mask_gt=None):
        x = self.backbone(x)
        return self.regressor(x, x_gt, mask_gt)


def sigma_loss(sigma, x, x_gt): #sigma_sq is also called "variance"
    if torch.any(torch.isnan(x)):
        print("x: found nan")
    if torch.any(torch.isinf(x)):
        print("x: found inf")
    if torch.any(torch.isnan(x_gt)):
        print("x_gt: found nan")
    if torch.any(torch.isinf(x_gt)):
        print("x_gt: found inf")
    delta = torch.abs(x - x_gt)
    if torch.any(torch.isnan(delta)):
        print("delta: found nan")
    if torch.any(torch.isinf(delta)):
        print("delta: found inf")
    if torch.any(torch.isnan(sigma)):
        print("sigma: found nan")
    if torch.any(torch.isinf(sigma)):
        print("sigma: found inf")
    eps = 0.01 # sqrt(0.00001) = 0.0003 ... that means we actually have approx 0.3 pixel of basic offset for sigma
    term1 = torch.div(torch.square((x - x_gt) * 1024.0), sigma*sigma + eps)
    term2 = torch.log(sigma * sigma + eps)
    # term 3 is to stay positive(after all it is sigma^2)
    term3 = 0#F.relu(-sigma)
    loss = term1 + term2 + term3
    #print("term1")
    #print(term1)
    #print("term2")
    #print(term2)
    #print("max term1")
    #print(torch.max(term1))
    #print("max term2")
    #print(torch.max(term2))
    #print("max sigma:")
    #print(torch.max(torch.abs(sigma_sq)))
    #print("term 1 den:")
    #print((2*torch.min(torch.abs(sigma_sq)) + eps))
    if torch.any(torch.isnan(term1)):
        print("term1: found nan")
    if torch.any(torch.isinf(term1)):
        print("term1: found inf")

    if torch.any(torch.isnan(term2)):
        print("term2: found nan")
    if torch.any(torch.isinf(term2)):
        print("term2: found inf")
    return torch.mean(loss) #torch.tensor(0)#


def train():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-V", "--version", help="show program version", action="store_true")
    parser.add_argument("-d", "--dataset_path", dest="path", action="store",
                        help="Path to the dataset.",
                        default=os.path.expanduser("~/datasets/structure_core_unity"))
    parser.add_argument("-n", "--npy_dataset", dest="is_npy", action="store_const",
                        help="Loads data directly form numpy files",
                        default=False, const=True)
    parser.add_argument("-b", "--batch_size", dest="batch_size", action="store",
                        help="The batch size during training",
                        type=int,
                        default=4)
    parser.add_argument("-e", "--experiment_name", dest="experiment_name", action="store",
                        help="The name of this training for tensorboard and checkpoints.",
                        default="result")
    parser.add_argument("-w", "--num_workers", dest="num_workers", action="store",
                        help="The number of threads working on loading and preprocessing data.",
                        type=int,
                        default=8)
    parser.add_argument("-g", "--gpu_list", dest="gpu_list", action="store",
                        nargs="+", type=int,
                        default=list(range(0, torch.cuda.device_count())))
    args = parser.parse_args()
    main_device = f"cuda:{args.gpu_list[0]}"# torch.cuda.device(args.gpu_list[0])
    #experiment_name = "cr8_2021_256_wide_reg_alpha10"
    args.experiment_name = "bb64_16_14_12c123_16x2each_lbb64x1_42sc_64_64_reg_lr01_alpha200_1nn"

    writer = SummaryWriter(f"tensorboard/{args.experiment_name}")

    # slit loading and storing of models for Backbone and Regressor
    load_regressor = "trained_models/line_bb64_16_14_12c123_32_32_32_64bb_42sc_64_128_reg_lr01_alpha50_1nn_regressor_chk.pt"
    load_backbone = "trained_models/line_bb64_16_14_12c123_32_32_32_64bb_42sc_64_128_reg_lr01_alpha50_1nn_backbone_chk.pt"

    # not loading any pretrained part of any model whatsoever
    load_regressor = ""
    load_backbone = ""

    num_epochs = 5000
    # todo: either do only 100 lines at a time, or go for
    tgt_res = (1216, 896)
    slice_in = (100, 128)
    slice_gt = (50, 64)
    #alpha = 1.0 * (1.0 / 4.0) * 1.0  # usually this is 0.1
    alpha = 200.0 #todo: back to 10 for quicker convergence!?
    alpha_sigma = 0#1e-10  # how much weight do we give correct confidence measures
    #learning_rate = 0.2 # learning rate of 0.2 was sufficient for many applications
    learning_rate = 0.01  # 0.02 for the branchless regressor (smaller since we feel it's not entirely stable)
    momentum = 0.90
    shuffle = True
    slice = False
    if slice:
        height = 64
    else:
        height = 448

    if load_regressor != "":
        regressor = torch.load(load_regressor)
        regressor.eval()
    else:
        #regressor = RegressorBranchless(height=1)
        #regressor = Regressor(classes=128, height=int(tgt_res[1]/2), ch_in=128, ch_latent_c=[128, 128])
        # regressor = Regressor2Stage()
        # regressor = Regressor1Stage(height=1)
        # regressor = Regressor1Branch(height=1)
        #regressor = CR8_reg_2_stage([16, 16], ch_latent=128)
        #regressor = CR8_reg_cond_mul_5(256, 32, ch_latent_c=[128, 128], ch_latent_r=[128, 4])
        if args.is_npy:
            #regressor = CR8_reg_cond_mul_6(classes=2048, superclasses=32, ch_in=128,
            #                               ch_latent_c=[128, 128],
            #                               ch_latent_r=[128, 32], concat=False)
            #regressor = CR8_reg_2stage(classes=[32, 32], superclasses=8, ch_in=128,
            #                           ch_latent_c=[128, 128],
            #                           ch_latent_r=[128, 32],
            #                           ch_latent_msk=[32, 16])
            regressor = CR8_reg_3stage(ch_in=64,
                                       ch_latent=[64, 64, 64],
                                       superclasses=42,#672*4,#672, #168,
                                       ch_latent_r=[64, 128],
                                       ch_latent_msk=[32, 16],
                                       classes=[16, 14, 12],
                                       pad=[0, 1, 2],
                                       ch_latent_c=[[32, 32], [32, 32], [32, 32]],
                                       regress_neighbours=1)
        else:
            #regressor = Regressor2(classes=256, superclasses=16, height=int(slice_gt[1]), ch_in=64,
            #                       ch_latent_c=[128, 128, 128], ch_latent_r=[256, 8])

            regressor = Reg_3stage(ch_in=64,
                                   height=height,#64,#448,
                                   ch_latent=[],#[128, 128, 128],#todo: make this of variable length
                                   superclasses=42,
                                   ch_latent_r=[32, 4],# 64/64
                                   ch_latent_msk=[32, 16],
                                   classes=[16, 14, 12],
                                   pad=[0, 1, 2],
                                   ch_latent_c=[[16, 16], [16, 16], [16, 16]],#todo: make these of variable length
                                   regress_neighbours=1)
            #classification is lacking:
            #TODO: maybe we have more channels here. [128, 256]
            # for classification one could split lines into groups
            # maybe by just splitting and stacking up the lines

            #regressor = Regressor(classes=128, height=int(slice_gt[1]), ch_in=128, ch_latent_c=[128, 128])

    if load_backbone != "":
        backbone = torch.load(load_backbone)
        backbone.eval()
        # fix parameters in the backbone (or maybe not!)
        # for param in backbone.parameters():
        #    param.requires_grad = False
    else:
        if args.is_npy:
            #backbone = CR8_bb_short(channels=[16, 32, 64], channels_sub=[64, 64, 128, 128])
            backbone = CR8_bb_short(channels=[16, 32, 64], channels_sub=[64, 64, 64, 64])
            #backbone = CR8_bb_short(channels=[8, 16, 32], channels_sub=[32, 32, 32, 32])
        else:
            #backbone = BackboneSliced(slices=1, height=slice_in[1])
            backbone = BackboneSliced2(slices=1, height=height*2,#896,#int(slice_in[1]),
                                       channels=[16, 32, 64], channels_sub=[64, 64, 64, 64, 64])

    model = CompositeModel(backbone, regressor)

    if len(args.gpu_list) > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=args.gpu_list)

    model.to(args.gpu_list[0])

    # for param_tensor in net.state_dict():
    #    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # the whole unity rendered dataset

    # the filtered dataset
    #scale = 1 # set to 2 when using numpy
    #datasets = {
    #    'train': DatasetRendered2(args.path, 0*scale, 20000*scale, tgt_res=tgt_res, is_npy=args.is_npy),
    #    'val': DatasetRendered2(args.path, 20000*scale, 20500*scale, tgt_res=tgt_res, is_npy=args.is_npy)
    #}
    datasets = GetDataset(args.path, is_npy=args.is_npy, tgt_res=tgt_res)

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size,
                                                  shuffle=shuffle, num_workers=args.num_workers)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    min_test_disparity = 100000.0
    step = 0
    for epoch in range(1, num_epochs):
        # TODO: setup code like so: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            print(phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # TODO: accumulate classification losses for each classification stage!!!
            loss_disparity_acc = 0
            loss_reg_acc = 0
            loss_sigma_acc = 0
            loss_class_acc = []
            loss_disparity_acc_sub = 0
            loss_reg_acc_sub = 0
            loss_sigma_acc_sub = 0
            loss_class_acc_sub = []

            for i_batch, sampled_batch in enumerate(dataloaders[phase]):
                step = step + 1
                ir, x_gt, mask_gt = sampled_batch
                ir = ir.to(main_device)
                mask_gt = mask_gt.to(main_device)
                x_gt = x_gt.to(main_device)
                #todo: instead of a query the slice variable should be set accordingly further up!
                if not args.is_npy and slice:
                    ir = ir[:, :, slice_in[0]:slice_in[0] + slice_in[1], :]
                    x_gt = x_gt[:, :, slice_gt[0]:slice_gt[0] + slice_gt[1], :]
                    mask_gt = mask_gt[:, :, slice_gt[0]:slice_gt[0] + slice_gt[1], :]


                if phase == 'train':
                    torch.autograd.set_detect_anomaly(True)
                    x, sigma, class_losses, x_real = model(ir, x_gt, mask_gt)
                    x_real = x_real.detach()

                    optimizer.zero_grad()
                    loss = torch.mean(torch.abs(x - x_gt)) #mask_gt
                    mask_mean = mask_gt.mean().item() + 0.0001
                    masked_reg = torch.mean(torch.abs(x-x_gt) * mask_gt) * (1.0/mask_mean)
                    loss_reg_acc += masked_reg.item()
                    loss_reg_acc_sub += masked_reg.item()

                    loss = loss * alpha

                    if alpha_sigma != 0.0:
                        loss_sigma = sigma_loss(sigma, x_real, x_gt)
                        loss_sigma_acc += loss_sigma.item()
                        loss_sigma_acc_sub += loss_sigma.item()
                        loss += loss_sigma * alpha_sigma

                    if len(loss_class_acc) == 0:
                        loss_class_acc = [0] * len(class_losses)
                        loss_class_acc_sub = [0] * len(class_losses)

                    for i, class_loss in enumerate(class_losses):
                        loss += torch.mean(class_loss)
                        loss_class_acc[i] += torch.mean(class_loss).item()
                        loss_class_acc_sub[i] += torch.mean(class_loss).item()

                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        x_real, sigma = model(ir)
                        #loss = torch.mean(torch.abs(x - x_gt)) #mask_gt
                        #loss_disparity_acc += loss.item()
                        #loss_disparity_acc_sub += loss.item()

                        if alpha_sigma != 0.0:
                            loss_sigma = sigma_loss(sigma, x, x_gt)
                            loss_sigma_acc += loss_sigma.item()
                            loss_sigma_acc_sub += loss_sigma.item()

                delta_disp = torch.mean(torch.abs(x_real - x_gt)).item()
                loss_disparity_acc += delta_disp
                loss_disparity_acc_sub += delta_disp


                if i_batch % 100 == 99:
                    writer.add_scalar(f'{phase}_subepoch/disparity_error',
                                      loss_disparity_acc_sub / 100.0 * 1024, step)
                    if alpha_sigma != 0.0:
                        writer.add_scalar(f'{phase}_subepoch/sigma_loss',
                                          loss_sigma_acc_sub / 100.0, step)

                    if phase == 'train':
                        writer.add_scalar(f'{phase}_subepoch/regression_error', loss_reg_acc_sub / 100.0 * 1024, step)


                        combo_loss = loss_reg_acc_sub * alpha + \
                                     loss_sigma_acc_sub * alpha_sigma + sum(loss_class_acc_sub)
                        print("batch {} loss: {}".format(i_batch, combo_loss / 100))
                    else: #val
                        print("batch {} disparity error: {}".format(i_batch, loss_disparity_acc_sub / 100*1024))

                    for i, class_loss in enumerate(loss_class_acc_sub):
                        writer.add_scalar(f'{phase}_subepoch/class_loss_{i}', class_loss / 100, step)
                        loss_class_acc_sub[i] = 0
                    loss_disparity_acc_sub = 0
                    loss_sigma_acc_sub = 0
                    loss_reg_acc_sub = 0

            writer.add_scalar(f"{phase}/disparity",
                              loss_disparity_acc / dataset_sizes[phase] * args.batch_size * 1024, step)
            if alpha_sigma != 0.0:
                writer.add_scalar(f"{phase}/sigma(loss)",
                                  loss_sigma_acc / dataset_sizes[phase] * args.batch_size, step)
            for i, class_loss in enumerate(loss_class_acc):
                #epoch_loss += class_loss / dataset_sizes[phase] * args.batch_size
                writer.add_scalar(f"{phase}/class_loss{i}", class_loss / dataset_sizes[phase] * args.batch_size, step)
            if phase == 'train':
                writer.add_scalar(f"{phase}/regression_stage",
                                  loss_reg_acc / dataset_sizes[phase] * args.batch_size * 1024, step)
                combo_loss = loss_reg_acc * alpha + loss_sigma_acc * alpha_sigma + sum(loss_class_acc_sub)
                combo_loss *= 1.0 / dataset_sizes[phase] * args.batch_size

                print(f"{phase} loss: {combo_loss}")
            else: # phase == 'val':
                disparity = loss_disparity_acc / dataset_sizes[phase] * args.batch_size * 1024

                print(f"{phase} disparity error: {disparity}")

                # store at the end of a epoch
                if not math.isnan(loss_disparity_acc) and not math.isinf(loss_disparity_acc):
                    if isinstance(model, nn.DataParallel):
                        module = model.module
                    else:
                        module = model

                    print("storing network")
                    torch.save(module.backbone, f"trained_models/{args.experiment_name}_backbone_chk.pt")
                    torch.save(module.regressor, f"trained_models/{args.experiment_name}_regressor_chk.pt")

                    if disparity < min_test_disparity:
                        print("storing network")
                        min_test_disparity = disparity
                        torch.save(module.backbone,
                                   f"trained_models/{args.experiment_name}_backbone.pt")  # maybe use type(x).__name__()
                        torch.save(module.regressor, f"trained_models/{args.experiment_name}_regressor.pt")

    writer.close()


if __name__ == '__main__':
    train()
