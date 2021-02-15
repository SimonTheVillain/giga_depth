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
from model.backbone import Backbone, BackboneSliced, BackboneSliced2, BackboneSliced3
from model.regressor import Regressor, Regressor2, Reg_3stage
from torch.utils.data import DataLoader
import math
import argparse
import yaml

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
    with open("configs/default.yaml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", dest="config", action="store",
                        help="Load the config file with parameters.",
                        default="")
    args, additional_args = parser.parse_known_args()
    if args.config != "":
        with open(args.config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)
            #todo: recursively merge both config structures!!!!!!!
    # parser.add_argument("-V", "--version", help="show program version", action="store_true")
    parser.add_argument("-d", "--dataset_path", dest="path", action="store",
                        help="Path to the dataset.",
                        default=os.path.expanduser(config["dataset"]["path"]))
    parser.add_argument("-npy", "--npy_dataset", dest="is_npy", action="store_const",
                        help="Loads data directly form numpy files",
                        default=bool(config["dataset"]["is_npy"]), const=True)
    parser.add_argument("-b", "--batch_size", dest="batch_size", action="store",
                        help="The batch size during training",
                        type=int,
                        default=int(config["training"]["batch_size"]))
    parser.add_argument("-e", "--epochs", dest="epochs", action="store",
                        help="The number of epochs before quitting training.",
                        default=config["training"]["epochs"])
    parser.add_argument("-n", "--experiment_name", dest="experiment_name", action="store",
                        help="The name of this training for tensorboard and checkpoints.",
                        default=config["training"]["name"])
    parser.add_argument("-w", "--num_workers", dest="num_workers", action="store",
                        help="The number of threads working on loading and preprocessing data.",
                        type=int,
                        default=config["dataset"]["workers"])
    parser.add_argument("-g", "--gpu_list", dest="gpu_list", action="store",
                        nargs="+", type=int,
                        default=list(range(0, torch.cuda.device_count())))
    parser.add_argument("-r", "--learning_rate", dest="learning_rate", action="store",
                        help="Learning rate for gradient descent algorithm.",
                        type=float,
                        default=config["training"]["learning_rate"])
    parser.add_argument("-m", "--momentum", dest="momentum", action="store",
                        help="Momentum for gradient descent algorithm.",
                        type=float,
                        default=config["training"]["momentum"])
    parser.add_argument("-a", "--alpha_reg", dest="alpha_reg", action="store",
                        help="The factor with which the regression error is incorporated into the loss.",
                        type=float,
                        default=config["training"]["alpha_reg"])
    parser.add_argument("-as", "--alpha_sigma", dest="alpha_sigma", action="store",
                        help="The factor with which mask error is incorporated into the loss.",
                        type=float,
                        default=config["training"]["alpha_sigma"])
    args = parser.parse_args(additional_args)
    main_device = f"cuda:{args.gpu_list[0]}"# torch.cuda.device(args.gpu_list[0])
    #experiment_name = "cr8_2021_256_wide_reg_alpha10"
    #args.experiment_name = "slice_bb64_16_14_12c123_nobbeach_168sc_32_reg_lr01_alpha10_1nn"

    writer = SummaryWriter(f"tensorboard/{args.experiment_name}")

    # slit loading and storing of models for Backbone and Regressor
    #load_regressor = "trained_models/line_bb64_16_14_12c123_32_32_32_64bb_42sc_64_128_reg_lr01_alpha50_1nn_regressor_chk.pt"
    #load_backbone = "trained_models/line_bb64_16_14_12c123_32_32_32_64bb_42sc_64_128_reg_lr01_alpha50_1nn_backbone_chk.pt"

    # not loading any pretrained part of any model whatsoever
    #load_regressor = ""
    #load_backbone = ""

    #num_epochs = 5000
    # todo: either do only 100 lines at a time, or go for
    #tgt_res = (1216, 896)
    #alpha = 1.0 * (1.0 / 4.0) * 1.0  # usually this is 0.1
    #alpha = 10.0 #todo: back to 10 for quicker convergence!?
    #alpha_sigma = 0#1e-10  # how much weight do we give correct confidence measures
    #learning_rate = 0.2 # learning rate of 0.2 was sufficient for many applications
    #learning_rate = 0.01  # 0.02 for the branchless regressor (smaller since we feel it's not entirely stable)
    #momentum = 0.90
    shuffle = True
    slice = True

    if config["regressor"]["load_file"] != "":
        regressor = torch.load(config["regressor"]["load_file"])
        regressor.eval()
    else:
        #https://stackoverflow.com/questions/334655/passing-a-dictionary-to-a-function-as-keyword-parameters
        regressor = Reg_3stage(ch_in=config["regressor"]["ch_in"],
                               height=config["regressor"]["lines"],#64,#448,
                               ch_latent=config["regressor"]["bb"],#[128, 128, 128],#todo: make this of variable length
                               superclasses=config["regressor"]["superclasses"],
                               ch_latent_r=config["regressor"]["ch_reg"],# 64/64 # in the current implementation there is only one stage between input
                               ch_latent_msk=config["regressor"]["msk"],
                               classes=config["regressor"]["classes"],
                               pad=config["regressor"]["padding"],
                               ch_latent_c=config["regressor"]["class_bb"],#todo: make these of variable length
                               regress_neighbours=config["regressor"]["regress_neighbours"])

    if config["backbone"]["load_file"] != "":
        backbone = torch.load(config["backbone"]["load_file"])
        backbone.eval()
        # fix parameters in the backbone (or maybe not!)
        # for param in backbone.parameters():
        #    param.requires_grad = False
    else:
        if args.is_npy:
            backbone = CR8_bb_short(channels=config["backbone"]["channels"],
                                    channels_sub=config["backbone"]["channels2"])
        else:
            backbone = BackboneSliced3(slices=1, height=config["dataset"]["slice_in"]["height"],
                                       channels=config["backbone"]["channels"],
                                       channels_sub=config["backbone"]["channels2"])

    model = CompositeModel(backbone, regressor)

    if len(args.gpu_list) > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=args.gpu_list)

    model.to(args.gpu_list[0])

    # for param_tensor in net.state_dict():
    #    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # the whole unity rendered dataset

    # the filtered dataset
    #scale = 1 # set to 2 when using numpy
    #datasets = {
    #    'train': DatasetRendered2(args.path, 0*scale, 20000*scale, tgt_res=tgt_res, is_npy=args.is_npy),
    #    'val': DatasetRendered2(args.path, 20000*scale, 20500*scale, tgt_res=tgt_res, is_npy=args.is_npy)
    #}
    datasets = GetDataset(args.path, is_npy=args.is_npy, tgt_res=config["dataset"]["tgt_res"])

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size,
                                                  shuffle=shuffle, num_workers=args.num_workers)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    min_test_disparity = 100000.0
    step = 0
    for epoch in range(1, args.epochs):
        # TODO: setup code like so: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
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
                    slice_in = (config["dataset"]["slice_in"]["start"], config["dataset"]["slice_in"]["height"])
                    slice_gt = (config["dataset"]["slice_out"]["start"], config["dataset"]["slice_out"]["height"])
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

                    loss = loss * args.alpha_reg

                    if args.alpha_sigma != 0.0:
                        loss_sigma = sigma_loss(sigma, x_real, x_gt)
                        loss_sigma_acc += loss_sigma.item()
                        loss_sigma_acc_sub += loss_sigma.item()
                        loss += loss_sigma * args.alpha_sigma

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

                        if args.alpha_sigma != 0.0:
                            loss_sigma = sigma_loss(sigma, x, x_gt)
                            loss_sigma_acc += loss_sigma.item()
                            loss_sigma_acc_sub += loss_sigma.item()

                delta_disp = torch.mean(torch.abs(x_real - x_gt)).item()
                loss_disparity_acc += delta_disp
                loss_disparity_acc_sub += delta_disp


                if i_batch % 100 == 99:
                    writer.add_scalar(f'{phase}_subepoch/disparity_error',
                                      loss_disparity_acc_sub / 100.0 * 1024, step)
                    if args.alpha_sigma != 0.0:
                        writer.add_scalar(f'{phase}_subepoch/sigma_loss',
                                          loss_sigma_acc_sub / 100.0, step)

                    if phase == 'train':
                        writer.add_scalar(f'{phase}_subepoch/regression_error', loss_reg_acc_sub / 100.0 * 1024, step)


                        combo_loss = loss_reg_acc_sub * args.alpha_reg + \
                                     loss_sigma_acc_sub * args.alpha_sigma + sum(loss_class_acc_sub)
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
            if args.alpha_sigma != 0.0:
                writer.add_scalar(f"{phase}/sigma(loss)",
                                  loss_sigma_acc / dataset_sizes[phase] * args.batch_size, step)
            for i, class_loss in enumerate(loss_class_acc):
                #epoch_loss += class_loss / dataset_sizes[phase] * args.batch_size
                writer.add_scalar(f"{phase}/class_loss{i}", class_loss / dataset_sizes[phase] * args.batch_size, step)
            if phase == 'train':
                writer.add_scalar(f"{phase}/regression_stage",
                                  loss_reg_acc / dataset_sizes[phase] * args.batch_size * 1024, step)
                combo_loss = loss_reg_acc * args.alpha_reg + loss_sigma_acc * args.alpha_sigma + sum(loss_class_acc_sub)
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
