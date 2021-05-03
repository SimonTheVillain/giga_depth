import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast
from dataset.dataset_rendered_2 import DatasetRendered2, GetDataset
from model.regressor_2Stage import Regressor2Stage
from model.regressor_1Stage import Regressor1Stage
from model.regressor_1branch import Regressor1Branch
from model.regressor_branchless import RegressorBranchless
from model.backbone_6_64 import Backbone6_64
from experiments.lines.model_lines_CR8_n import *
from model.backbone import *
from model.backboneSliced import *
from model.regressor import Regressor, Regressor2, Reg_3stage
from torch.utils.data import DataLoader
import math
import argparse
import yaml
from params import parse_args

from torch.utils.tensorboard import SummaryWriter


class CompositeModel(nn.Module):
    def __init__(self, backbone, regressor, half_precision=False):
        super(CompositeModel, self).__init__()
        self.half_precision = half_precision
        self.backbone = backbone
        self.regressor = regressor

        # TODO: remove this debug(or at least make it so it can run with other than 64 channels
        # another TODO: set affine parameters to false!
        # self.bn = nn.BatchNorm2d(64, affine=False)

    def forward(self, x, x_gt=None):

        if x_gt != None:
            if self.half_precision:
                with autocast():
                    x, debugs = self.backbone(x, True)
                x = x.type(torch.float32)
            else:
                x, debugs = self.backbone(x, True)
            results = self.regressor(x, x_gt)
            # todo: batch norm the whole backbone and merge two dicts:
            # https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python-taking-union-o
            # z = {**x, **y}
            for key, val in debugs.items():
                results[-1][key] = val
            return results
        else:

            if self.half_precision:
                with autocast():
                    x = self.backbone(x)
                x = x.type(torch.float32)
            else:
                x = self.backbone(x)
            return self.regressor(x, x_gt)


class MaskLoss(nn.Module):
    def __init__(self, type="mask"):
        super(MaskLoss, self).__init__()
        self.type = type
        if type=="mask":
            self.loss = torch.nn.BCEWithLogitsLoss()
        else:
            print("loss not implemented yet")

    def forward(self, sigma, x, x_gt, mask_gt):
        if self.type == "mask":
            return self.loss(sigma, mask_gt)


def sigma_loss(sigma, x, x_gt, mask_gt, mode):  # sigma_sq is also called "variance"
    if mode == "mask_direct":
        return torch.abs(sigma - mask_gt).mean()
    if mode == "mask":
        # in the mask mode every pixel whose estimate is off by fewer than 0.5 pixel is valid
        mask_gt_generated = (torch.abs(x-x_gt) < (0.5 / 1024.0)).type(torch.float32)
        mask_gt_generated[mask_gt == 0.0] = 0
        return torch.abs(sigma - mask_gt).mean()# here we have it! proper mask

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
    #ideally this woudl be 0.001 so we are far away from the preferred precision of 0.1 pixel (0.1*0.1 = 0.01)
    eps = 0.001  # sqrt(0.00001) = 0.0003 ... that means we actually have approx 0.3 pixel of basic offset for sigma
    #print(f"max disparity error {torch.max(torch.abs(x-x_gt))}")
    term1 = torch.div(torch.square(x - x_gt).clamp(0, 1) * (1024.0*1024.0), sigma * sigma + eps)
    term2 = torch.log(sigma * sigma + eps)
    sigma_sq = sigma * sigma
    # term 3 is to stay positive(after all it is sigma^2)
    term3 = 0  # F.relu(-sigma)
    loss = term1 + term2 + term3
    if torch.any(torch.isnan(term1)):
        print("term1: found nan")
    if torch.any(torch.isinf(term1)):
        print("term1: found inf")

    if torch.any(torch.isnan(term2)):
        print("term2: found nan")
    if torch.any(torch.isinf(term2)):
        print("term2: found inf")
    return torch.mean(loss)  # torch.tensor(0)#


def train():
    args, config = parse_args() # todo: rename to params

    #TODO: integrate these new parameters:
    apply_mask_reg_loss = True
    dataset_format = 3

    mask_loss = MaskLoss(config["training"]["sigma_mode"])

    outlier_thresholds = list(set.union(set(args.outlier_thresholds), set(args.relative_outlier_thresholds)))


    main_device = f"cuda:{args.gpu_list[0]}"  # torch.cuda.device(args.gpu_list[0])
    # experiment_name = "cr8_2021_256_wide_reg_alpha10"
    # args.experiment_name = "slice_bb64_16_14_12c123_nobbeach_168sc_32_reg_lr01_alpha10_1nn"

    writer = SummaryWriter(f"tensorboard/{args.experiment_name}")

    shuffle = True
    slice = True

    if config["regressor"]["load_file"] != "":
        regressor = torch.load(config["regressor"]["load_file"])
        regressor.eval()
    else:
        # https://stackoverflow.com/questions/334655/passing-a-dictionary-to-a-function-as-keyword-parameters
        regressor = Reg_3stage(ch_in=config["regressor"]["ch_in"],
                               height=config["regressor"]["lines"],  # 64,#448,
                               ch_latent=config["regressor"]["bb"],
                               # [128, 128, 128],#todo: make this of variable length
                               superclasses=config["regressor"]["superclasses"],
                               ch_latent_r=config["regressor"]["ch_reg"],
                               # 64/64 # in the current implementation there is only one stage between input
                               ch_latent_msk=config["regressor"]["msk"],
                               classes=config["regressor"]["classes"],
                               pad=config["regressor"]["padding"],
                               ch_latent_c=config["regressor"]["class_bb"],  # todo: make these of variable length
                               regress_neighbours=config["regressor"]["regress_neighbours"],
                               reg_line_div=config["regressor"]["reg_line_div"],
                               c3_line_div=config["regressor"]["c3_line_div"],
                               close_far_separation=config["regressor"]["close_far_separation"],
                               sigma_mode=config["regressor"]["sigma_mode"])

    if config["backbone"]["load_file"] != "":
        backbone = torch.load(config["backbone"]["load_file"])
        backbone.eval()
        # fix parameters in the backbone (or maybe not!)
        # for param in backbone.parameters():
        #    param.requires_grad = False
    else:
        #todo: put that selection into a separate file
        if args.is_npy:
            backbone = CR8_bb_short(channels=config["backbone"]["channels"],
                                    channels_sub=config["backbone"]["channels2"])
        else:
            if config["backbone"]["name"] == "BackboneNoSlice3":
                print("BackboneNoSlice3")
                backbone = BackboneNoSlice3(height=config["dataset"]["slice_in"]["height"],
                                            channels=config["backbone"]["channels"],
                                            channels_sub=config["backbone"]["channels2"],
                                            use_bn=True, lcn=args.LCN)
            if config["backbone"]["name"] == "BackboneU1":
                print("BACKBONEU1")
                backbone = BackboneU1()
            if config["backbone"]["name"] == "BackboneU2":
                print("BACKBONEU2")
                backbone = BackboneU2()
            if config["backbone"]["name"] == "BackboneU3":
                print("BACKBONEU3")
                backbone = BackboneU3()

            if config["backbone"]["name"] == "BackboneU4":
                print("BACKBONEU4")
                backbone = BackboneU4()

            if config["backbone"]["name"] == "BackboneU5":
                print("BACKBONEU5")
                backbone = BackboneU5(norm=config["backbone"]["norm"], lcn=args.LCN)

            if config["backbone"]["name"] == "BackboneU5Sliced":
                print("BACKBONEU5Sliced")
                backbone = BackboneU5Sliced(slices=config["backbone"]["slices"], lcn=args.LCN)

    model = CompositeModel(backbone, regressor, args.half_precision)

    if len(args.gpu_list) > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=args.gpu_list)

    model.to(args.gpu_list[0])

    # for param_tensor in net.state_dict():
    #    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    else:
        if args.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(),
                                   lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
        else:
            print("Optimizer argument must either be sgd or adam!")
            return

    if "key_epochs" in config["training"]:
        key_steps = config["training"]["key_epochs"]
        lr_scales = config["training"]["lr_scales"]
    else:
        key_steps = []
        lr_scales = [1.0]
    key_steps.insert(0, -1)

    #assert
    if isinstance(args.alpha_reg, list):
        alpha_regs = args.alpha_reg
    else:
        alpha_regs = [args.alpha_reg] * len(key_steps)

    if isinstance(args.alpha_sigma, list):
        alpha_sigmas = args.alpha_sigma
    else:
        alpha_sigmas = [args.alpha_sigma] * len(key_steps)

    if isinstance(args.edge_weight, list):
        edge_weights = args.edge_weight
    else:
        edge_weights = [args.edge_weight] * len(key_steps)


    def find_index(epoch, key_steps):
        for i in range(0, len(key_steps)):
            if epoch < key_steps[i]:
                return i - 1
        return len(key_steps) - 1
    def lr_lambda(ep): return lr_scales[find_index(ep, key_steps)]
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    if args.half_precision:
        scaler = GradScaler()
    # print(f"weight_decay (DEBUG): {args.weight_decay}")
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # the whole unity rendered dataset

    # the filtered dataset
    # scale = 1 # set to 2 when using numpy
    # datasets = {
    #    'train': DatasetRendered2(args.path, 0*scale, 20000*scale, tgt_res=tgt_res, is_npy=args.is_npy),
    #    'val': DatasetRendered2(args.path, 20000*scale, 20500*scale, tgt_res=tgt_res, is_npy=args.is_npy)
    # }
    datasets = GetDataset(args.path, is_npy=args.is_npy, tgt_res=config["dataset"]["tgt_res"],
                          version=dataset_format)

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size,
                                                  shuffle=shuffle, num_workers=args.num_workers)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    min_test_disparity = 100000.0
    step = -1
    for epoch in range(1, args.epochs):
        # TODO: setup code like so: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        print(f'Epoch {epoch}/{args.epochs - 1}')
        print('-' * 10)
        alpha_reg = alpha_regs[find_index(epoch-1, key_steps)]
        alpha_sigma = float(alpha_sigmas[find_index(epoch-1, key_steps)])
        edge_weight = float(edge_weights[find_index(epoch-1, key_steps)])

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        #todo: Rename alpha_sigma according to the used training.sigma_mode
        print(f"alpha_reg {alpha_reg}, alpha_sigma {alpha_sigma}, learning_rate {get_lr(optimizer)}")

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

            mask_weight_acc = 0.0
            mask_weight_acc_sub = 0.0


            outlier_acc = [0.0] * len(outlier_thresholds)
            outlier_acc_sub = [0.0] * len(outlier_thresholds)

            model.zero_grad()
            for i_batch, sampled_batch in enumerate(dataloaders[phase]):
                step = step + 1
                ir, x_gt, mask_gt, edge_mask = sampled_batch
                ir = ir.to(main_device)
                mask_gt = mask_gt.to(main_device)
                x_gt = x_gt.to(main_device)
                edge_mask = (edge_mask * edge_weight + 1).to(main_device)

                # todo: instead of a query the slice variable should be set accordingly further up!
                if not args.is_npy and slice:
                    slice_in = (config["dataset"]["slice_in"]["start"], config["dataset"]["slice_in"]["height"])
                    slice_gt = (config["dataset"]["slice_out"]["start"], config["dataset"]["slice_out"]["height"])
                    ir = ir[:, :, slice_in[0]:slice_in[0] + slice_in[1], :]
                    x_gt = x_gt[:, :, slice_gt[0]:slice_gt[0] + slice_gt[1], :]
                    mask_gt = mask_gt[:, :, slice_gt[0]:slice_gt[0] + slice_gt[1], :]
                    edge_mask = edge_mask[:, :, slice_gt[0]:slice_gt[0] + slice_gt[1], :]

                if phase == 'train':
                    torch.autograd.set_detect_anomaly(True)
                    x, sigma, class_losses, x_real, debug = model(ir, x_gt)
                    x_real = x_real.detach()

                    # log weights and activations of different layers
                    if False:
                        for key, value in debug.items():
                            writer.add_scalar(f'debug/{key}', value.item(), step)

                    optimizer.zero_grad()
                    #mask_mean = mask_gt.mean().item() + 0.0001
                    #masked_reg = torch.mean(torch.abs(x - x_gt) * mask_gt) * (1.0 / mask_mean)

                    if apply_mask_reg_loss:
                        loss = torch.mean(torch.abs(x - x_gt) * mask_gt * edge_mask)
                        loss_reg = torch.mean(torch.abs(x - x_gt) * mask_gt)
                        mask_gt_weight = mask_gt.mean().item()
                        mask_weight_acc_sub += mask_gt_weight
                        mask_weight_acc += mask_gt_weight
                    else:
                        loss = torch.mean(torch.abs(x - x_gt) * edge_mask)
                        loss_reg = torch.mean(torch.abs(x - x_gt))
                        mask_weight_acc_sub += 1.0
                        mask_weight_acc += 1.0

                    loss_reg_acc += loss_reg.item()
                    loss_reg_acc_sub += loss_reg.item()
                    loss = loss * alpha_reg

                    if alpha_sigma != 0.0:
                        loss_sigma = mask_loss(sigma, x_real, x_gt, mask_gt)
                        loss_sigma_acc += loss_sigma.item()
                        loss_sigma_acc_sub += loss_sigma.item()
                        loss += loss_sigma * alpha_sigma

                    if len(loss_class_acc) == 0:
                        loss_class_acc = [0] * len(class_losses)
                        loss_class_acc_sub = [0] * len(class_losses)


                    for i, class_loss in enumerate(class_losses):
                        if apply_mask_reg_loss:
                            loss += torch.mean(class_loss * mask_gt * edge_mask)
                            mean_class_loss = torch.mean(class_loss * mask_gt * edge_mask).item()
                            loss_class_acc[i] += mean_class_loss
                            loss_class_acc_sub[i] += mean_class_loss
                        else:
                            loss += torch.mean(class_loss * edge_mask)
                            mean_class_loss = torch.mean(class_loss * edge_mask).item()
                            loss_class_acc[i] += mean_class_loss
                            loss_class_acc_sub[i] += mean_class_loss

                    #print(f"loss {loss.item()}")
                    loss = loss / float(args.accumulation_steps)
                    if args.half_precision:
                        scaler.scale(loss).backward()
                        if (i_batch + 1) % args.accumulation_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            model.zero_grad()
                    else:
                        loss.backward()
                        if (i_batch + 1) % args.accumulation_steps == 0:
                            optimizer.step()
                            model.zero_grad()
                else: # val
                    with torch.no_grad():
                        x_real, sigma = model(ir)
                        # loss = torch.mean(torch.abs(x - x_gt)) #mask_gt
                        # loss_disparity_acc += loss.item()
                        # loss_disparity_acc_sub += loss.item()
                        if apply_mask_reg_loss:
                            mask_weight_acc_sub += mask_gt.mean().item()
                            mask_weight_acc += mask_gt.mean().item()
                        else:
                            mask_weight_acc_sub += 1.0
                            mask_weight_acc += 1.0

                        if alpha_sigma != 0.0:
                            loss_sigma = mask_loss(sigma, x_real, x_gt, mask_gt)
                            loss_sigma_acc += loss_sigma.item()
                            loss_sigma_acc_sub += loss_sigma.item()

                delta = torch.abs(x_real - x_gt)
                if apply_mask_reg_loss:
                    delta_disp = torch.mean(delta * mask_gt).item()
                    loss_disparity_acc += delta_disp
                    loss_disparity_acc_sub += delta_disp
                else:
                    delta_disp = torch.mean(delta).item()
                    loss_disparity_acc += delta_disp
                    loss_disparity_acc_sub += delta_disp

                focal_cam = 1115.0 #approx.
                focal_projector = 850 # chosen in dataset etc
                delta_scaled = delta * 1024.0 * focal_cam / focal_projector
                for i, th in enumerate(outlier_thresholds):
                    item = (delta_scaled > th).type(torch.float32).mean().item()
                    outlier_acc[i] += item
                    outlier_acc_sub[i] += item

                # print progress every 99 steps!
                if i_batch % 100 == 99:
                    writer.add_scalar(f'{phase}_subepoch/disparity_error',
                                      loss_disparity_acc_sub / 100.0 * 1024, step)
                    if alpha_sigma > 1e-6: # only plot when the sigma loss has meaningful weight
                        # todo: Rename sigma_loss according to the used training.sigma_mode
                        writer.add_scalar(f'{phase}_subepoch/sigma_loss',
                                          loss_sigma_acc_sub / 100.0, step)

                    if phase == 'train':
                        writer.add_scalar(f'{phase}_subepoch/regression_error', loss_reg_acc_sub / mask_weight_acc_sub * 1024, step)

                        combo_loss = loss_reg_acc_sub * alpha_reg / mask_weight_acc_sub
                        combo_loss += sum(loss_class_acc_sub) / mask_weight_acc_sub
                        combo_loss += loss_sigma_acc_sub * alpha_sigma / 100
                        print("batch {} loss: {}".format(i_batch, combo_loss))
                    else:  # val
                        print("batch {} disparity error: {}".format(i_batch,
                                                                    loss_disparity_acc_sub / mask_weight_acc_sub * 1024))

                    for i, class_loss in enumerate(loss_class_acc_sub):
                        #if i in plotable_class_losses:
                        writer.add_scalar(f'{phase}_subepoch/class_loss_{i}', class_loss / mask_weight_acc_sub, step)
                        loss_class_acc_sub[i] = 0

                    for th in args.relative_outlier_thresholds[1:]:
                        i = outlier_thresholds.index(th)
                        ind_ref = outlier_thresholds.index(args.relative_outlier_thresholds[0])
                        ref = outlier_acc_sub[ind_ref] / 100
                        tgt = outlier_acc_sub[i] / 100
                        th_ref = outlier_thresholds[ind_ref]
                        th_tgt = outlier_thresholds[i]
                        writer.add_scalar(f"{phase}_sub_outlier_ratio/o({th_tgt}|{th_ref})",
                                          (tgt - ref) / (1.0 - ref), step)

                    for th in args.outlier_thresholds:
                        i = outlier_thresholds.index(th)
                        writer.add_scalar(f"{phase}_sub_outlier_ratio/o({th})", outlier_acc_sub[i] / 100, step)
                    outlier_acc_sub = [0.0] * len(outlier_thresholds)
                    writer.add_scalar(f"{phase}_sub_outlier_ratio/valid", mask_weight_acc_sub / 100, step)


                    loss_disparity_acc_sub = 0
                    loss_sigma_acc_sub = 0
                    loss_reg_acc_sub = 0
                    mask_weight_acc_sub = 0

            #write progress every epoch!
            writer.add_scalar(f"{phase}/disparity",
                              loss_disparity_acc / dataset_sizes[phase] * args.batch_size * 1024, step)
            if alpha_sigma > 1e-6:
                # todo: Rename sigma_loss according to the used training.sigma_mode
                writer.add_scalar(f"{phase}/sigma(loss)",
                                  loss_sigma_acc / dataset_sizes[phase] * args.batch_size, step)

            for th in args.relative_outlier_thresholds[1:]:
                i = outlier_thresholds.index(th)
                ind_ref = outlier_thresholds.index(args.relative_outlier_thresholds[0])
                ref = outlier_acc[ind_ref] / dataset_sizes[phase] * args.batch_size
                tgt = outlier_acc[i] / dataset_sizes[phase] * args.batch_size
                th_ref = outlier_thresholds[ind_ref]
                th_tgt = outlier_thresholds[i]
                writer.add_scalar(f"{phase}_outlier_ratio/o({th_tgt}|{th_ref})",
                                  (tgt - ref) / (1.0 - ref), step)

            for th in args.outlier_thresholds:
                i = outlier_thresholds.index(th)
                writer.add_scalar(f"{phase}_outlier_ratio/o({th})",
                                  outlier_acc[i] / dataset_sizes[phase] * args.batch_size, step)
            writer.add_scalar(f"{phase}_outlier_ratio/valid",
                              mask_weight_acc / dataset_sizes[phase] * args.batch_size, step)

            for i, class_loss in enumerate(loss_class_acc):
                # epoch_loss += class_loss / dataset_sizes[phase] * args.batch_size
                # if i in plotable_class_losses:
                writer.add_scalar(f"{phase}/class_loss{i}", class_loss / mask_weight_acc, step)
            if phase == 'train':
                writer.add_scalar(f"{phase}/regression_stage",
                                  loss_reg_acc / mask_weight_acc * 1024.0, step)
                #combo_loss = loss_reg_acc * alpha_reg + loss_sigma_acc * alpha_sigma + sum(loss_class_acc_sub)
                #combo_loss *= 1.0 / dataset_sizes[phase] * args.batch_size

                combo_loss = loss_reg_acc * alpha_reg / mask_weight_acc
                combo_loss += sum(loss_class_acc) / mask_weight_acc
                combo_loss += loss_sigma_acc * alpha_sigma / dataset_sizes[phase] * args.batch_size
                print(f"{phase} loss: {combo_loss}")
            else:  # phase == 'val':
                disparity = loss_disparity_acc / mask_weight_acc * 1024.0

                print(f"{phase} disparity error: {disparity}")

                # store at the end of a validation phase of this epoch!
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
        scheduler.step()
    writer.close()


if __name__ == '__main__':
    train()
