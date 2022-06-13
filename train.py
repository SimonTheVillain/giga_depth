import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
from dataset.datasets import GetDataset
from model.composite_model import CompositeModel, GetModel
from model.backbone import *
from model.backboneSliced import *
from model.regressor import Reg_3stage
from torch.utils.data import DataLoader
import math
from params import parse_args

from torch.utils.tensorboard import SummaryWriter


def train():
    args, config = parse_args()  # todo: rename to params

    # TODO: integrate these new parameters:
    apply_mask_reg_loss = True

    outlier_thresholds = list(set.union(set(args.outlier_thresholds), set(args.relative_outlier_thresholds)))

    main_device = f"cuda:{args.gpu_list[0]}"  # torch.cuda.device(args.gpu_list[0])
    # experiment_name = "cr8_2021_256_wide_reg_alpha10"
    # args.experiment_name = "slice_bb64_16_14_12c123_nobbeach_168sc_32_reg_lr01_alpha10_1nn"

    writer = SummaryWriter(f"tensorboard/{args.experiment_name}")

    shuffle = True
    slice = True

    model = GetModel(args, config)
    if False:
        model = torch.load(f"trained_models/dis_def_lcn_j2_c960_v2_chk.pt")
        model.eval()
        model.cuda()

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

    # assert
    if isinstance(args.alpha_reg, list):
        alpha_regs = args.alpha_reg
    else:
        alpha_regs = [args.alpha_reg] * len(key_steps)

    if isinstance(args.edge_weight, list):
        edge_weights = args.edge_weight
    else:
        edge_weights = [args.edge_weight] * len(key_steps)

    def find_index(epoch, key_steps):
        for i in range(0, len(key_steps)):
            if epoch < key_steps[i]:
                return i - 1
        return len(key_steps) - 1

    def lr_lambda(ep):
        return lr_scales[find_index(ep, key_steps)]

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    scheduler.step(epoch=args.start_epoch)
    #for i in range(args.start_epoch):
    #    scheduler.step()

    if args.half_precision:
        scaler = GradScaler()

    datasets, _, _, _, _, tgt_res = \
        GetDataset(args.path,
                   vertical_jitter=config["dataset"]["vertical_jitter"],
                   tgt_res=config["dataset"]["tgt_res"],
                   version=args.dataset_type)
    width = tgt_res[0]

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size,
                                                  shuffle=shuffle, num_workers=args.num_workers)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    min_test_disparity = 100000.0
    step = -1
    for epoch in range(args.start_epoch, args.epochs+1):
        # TODO: setup code like so: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        print(f'Epoch {epoch}/{args.epochs}')
        print('-' * 10)
        alpha_reg = alpha_regs[find_index(epoch - 1, key_steps)]
        edge_weight = float(edge_weights[find_index(epoch - 1, key_steps)])

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        # todo: Rename alpha_sigma according to the used training.sigma_mode
        print(f"alpha_reg {alpha_reg}, learning_rate {get_lr(optimizer)}")

        for phase in ['train', 'val']:
            print(phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # TODO: accumulate classification losses for each classification stage!!!
            loss_disparity_acc = 0
            loss_reg_acc = 0
            loss_class_acc = []
            loss_disparity_acc_sub = 0
            loss_reg_acc_sub = 0
            loss_class_acc_sub = []

            mask_weight_acc = 0.0
            mask_weight_acc_sub = 0.0

            outlier_acc = [0.0] * len(outlier_thresholds)
            outlier_acc_sub = [0.0] * len(outlier_thresholds)

            model.zero_grad()
            for i_batch, sampled_batch in enumerate(dataloaders[phase]):

                step = step + 1
                if len(sampled_batch) == 4:
                    ir, x_gt, mask_gt, edge_mask = sampled_batch
                    rendered_msk = torch.ones((ir.shape[0], 1, 1, 1), dtype=torch.float32)
                else:
                    ir, x_gt, mask_gt, edge_mask, rendered_msk = sampled_batch

                ir = ir.to(main_device)
                mask_gt = mask_gt.to(main_device)
                x_gt = x_gt.to(main_device)
                rendered_msk = rendered_msk.to(main_device)

                # scale the normalized x_pos so it extends a bit to the left and right of the projector
                #x_gt = x_gt * (1.0 - 2.0 * args.pad_proj) + args.pad_proj
                if False:
                    cv2.imshow("x_gt", x_gt[0,0,:,:].detach().cpu().numpy())
                    cv2.imshow("mask_gt", mask_gt[0,0,:,:].detach().cpu().numpy())
                    cv2.imshow("ir", ir[0,0,:,:].detach().cpu().numpy())
                    cv2.imshow("edge_mask", edge_mask[0,0,:,:].detach().cpu().numpy())
                    cv2.waitKey()
                mask_gt[torch.logical_or(x_gt < 0.0, x_gt > 1.0)] = 0

                edge_mask = (edge_mask * edge_weight + 1).to(main_device)

                # todo: instead of a query the slice variable should be set accordingly further up!
                if slice:
                    slice_in = (config["dataset"]["slice_in"]["start"], config["dataset"]["slice_in"]["height"])
                    slice_gt = (config["dataset"]["slice_out"]["start"], config["dataset"]["slice_out"]["height"])
                    ir = ir[:, :, slice_in[0]:slice_in[0] + slice_in[1], :]
                    x_gt = x_gt[:, :, slice_gt[0]:slice_gt[0] + slice_gt[1], :]
                    mask_gt = mask_gt[:, :, slice_gt[0]:slice_gt[0] + slice_gt[1], :]
                    edge_mask = edge_mask[:, :, slice_gt[0]:slice_gt[0] + slice_gt[1], :]

                if phase == 'train':
                    torch.autograd.set_detect_anomaly(True)
                    x, class_losses, x_real, debug = model(ir, x_gt)
                    x_real = x_real.detach()
                    #delta = torch.abs(x - x_gt) * (1.0 / (1.0 - 2.0 * args.pad_proj))
                    delta = torch.abs(x - x_gt)

                    if False:
                        cv2.imshow("x_gt", x_gt[0, 0, :, :].detach().cpu().numpy())
                        cv2.imshow("x_out", x_real[0, 0, :, :].detach().cpu().numpy())
                        cv2.imshow("mask_gt", mask_gt[0, 0, :, :].detach().cpu().numpy())
                        cv2.imshow("ir", ir[0, 0, :, :].detach().cpu().numpy())
                        cv2.waitKey()

                    # log weights and activations of different layers
                    if False:
                        for key, value in debug.items():
                            writer.add_scalar(f'debug/{key}', value.item(), step)

                    optimizer.zero_grad()
                    # mask_mean = mask_gt.mean().item() + 0.0001
                    # masked_reg = torch.mean(torch.abs(x - x_gt) * mask_gt) * (1.0 / mask_mean)

                    if apply_mask_reg_loss:
                        loss = torch.mean(delta * mask_gt * rendered_msk * edge_mask)
                        loss_reg = torch.mean(delta * mask_gt)
                        mask_gt_weight = mask_gt.mean().item()
                        mask_weight_acc_sub += mask_gt_weight
                        mask_weight_acc += mask_gt_weight
                    else:
                        loss = torch.mean(delta * edge_mask)
                        loss_reg = torch.mean(delta)
                        mask_weight_acc_sub += 1.0
                        mask_weight_acc += 1.0

                    loss_reg_acc += loss_reg.item()
                    loss_reg_acc_sub += loss_reg.item()
                    loss = loss * alpha_reg

                    if len(loss_class_acc) == 0:
                        loss_class_acc = [0] * len(class_losses)
                        loss_class_acc_sub = [0] * len(class_losses)

                    reg_loss_msk = [torch.ones_like(rendered_msk),
                                    torch.ones_like(rendered_msk),
                                    #torch.ones_like(rendered_msk)]#
                                    rendered_msk]
                    for i, class_loss in enumerate(class_losses):
                        if apply_mask_reg_loss:
                            loss += torch.mean(class_loss * reg_loss_msk[i] * mask_gt * edge_mask)
                            mean_class_loss = torch.mean(class_loss * mask_gt * edge_mask).item()
                            loss_class_acc[i] += mean_class_loss
                            loss_class_acc_sub[i] += mean_class_loss
                        else:
                            loss += torch.mean(class_loss * reg_loss_msk[i] * edge_mask)
                            mean_class_loss = torch.mean(class_loss * edge_mask).item()
                            loss_class_acc[i] += mean_class_loss
                            loss_class_acc_sub[i] += mean_class_loss

                    # print(f"loss {loss.item()}")
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
                else:  # val
                    with torch.no_grad():
                        x_real = model(ir)
                        # loss = torch.mean(torch.abs(x - x_gt)) #mask_gt
                        # loss_disparity_acc += loss.item()
                        # loss_disparity_acc_sub += loss.item()
                        if apply_mask_reg_loss:
                            mask_weight_acc_sub += mask_gt.mean().item()
                            mask_weight_acc += mask_gt.mean().item()
                        else:
                            mask_weight_acc_sub += 1.0
                            mask_weight_acc += 1.0



                #delta = torch.abs(x_real - x_gt) * (1.0 / (1.0 - 2.0 * args.pad_proj))
                delta = torch.abs(x_real - x_gt)
                if apply_mask_reg_loss:
                    delta_disp = torch.mean(delta * mask_gt).item()
                    loss_disparity_acc += delta_disp
                    loss_disparity_acc_sub += delta_disp
                else:
                    delta_disp = torch.mean(delta).item()
                    loss_disparity_acc += delta_disp
                    loss_disparity_acc_sub += delta_disp

                # focal_cam = 1115.0 #approx.
                # focal_projector = 850 # chosen in dataset etc
                #delta_scaled = delta * width * (1.0 / (1.0 - 2.0 * args.pad_proj))
                delta_scaled = delta * width
                for i, th in enumerate(outlier_thresholds):
                    item = (delta_scaled > th).type(torch.float32).mean().item()
                    outlier_acc[i] += item
                    outlier_acc_sub[i] += item

                # print progress every 99 steps!
                if i_batch % 100 == 99:
                    cv2.imwrite("tmp/input.png", ir[0, 0, :, :].detach().cpu().numpy()*255)
                    cv2.imwrite("tmp/x_gt.png", x_gt[0, 0, :, :].detach().cpu().numpy()*255)
                    cv2.imwrite("tmp/x_real.png", x_real[0, 0, :, :].detach().cpu().numpy()*255)
                    if False: # TODO: remove this debug
                        cv2.imshow("x_gt", x_gt[0,0,:,:].detach().cpu().numpy())
                        cv2.imshow("tmp/x_real.png", x_real[0, 0, :, :].detach().cpu().numpy())
                        cv2.imshow("mask_gt", mask_gt[0,0,:,:].detach().cpu().numpy())
                        cv2.imshow("ir", ir[0,0,:,:].detach().cpu().numpy())
                        cv2.imshow("edge_mask", edge_mask[0,0,:,:].detach().cpu().numpy())
                        cv2.waitKey()
                    writer.add_scalar(f'{phase}_subepoch/disparity_error',
                                      loss_disparity_acc_sub / 100.0 * width, step)

                    if phase == 'train':
                        writer.add_scalar(f'{phase}_subepoch/regression_error',
                                          loss_reg_acc_sub / mask_weight_acc_sub * 1024, step)

                        combo_loss = loss_reg_acc_sub * alpha_reg / mask_weight_acc_sub
                        combo_loss += sum(loss_class_acc_sub) / mask_weight_acc_sub
                        print("batch {} loss: {}".format(i_batch, combo_loss))
                    else:  # val
                        print("batch {} disparity error: {}".format(i_batch,
                                                                    loss_disparity_acc_sub / mask_weight_acc_sub * 1024))

                    for i, class_loss in enumerate(loss_class_acc_sub):
                        # if i in plotable_class_losses:
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
                    loss_reg_acc_sub = 0
                    mask_weight_acc_sub = 0

            # write progress every epoch!
            writer.add_scalar(f"{phase}/disparity",
                              loss_disparity_acc / dataset_sizes[phase] * args.batch_size * width, step)

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
                                  loss_reg_acc / mask_weight_acc * width, step)

                combo_loss = loss_reg_acc * alpha_reg / mask_weight_acc
                combo_loss += sum(loss_class_acc) / mask_weight_acc
                print(f"{phase} loss: {combo_loss}")
            else:  # phase == 'val':
                disparity = loss_disparity_acc / mask_weight_acc * width

                print(f"{phase} disparity error: {disparity}")

                # store at the end of a validation phase of this epoch!
                if not math.isnan(loss_disparity_acc) and not math.isinf(loss_disparity_acc):
                    if isinstance(model, nn.DataParallel):
                        module = model.module
                    else:
                        module = model

                    print("storing network")
                    torch.save(module, f"trained_models/{args.experiment_name}_chk.pt")

                    if disparity < min_test_disparity:
                        print("storing network")
                        min_test_disparity = disparity
                        torch.save(module, f"trained_models/{args.experiment_name}.pt")

        scheduler.step()
    writer.close()


if __name__ == '__main__':
    train()
