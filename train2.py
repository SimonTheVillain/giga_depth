import torch
from params import parse_args
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
from model.losses import *
from model.composite_model import GetModel
from torch.utils.tensorboard import SummaryWriter
from dataset.datasets import GetDataset
import math


def find_index(step, key_steps):
    for i in range(0, len(key_steps)):
        if step < key_steps[i]:
            return i - 1
    return len(key_steps) - 1


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train():
    width = 1216
    args, config = parse_args()

    model = GetModel(args, config)
    model.cuda()

    stereo_loss = StereoLoss(1, 1216//2)
    stereo_loss.cuda()
    mask_loss = MaskLoss(config["training"]["sigma_mode"])
    mask_loss.cuda()

    assert args.optimizer == "sgd", "No other optimizers than sgd are tested/allowed!"
    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    writer = SummaryWriter(f"tensorboard/{args.experiment_name}")



    if "key_epochs" in config["training"]:
        key_steps = config["training"]["key_epochs"]
        lr_scales = config["training"]["lr_scales"]
    else:
        key_steps = []
        lr_scales = [1.0]
    key_steps.insert(0, -1)
    def lr_lambda(ep):
        return lr_scales[find_index(ep, key_steps)]

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)



    if "key_epochs" in config["training"]:
        key_steps = config["training"]["key_steps"]
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


    if isinstance(args.alpha_regs_conv, list):
        alpha_regs_conv = args.alpha_regs_conv
    else:
        alpha_regs_conv = [args.alpha_regs_conv] * len(key_steps)

    if isinstance(args.alpha_sigma, list):
        alpha_sigmas = args.alpha_sigma
    else:
        alpha_sigmas = [args.alpha_sigma] * len(key_steps)

    if isinstance(args.edge_weight, list):
        edge_weights = args.edge_weight
    else:
        edge_weights = [args.edge_weight] * len(key_steps)

    if isinstance(args.alpha_reg_photometric, list):
        alpha_reg_photometric = args.alpha_reg_photometric
    else:
        alpha_reg_photometric = [args.alpha_reg_photometric] * len(key_steps)

    if args.half_precision:
        scaler = GradScaler()

    assert args.dataset_type == "structure_core_combo", "Dataset type needs to be structure_core_combo"
    datasets, _, _, _, _, tgt_res = \
        GetDataset(args.path,
                   vertical_jitter=config["dataset"]["vertical_jitter"],
                   tgt_res=config["dataset"]["tgt_res"],
                   version=args.dataset_type)

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size,
                                                  shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    optimizer.zero_grad()
    min_test_disparity = 10000000
    step = 0
    for epoch in range(args.epochs + 1):
        print(f'Epoch {epoch}/{args.epochs}')
        print('-' * 10)

        stage_ind = find_index(epoch - 1, key_steps)
        alpha_reg = alpha_regs[stage_ind]
        alpha_reg_conv = alpha_regs_conv[stage_ind]
        alpha_photometric = float(alpha_reg_photometric[stage_ind])
        alpha_sigma = float(alpha_sigmas[stage_ind])
        edge_weight = float(edge_weights[stage_ind])

        print(f"stage {stage_ind}: learning_rate {get_lr(optimizer)}, alpha_reg {alpha_reg}, "
              f"alpha_reg_conv {alpha_reg_conv}, alpha_msk(sigma) {alpha_sigma}, alpha_photometric {alpha_photometric},"
              f"edge_weight {edge_weight}")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            delta_acc_sub = 0
            delta_acc = 0
            valid_acc_sub = 0
            valid_acc = 0
            delta_conv_acc_sub = 0
            delta_conv_acc = 0
            loss_acc_sub = 0
            loss_acc = 0
            for i_batch, sampled_batch in enumerate(dataloaders[phase]):
                for i in range(len(sampled_batch)):
                    sampled_batch[i] = sampled_batch[i].cuda()
                ir, irr, gt_x, gt_msk, edge_mask, has_gt = sampled_batch
                use_stereo = 1.0 - has_gt# use stereo consistency only where we don't have groundtruth
                step = step + 1

                edge_mask = (edge_mask * edge_weight + 1.0)

                if phase == 'train':
                    torch.autograd.set_detect_anomaly(True)
                    x, msk, class_losses, x_real, debug, x_conv, msk_conv = model(ir, gt_x)

                    if has_gt[0] == 1.0 and False:#TODO: remove this desparate debug measure
                        cv2.imshow("x1", x[0,0,:,:].detach().cpu().numpy())
                        cv2.imshow("x2", x[0,1,:,:].detach().cpu().numpy())
                        cv2.imshow("x3", x[0,2,:,:].detach().cpu().numpy())
                        cv2.imshow("x_real", x_real[0,0,:,:].detach().cpu().numpy())
                        cv2.imshow("x_gt", gt_x[0,0,:,:].detach().cpu().numpy())
                        cv2.imshow("x_conv", x_conv[0,0,:,:].detach().cpu().numpy())
                        cv2.imshow("ir", ir[0,0,:,:].detach().cpu().numpy())
                        cv2.waitKey(1)
                    #####################LOSSES####################
                    delta = torch.abs(x - gt_x)
                    delta_conv = torch.abs(x_conv - gt_x)
                    loss = torch.mean(delta * gt_msk * edge_mask * has_gt) * alpha_reg
                    loss += torch.mean(delta_conv * gt_msk * edge_mask * has_gt) * alpha_reg_conv

                    for i, class_loss in enumerate(class_losses):
                        loss += torch.mean(class_loss * gt_msk * edge_mask)

                    #TODO: put back in
                    loss_consistency = stereo_loss(ir, irr, x_conv)
                    loss += torch.mean(loss_consistency * use_stereo) * alpha_photometric

                    #todo: put back in
                    #losses for mask
                    loss_sigma = mask_loss(msk, x_real, gt_x, gt_msk)
                    loss += torch.mean(loss_sigma * has_gt) * alpha_sigma
                    loss_sigma = mask_loss(msk_conv, x_conv, gt_x, gt_msk)
                    loss += torch.mean(loss_sigma * has_gt) * alpha_sigma
                    #optimize:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    ###################logger:
                    #calculate the delta of the real estimate!
                    delta = torch.abs((x_real - gt_x) * gt_msk).detach().mean().item()
                    mean = gt_msk.mean().detach().item()
                    if mean != 0.0:
                        writer.add_scalar("mean_disp_error", delta/mean * width, step)
                    delta_acc += delta
                    delta_acc_sub += delta
                    valid_acc += mean
                    valid_acc_sub += mean
                    delta = delta_conv.detach().mean().item()
                    delta_conv_acc += delta
                    delta_conv_acc_sub += delta
                    loss_acc_sub += loss.detach().item()
                    loss_acc += loss.detach().item()
                else: # val
                    with torch.no_grad():
                        x_real, msk_conv, x_conv, msk_conv = model(ir)
                        delta = torch.abs((x_real - gt_x) * gt_msk).detach().mean().item()
                        delta_conv = torch.abs(x_conv - gt_x).detach().mean().item()
                        mean = gt_msk.mean().detach().item()

                        delta_acc += delta
                        delta_conv_acc += delta_conv
                        valid_acc += mean
                        valid_acc_sub += mean

                if i_batch % 100 == 99:
                    if phase == 'train':
                        writer.add_scalar(f'{phase}_subepoch/disparity_error',
                                          delta_acc_sub / valid_acc_sub * width, step)
                        writer.add_scalar(f'{phase}_subepoch/disparity_error(fully_conv)',
                                          delta_conv_acc_sub / 100.0 * width, step)
                        writer.add_scalar(f'{phase}_subepoch/loss',
                                          loss_acc_sub / 100.0, step)

                        delta_acc_sub = 0
                        delta_conv_acc_sub = 0
                        valid_acc_sub = 0

                    print(f"batch {i_batch} loss: {loss_acc_sub/100}")
                    loss_acc_sub = 0


            writer.add_scalar(f'{phase}/disparity_error',
                              delta_acc / valid_acc * width, step)
            writer.add_scalar(f'{phase}/disparity_error(fully_conv)',
                              delta_conv_acc / float(dataset_sizes[phase]) * width, step)
            if phase == "train":
                writer.add_scalar(f'{phase}/loss',
                                  loss_acc / float(dataset_sizes[phase]), step)
            #write out result
            if phase == 'val':
                disparity = delta_acc / valid_acc * width

                print(f"{phase} disparity error: {disparity}")

                # store at the end of a validation phase of this epoch!
                if not math.isnan(delta_acc) and not math.isinf(delta_acc):
                    if isinstance(model, nn.DataParallel):
                        module = model.module
                    else:
                        module = model

                    print("storing network")
                    torch.save(module.backbone, f"trained_models/{args.experiment_name}_backbone_chk.pt")
                    torch.save(module.regressor, f"trained_models/{args.experiment_name}_regressor_chk.pt")
                    torch.save(module.regressor_conv, f"trained_models/{args.experiment_name}_regressor_conv_chk.pt")

                    if disparity < min_test_disparity:
                        print("storing best network")
                        min_test_disparity = disparity
                        torch.save(module.backbone, f"trained_models/{args.experiment_name}_backbone.pt")
                        torch.save(module.regressor, f"trained_models/{args.experiment_name}_regressor.pt")
                        torch.save(module.regressor_conv, f"trained_models/{args.experiment_name}_regressor_conv.pt")

        scheduler.step()
    writer.close()

if __name__ == '__main__':
    train()