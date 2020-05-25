import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.nn.functional as F
import torch.optim as optim
from experiments.lines.dataset_lines import DatasetLines
from experiments.lines.model_lines_c_1 import Model_Lines_C_1
from experiments.lines.model_lines_c_n import Model_Lines_C_n
from experiments.lines.model_lines_c_r_n import Model_Lines_C_R_n
from experiments.lines.model_lines_CR2_n import Model_Lines_CR2_n
from experiments.lines.model_lines_CR3_n import Model_Lines_CR3_n
from experiments.lines.model_lines_CR4_n import Model_Lines_CR4_n
from experiments.lines.model_lines_CR5_n import Model_Lines_CR5_n
from experiments.lines.model_lines_CR6_n import Model_Lines_CR6_n
from experiments.lines.model_lines_CR7_n import Model_Lines_CR7_n
from torch.utils.data import DataLoader
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter


# if not torch.cuda.is_available():
#    raise Exception("No GPU found, please run without --cuda")


def plot_disp(input, vmin, vmax, mask=None):
    vmin = 0
    vmax = 1
    width = input.shape[1]
    height = input.shape[0]
    gradient = np.array(np.arange(0, width)) / width
    gradient = np.tile(gradient, (height, 1))
    output = input
    # output = input - gradient
    if mask is not None:
        pass
        # output = output*mask
    plt.imshow(output, vmin=vmin, vmax=vmax)


smooth_l1 = nn.SmoothL1Loss(reduction='none')


def calc_x_pos(class_inds, regressions, class_count, neighbourhood_regression):
    #print(class_inds.shape)
    #print(regressions.shape)
    regressions = torch.gather(regressions, dim=1, index=class_inds)
    if neighbourhood_regression:
        regressions = (regressions * (1.0/3.0) - 1.0/3.0) * (1.0 / class_count)
    else:
        regressions = regressions * (1.0 / class_count)
    x = class_inds * (1.0 / class_count) + regressions
    return x


def calc_depth_right(right_x_pos, half_res=False):
    device = right_x_pos.device
    projector_width = 1280.0
    fxr = 1115.44
    cxr = 604.0
    fxl = 1115.44
    cxl = 604.0
    cxr = cxl = 608.0  # 1216/2 (lets put it right in the center since we are working on
    fxp = 1115.44
    cxp = 640.0  # put the center right in the center
    b1 = 0.0634
    b2 = 0.07501
    epsilon = 0.01  # very small pixel offsets should be forbidden
    if half_res:
        fxr = fxr * 0.5
        cxr = cxr * 0.5
        fxl = fxl * 0.5
        cxl = cxl * 0.5

    xp = right_x_pos[:, [0], :, :] * projector_width  # float(right.shape[3])
    # xp = debug_gt_r * 1280.0 #debug
    xr = np.asmatrix(np.array(range(0, right_x_pos.shape[3]))).astype(np.float32)
    xr = torch.tensor(np.matlib.repeat(xr, right_x_pos.shape[2], 0), device=device)
    xr = xr.unsqueeze(0).unsqueeze(0).repeat((right_x_pos.shape[0], 1, 1, 1))
    z_ = (xp - cxp) * fxr - (xr - cxr) * fxp

    z = torch.div(b1 * fxp * fxr, z_)
    return z


def combo_loss(classes, regressions, mask, gt_mask, gt, neighbourhood_regression, enable_masking=True, class_count=0):
    # calculate the regression loss on the groundtruth label
    gt_class_label = torch.clamp((gt * class_count).type(torch.int64), 0, class_count - 1)
    reg = calc_x_pos(gt_class_label,
                     regressions, class_count, neighbourhood_regression)
    loss_reg = torch.abs(reg - gt)
    if neighbourhood_regression:
        #neighbour left and right
        reg = calc_x_pos(torch.clamp(gt_class_label - 1, 0, class_count-1).type(torch.int64),
                         regressions, class_count, neighbourhood_regression)
        loss_reg += torch.abs(reg - gt)
        reg = calc_x_pos(torch.clamp(gt_class_label + 1, 0, class_count-1).type(torch.int64),
                         regressions, class_count, neighbourhood_regression)
        loss_reg += torch.abs(reg - gt)
    # other idea: regression only where class is correct:
    #create mask :
    #class_pred = torch.argmax(classes, dim=1).unsqueeze(dim=1)
    #class_hit_mask = (classes == gt_class_label).type(torch.float32)
    #loss_reg = loss_reg * class_hit_mask
    # TODO: add regression loss for neighbouring labels

    # Calculate the class loss:
    target_label = (gt * class_count).type(torch.int64)  # i know 64 bit is a waste!
    target_label = torch.clamp(target_label, 0, class_count - 1)
    target_label = target_label.squeeze(1)
    loss_class = F.cross_entropy(classes, target_label, reduction='none')

    # calculate the true offset in disparity
    class_pred = torch.argmax(classes, dim=1).unsqueeze(dim=1)
    disp_pure = calc_x_pos(class_pred, regressions, class_count, neighbourhood_regression)
    loss_disp = torch.abs(gt - disp_pure)
    if enable_masking:
        loss_class = loss_class * gt_mask
        loss_reg = loss_reg * gt_mask

    # loss for mask
    loss_mask = torch.abs(gt_mask - mask)

    # depth loss
    loss_depth = torch.abs(calc_depth_right(disp_pure) - calc_depth_right(gt))

    if True:
        loss_disp = torch.mean(loss_disp)
        loss_depth = torch.mean(loss_depth)
        loss_mask = torch.mean(loss_mask)
        loss_reg = torch.mean(loss_reg)
        loss_class = torch.mean(loss_class)
    return loss_class, loss_reg, loss_mask, loss_depth, loss_disp


def train():
    projector_width = 1280.0
    dataset_path = '/media/simon/ssd_data/data/dataset_filtered_strip_100_31'

    if os.name == 'nt':
        dataset_path = 'D:/dataset_filtered_strip_100_31'

    writer = SummaryWriter('tensorboard/Model_Lines_CR7_256_no_nb')#model_stripe_c_r_256_lr_1_no_nb_3')
    #writer = SummaryWriter('tensorboard/dump')#model_stripe_c_r_256_lr_1_no_nb_3')

    model_path_src = "../../trained_models/Model_Lines_CR7_256_chckpt.pt"
    load_model = True
    model_path_dst = "../../trained_models/Model_Lines_CR7_256_no_nb.pt"
    store_checkpoints = True
    model_path_unconditional = "../../trained_models/Model_Lines_CR7_256_chckpt.pt"
    unconditional_chckpts = True
    num_epochs = 5000
    batch_size = 32# 16
    num_workers = 8  # 8
    show_images = False
    shuffle = False
    enable_mask = True
    alpha_mask = 0.01
    alpha_regression = 500 # TODO: check this out for even faster training of subpixel accuracy
    alpha_regression = 100 # 10 is a good value to improve subpixel accuracy only 1 is tested to train classification
    use_smooth_l1 = False
    neighbourhood_regression = False
    learning_rate = 0.01  # formerly it was 0.001 but alpha_mask used to be 10 # maybe after this we could use 0.01 / 1280.0
    learning_rate = 0.1 #0.1 for best convergence at the initial iterations (getting the classification going)
    learning_rate = 0.01 # get some extra regression performance later on
    #learning_rate = 1.0
    # learning_rate = 0.00001# should be about 0.001 for disparity based learning
    momentum = 0.90
    projector_width = 1280
    batch_accumulation = 1
    class_count = 256

    min_test_epoch_loss = 100000.0

    if load_model:
        model = torch.load(model_path_src)
        model.eval()
        # copy over parameter of old model
        #model = Model_Lines_C_R_n(class_count, model)
    else:
        model = Model_Lines_CR7_n(class_count)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    # for param_tensor in net.state_dict():
    #    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = optim.Adam(net.parameters(), lr=0.001)

    # the whole unity rendered dataset

    # the filtered dataset
    datasets = {
        'train': DatasetLines(dataset_path, 0, 8000),
        'val': DatasetLines(dataset_path, 8000, 9000),
        'test': DatasetLines(dataset_path, 9000, 9999)
    }

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                                  shuffle=shuffle, num_workers=num_workers)
                   for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}

    step = 225900
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

            loss_class_running = 0.0
            loss_reg_running = 0.0
            loss_mask_running = 0.0
            loss_depth_running = 0.0
            loss_disp_running = 0.0

            loss_class_subepoch = 0.0
            loss_reg_subepoch = 0.0
            loss_mask_subepoch = 0.0
            loss_depth_subepoch = 0.0
            loss_disp_subepoch = 0.0
            for i_batch, sampled_batch in enumerate(dataloaders[phase]):
                step = step + 1
                image_r, mask_gt, gt = sampled_batch["image"], sampled_batch["mask"], sampled_batch["gt"]
                if torch.cuda.device_count() == 1:
                    image_r = image_r.cuda()
                    mask_gt = mask_gt.cuda()
                    gt = gt.cuda()

                if phase == 'train':
                    class_output, regression_output, mask_output, latent = model(image_r)
                    if (i_batch - 1) % batch_accumulation == 0:
                        optimizer.zero_grad()

                    loss_class, loss_reg, loss_mask, loss_depth, loss_disp = \
                        combo_loss(class_output, regression_output, mask_output, mask_gt.cuda(), gt.cuda(),
                                   neighbourhood_regression, enable_masking=enable_mask, class_count=class_count)
                    loss = loss_class + alpha_regression * loss_reg + alpha_mask * loss_mask
                    loss.backward()
                    if i_batch % batch_accumulation == 0:
                        optimizer.step()
                else:
                    with torch.no_grad():
                        class_output, regression_output, mask_output, latent = model(image_r)
                        loss_class, loss_reg, loss_mask, loss_depth, loss_disp = \
                            combo_loss(class_output, regression_output, mask_output, mask_gt.cuda(), gt.cuda(),
                                       neighbourhood_regression, enable_masking=enable_mask, class_count=class_count)
                        loss = loss_class + alpha_regression * loss_reg + alpha_mask * loss_mask

                writer.add_scalar('batch_{}/loss_combined'.format(phase), loss.item(), step)
                writer.add_scalar('batch_{}/loss_class'.format(phase), loss_class.item(), step)
                writer.add_scalar('batch_{}/loss_reg'.format(phase), loss_reg.item() * projector_width, step)
                writer.add_scalar('batch_{}/loss_mask'.format(phase), loss_mask.item(), step)
                writer.add_scalar('batch_{}/loss_depth'.format(phase), loss_depth.item(), step)
                writer.add_scalar('batch_{}/loss_disp'.format(phase), loss_disp.item() * projector_width, step)

                if False:
                    print("combined = {}, class = {}, regression = {}, mask = {}".format(loss.item(),
                                                                                         loss_class.item(),
                                                                                         loss_reg.item(),
                                                                                         loss_mask.item()))

                loss_class_subepoch += loss_class.item()
                loss_reg_subepoch += loss_reg.item()
                loss_mask_subepoch += loss_mask.item()
                loss_depth_subepoch += loss_depth.item()
                loss_disp_subepoch += loss_disp.item()

                if show_images:
                    fig = plt.figure()
                    fig.add_subplot(2, 1, 1)
                    plt.imshow(image_r[0, 0, :, :].cpu().detach().numpy(), vmin=0, vmax=1)

                    fig.add_subplot(2, 1, 2)
                    disp = calc_x_pos(torch.argmax(class_output, dim=1).unsqueeze(1),
                                      regression_output, class_count, neighbourhood_regression)
                    data_1 = disp[0, 0, 0, :].cpu().detach().numpy().flatten()
                    data_2 = (torch.argmax(class_output, dim=1) * (1.0 / class_count)).cpu().detach().numpy()
                    data_2 = data_2[0, 0, :].flatten()
                    x = np.array(range(0, data_1.shape[0]))
                    data_gt = gt[0, 0, 0, :].cpu().detach().numpy()
                    plt.plot(x, data_gt * projector_width,
                             x, data_2 * projector_width,
                             x, data_1 * projector_width)
                    plt.legend(["groundtruth", "class", "class+regression"])
                    plt.show()

                    depth_1 = calc_depth_right(gt[:, [0], :, :], half_res=False)[0, 0, 0, :].cpu().detach().numpy()
                    depth_2 = torch.argmax(class_output, dim=1) * (1.0 / class_count)
                    depth_2 = depth_2.unsqueeze(dim=1)
                    depth_2 = calc_depth_right(depth_2, half_res=False)[0, 0, 0, :].cpu().detach().numpy()
                    depth_3 = calc_depth_right(disp, half_res=False)[0, 0, 0, :].cpu().detach().numpy()
                    fig = plt.figure()
                    plt.plot(x, depth_1, x, depth_2, x, depth_3)
                    plt.legend(["groundtruth", "class", "class+regression"])
                    plt.show()

                # print("FUCK YEAH")
                if i_batch % 100 == 99:
                    print("batch {} loss {}".format(i_batch, loss_disp_subepoch / 100))

                    writer.add_scalar('subepoch_{}/loss_class'.format(phase),
                                      loss_class_subepoch / 100.0, step)
                    writer.add_scalar('subepoch_{}/loss_reg'.format(phase),
                                      loss_reg_subepoch / 100.0 * projector_width, step)
                    writer.add_scalar('subepoch_{}/loss_disp'.format(phase),
                                      loss_disp / 100.0 * projector_width, step)
                    writer.add_scalar('subepoch_{}/loss_mask'.format(phase), loss_mask_subepoch / 100.0, step)
                    writer.add_scalar('subepoch_{}/loss_deph'.format(phase), loss_depth_subepoch / 100, step)
                    loss_class_subepoch = 0.0
                    loss_reg_subepoch = 0.0
                    loss_mask_subepoch = 0.0
                    loss_depth_subepoch = 0.0
                    loss_disp_subepoch = 0.0

                loss_class_running += loss_class.item() * batch_size
                loss_reg_running += loss_reg.item() * batch_size
                loss_mask_running += loss_mask.item() * batch_size
                loss_depth_running += loss_depth.item() * batch_size
                loss_disp_running += loss_disp.item() * batch_size

            epoch_loss = (loss_class_running + alpha_regression * loss_reg_running + alpha_mask * loss_mask_running) / \
                         dataset_sizes[phase]

            writer.add_scalar('epoch_{}/loss_class'.format(phase), loss_class_running / dataset_sizes[phase], step)
            writer.add_scalar('epoch_{}/loss_reg'.format(phase),
                              loss_reg_running / dataset_sizes[phase] * projector_width, step)
            writer.add_scalar('epoch_{}/loss_disp'.format(phase),
                              loss_disp_running / dataset_sizes[phase] * projector_width, step)
            writer.add_scalar('epoch_{}/loss_mask'.format(phase), loss_mask_running / dataset_sizes[phase], step)
            writer.add_scalar('epoch_{}/loss_depth'.format(phase), loss_depth_running / dataset_sizes[phase], step)

            writer.add_scalar('epoch_{}/loss_epoch'.format(phase),
                              epoch_loss, step)
            print('{} Loss: {:.4f}'.format(phase, loss_disp_running / dataset_sizes[phase]))

            # print(net.conv_dwn_0_1.bias)
            # print(net.conv_dwn_0_1.bias.grad == 0)
            # print(net.conv_dwn_0_1.weight.grad == 0)

            # store at the end of a epoch
            if phase == 'val' and unconditional_chckpts:
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module, model_path_unconditional)
                else:
                    torch.save(model, model_path_unconditional)
            if phase == 'val' and store_checkpoints:
                if epoch_loss < min_test_epoch_loss:
                    print("storing network")
                    min_test_epoch_loss = epoch_loss
                    if isinstance(model, nn.DataParallel):
                        torch.save(model.module, model_path_dst)
                    else:
                        torch.save(model, model_path_dst)

    writer.close()

    # TODO: test at end of epoch

    # TODO: at the end of all validate


if __name__ == '__main__':
    train()
