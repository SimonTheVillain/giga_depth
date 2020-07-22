import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
#from torch.cuda.amp.autocast import autocast
from dataset.dataset_rendered import DatasetRendered
from model.model_CR10_2hs import Model_CR10_2_hsn
from model.model_CR10_3hs import Model_CR10_3_hsn
from model.model_CR10_4hs import Model_CR10_4_hsn
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from os.path import expanduser

import matplotlib
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

#if not torch.cuda.is_available():
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

def depth_loss_func(right, mask, alpha, gt_depth, half_res):
    alpha = alpha
    device = right.device
    fxr = 1115.44
    cxr = 604.0
    cxr = 608.0 #1216/2 (lets put it right in the center since we are working on
    fxp = 1115.44
    cxp = 640.0# put the center right in the center
    b1 = 0.0634
    epsilon = 0.1 #very small pixel offsets should be forbidden
    if half_res:
        fxr = fxr * 0.5
        cxr = cxr * 0.5

    xp = right[:, [0], :, :] * 1280.0#float(right.shape[3])
    #xp = debug_gt_r * 1280.0 #debug
    xr = np.asmatrix(np.array(range(0, right.shape[3]))).astype(np.float32)
    xr = torch.tensor(np.matlib.repeat(xr, right.shape[2], 0), device=device)
    z_ = (xp - cxp) * fxr - (xr - cxr) * fxp
    #z_ = -z_ #todo: remove

    z = torch.div(b1 * fxp * fxr,  z_)
    z[torch.abs(z_) < epsilon] = 0
    loss = alpha * torch.mean(torch.abs(z - gt_depth))

    #mask:
    loss_mask = alpha * torch.mean(torch.abs(right[:, 1, :, :] - mask))
    loss += loss_mask
    loss_mask = loss_mask.item()

    loss_unweighted = loss.item()
    loss_disparity = loss_unweighted

    return loss, loss_unweighted, loss_disparity, loss_mask




def l1_and_mask_loss(output, mask, gt, offsets, half_res=True, enable_masking=True, use_smooth_l1=False, debug_depth=None):
    width = 1280
    subpixel = 30
    l1_scale = width * subpixel
    if use_smooth_l1:
        loss_disparity = (1.0 / l1_scale) * smooth_l1(output[:, [0], :, :] * l1_scale, gt * l1_scale)
    else:
        loss_disparity = torch.abs(output[:, [0], :, :] - gt)

    if enable_masking:
        loss_disparity = loss_disparity * mask

    loss_mask = torch.mean(torch.abs(mask - output[:, [1], :, :]))

    loss_disparity = torch.mean(loss_disparity)

    loss_depth = torch.mean(torch.abs(calc_depth_right(output[:, [0], :, :], offsets, half_res) -
                                      calc_depth_right(gt, offsets, half_res)))
    debug = False
    if debug:
        z_gt = calc_depth_right(gt, offsets)
        z = calc_depth_right(output[:, [0], :, :], offsets)
        fig = plt.figure()
        count_y = 2
        count_x = 2
        fig.add_subplot(count_y, count_x, 1)
        plt.imshow(z_gt[0, 0, :, :].detach().cpu(), vmin=0, vmax=2)#, vmin=0, vmax=10)
        # plt.imshow(debug_right[0, 0, :, :].detach().cpu(), vmin=0, vmax=1)
        plt.title("depth")
        fig.add_subplot(count_y, count_x, 2)
        plt.imshow(z[0, 0, :, :].detach().cpu(), vmin=0, vmax=10)
        plt.title("depth_estimate")

        fig.add_subplot(count_y, count_x, 3)
        plt.imshow(debug_depth[0, 0, :, :].detach().cpu(), vmin=0, vmax=2)
        plt.title("depth_debug")
        plt.show()
    return loss_disparity, loss_mask, loss_depth


def sqr_loss(output, mask, gt, enable_masking, use_smooth_l1=False):
    width = 1280
    subpixel = 30  # targeted subpixel accuracy
    l1_scale = width * subpixel
    if enable_masking:
        mean = torch.mean(mask) + 0.0001  # small epsilon for not crashing! #WHY CRASHING!!!! am i stupid?
        if use_smooth_l1:
            loss = \
                width * (1.0 / l1_scale) * torch.mean(smooth_l1(output[:, [0], :, :] * l1_scale, gt * l1_scale) * mask)
        else:
            loss = width * 1.0 * torch.mean(((output[:, 0, :, :] - gt) * mask) ** 2)
        loss_unweighted = loss.item()
        loss_disparity = loss.item() / mean.item()
        loss_mask = torch.mean(torch.abs(output[:, 1, :, :] - mask))  # todo: maybe replace this with the hinge loss
        loss += loss_mask
        loss_mask = loss_mask.item()
        loss_unweighted += loss_mask
    else:
        # loss = alpha * torch.mean((output[:, 0, :, :] - gt) ** 2)
        if use_smooth_l1:
            loss = width * (1.0 / l1_scale) * torch.mean(smooth_l1(output[:, [0], :, :] * l1_scale, gt * l1_scale))
        else:
            loss = width * 1.0 * torch.mean((output[:, 0, :, :] - gt) ** 2)
        loss_unweighted = loss.item()
        loss_disparity = loss
        loss_mask = torch.mean(torch.abs(output[:, 1, :, :] - mask))  # todo: maybe replace this with the hinge loss
        loss += loss_mask
        loss_mask = loss_mask.item()
        loss_unweighted += loss_mask


    # test = torch.clamp(1.0 - (output[:, 1, :, :] - 0.5) * (fmask - 0.5) * 4.0, min=0) #this should actually do what we want!
    # loss += torch.mean(test)

    # print(test.shape)#todo. debug! the hinge loss is weird
    # loss += torch.mean(torch.max(torch.Tensor([0]).cuda(), 1.0 - (output[:, 1, :, :] - 0.5) * (fmask - 0.5) * 4.0))#hinge loss
    # loss = torch.nn.modules.loss.SmoothL1Loss()

    # print(mask.shape)
     #plt.imshow((mask[0, 0, :] == 0).float().cpu())

    # TODO: find out why this is all zero
    # print(output.shape)
    # plt.imshow(output[0, 0, :].cpu().detach())

    # print(gt.shape)
    # plt.imshow((gt[0, :]).cpu())
    # plt.imshow((gt[0, 0, :] == 0).cpu())
    # plt.show()
    return loss_disparity, loss_mask

def calc_x_pos(class_inds, regression, class_count):
    regression = regression * (1.0 / class_count)
    x = class_inds * (1.0 / class_count) + regression
    return x

def calc_depth_right(right_x_pos, half_res=False):#todo: offset!
    device = right_x_pos.device
    projector_width = 1280.0
    fxr = 1115.44
    cxr = 604.0
    fxl = 1115.44
    cxl = 604.0
    cxr = cxl = 608.0  # 1216/2 (lets put it right in the center since we are working on
    fxp = 1115.44
    cxp = 640.0  # put the center right in the center
    b1 = 0.0634#baseline projector to right camera
    b2 = 0.07501#baselne between both cameras
    epsilon = 0.01  # very small pixel offsets should be forbidden
    if half_res:
        fxr = fxr * 0.5
        cxr = cxr * 0.5
        fxl = fxl * 0.5
        cxl = cxl * 0.5

    xp = right_x_pos[:, [0], :, :] * projector_width  # float(right.shape[3])
    if half_res:
        pass
        #xp = xp - 0.5 #why the downsampling shifted everything by 0.5 pixel?

    xr = np.asmatrix(np.array(range(0, right_x_pos.shape[3]))).astype(np.float32)
    xr = torch.tensor(np.matlib.repeat(xr, right_x_pos.shape[2], 0), device=device)
    xr = xr.unsqueeze(0).unsqueeze(0).repeat((right_x_pos.shape[0], 1, 1, 1))
    z_ = (xp - cxp) * fxr - (xr - cxr) * fxp

    z = torch.div(b1 * fxp * fxr, z_)
    return z

def combo_loss(classes, regression, mask, gt_mask, gt,
               enable_masking=True, class_count=0, half_res=False):
    # calculate the regression loss on the groundtruth label
    gt_class_label = torch.clamp((gt * class_count).type(torch.int64), 0, class_count - 1)
    #print(gt_class_label)
    reg = calc_x_pos(gt_class_label,
                     regression, class_count)
    #print(reg-gt)
    loss_reg = torch.abs(reg - gt)

    # Calculate the class loss:
    gt_class_label = gt_class_label.squeeze(1)
    loss_class = F.cross_entropy(classes, gt_class_label, reduction='none')
    #print(target_label.shape)
    #print(target_label)
    # calculate the true offset in disparity
    class_pred = torch.argmax(classes, dim=1).unsqueeze(dim=1)
    disp_pure = calc_x_pos(class_pred, regression, class_count)
    #print(gt - disp_pure)
    loss_disp = torch.abs(gt - disp_pure)
    if enable_masking:
        loss_class = loss_class * gt_mask
        loss_reg = loss_reg * gt_mask

    # loss for mask
    loss_mask = torch.abs(gt_mask - mask)

    calc_depth_right(gt, half_res)
    # depth loss
    loss_depth = torch.abs(calc_depth_right(disp_pure, half_res) - calc_depth_right(gt, half_res))

    if True:
        loss_disp = torch.mean(loss_disp)
        loss_depth = torch.mean(loss_depth)
        loss_mask = torch.mean(loss_mask)
        loss_reg = torch.mean(loss_reg)
        loss_class = torch.mean(loss_class)
    return loss_class, loss_reg, loss_mask, loss_depth, loss_disp

def train():
    path = expanduser("~/giga_depth_results/")


    half_precision = False
    single_gpu = False
    dataset_path = "/media/simon/ssd_data/data/dataset_reduced_0_08"

    if os.name == 'nt':
        dataset_path = "D:/dataset_filtered"
    writer = SummaryWriter(path + 'tensorboard/CR_10_4hs_2')
    #writer = SummaryWriter('tensorboard/dump')

    model_path_src = path + "trained_models/CR_10_4hs_chckpt.pt"
    load_model = True
    model_path_dst = path + "trained_models/CR_10_4hs_2.pt"
    model_path_unconditional = path + "trained_models/CR_10_4hs_2_chckpt.pt"
    unconditional_chckpts = True
    crop_div = 1
    crop_res = (896, 1216/crop_div)
    store_checkpoints = True
    num_epochs = 5000
    batch_size = 2
    num_workers = 8# 8
    show_images = False
    shuffle = False
    half_res = True
    enable_mask = False
    alpha_mask = 0.1
    alpha_regression = 100 # 10 is a good value to improve subpixel accuracy only 1 is tested to train classification
    alpha_regression = 1#TODO: go back to a weight of 100 for the regression part
    learning_rate = 1.1 #0.1 for the coarse part
    momentum = 0.90
    projector_width = 1280
    batch_accumulation = 1
    class_count = 128
    slices = 8 # 16 slices for the whole image
    single_slice = True

    core_image_height = crop_res[0]
    if single_slice:
        dataset_path = "/media/simon/ssd_data/data/dataset_filtered_slice_142"
        slices = 1
        padding = Model_CR10_3_hsn.padding()
        crop_top = padding
        crop_bottom = padding
        #full resolution slice height
        slice_height = 142#56 (in case of a full image with 16 slices
        crop_res = (142, 1216/crop_div)#56
        core_image_height = 142
        slice_offset = 0
        slice_height_2 = int(slice_height/2)
        slice_offset_2 = int(slice_offset/2)
        #TODO: enable training with padding provided by the input images
        pad_top = False
        pad_bottom = False
        half_res = True

        #all for model 10.2
        padding = 0
        crop_top = padding
        crop_bottom = padding
        crop_res = (142, 1216 / crop_div)  # 56
        core_image_height = 142

        batch_size = 12#12




    min_test_epoch_loss = 100000.0


    if load_model:
        model = torch.load(model_path_src)
        model.eval()
        #model_old = model
        #model = Model_CR10_3_hsn(class_count, core_image_height)
        #model.copy_backbone(model_old)
        #model_old = None
    else:
        #model = Model_CR8_n(class_count, crop_res[0])
        #model = Model_CR10_hsn(slices, class_count, core_image_height, pad_top, pad_bottom)
        #model = Model_CR10_2_hsn(slices, class_count, core_image_height)
        model = Model_CR10_4_hsn(class_count, core_image_height)

    speed_test = False
    if speed_test:
        model.cuda()
        model.eval()
        with torch.no_grad():
            test = torch.rand((1, 1, int(crop_res[0]), crop_res[1]), dtype=torch.float32).cuda()
            torch.cuda.synchronize()
            tsince = int(round(time.time() * 1000))
            for i in range(0, 100):
                model(test)
                torch.cuda.synchronize()
            ttime_elapsed = int(round(time.time() * 1000)) - tsince
            print('test time elapsed {}ms'.format(ttime_elapsed/100))
        model.train()



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1 and not single_gpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    if half_precision:
        model.half()
        #https: // pytorch.org / docs / stable / notes / amp_examples.html
        scaler = GradScaler()

    # for param_tensor in net.state_dict():
    #    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = optim.Adam(net.parameters(), lr=0.001)

    #the whole unity rendered dataset

    #the filtered dataset
    datasets = {
        'train': DatasetRendered(dataset_path, 0, 8000, half_res, crop_res),
        'val': DatasetRendered(dataset_path, 8000, 9000, half_res, crop_res),
        'test': DatasetRendered(dataset_path, 9000, 10000, half_res, crop_res)
    }
    if single_slice:

        datasets = {
            'train': DatasetRendered(dataset_path, 0, 8000, half_res, crop_res, single_slice, crop_top, crop_bottom),
            'val': DatasetRendered(dataset_path, 8000, 9000, half_res, crop_res, single_slice, crop_top, crop_bottom),
            'test': DatasetRendered(dataset_path, 9000, 10000, half_res, crop_res, single_slice, crop_top, crop_bottom)
        }

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                                  shuffle=shuffle, num_workers=num_workers)
                   for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}

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
                image_r, mask_gt, gt, gt_d, offsets = sampled_batch["image"][:, [0], :, :], sampled_batch["mask"], \
                                                 sampled_batch["gt"], sampled_batch["gt_d"], sampled_batch["offset"]


                if False:
                    x = np.array(range(0, gt_d.shape[3]))
                    depth_1 = calc_depth_right(gt[:, [0], :, :], half_res=half_res)[0, 0, 0, :].cpu().detach().numpy()
                    fig = plt.figure()
                    plt.plot(x, depth_1, x, gt_d[0, 0, 0, :].cpu().detach().numpy())
                    plt.legend(["groundtruth", "groundtruth 2"])
                    plt.show()

                offsets = offsets.cuda()
                if torch.cuda.device_count() == 1 or single_gpu:
                    image_r = image_r.cuda()
                    mask_gt = mask_gt.cuda()
                    gt = gt.cuda()
                    offsets = offsets.cuda()
                if half_precision:
                    image_r = image_r.half()
                    mask_gt = mask_gt.half()
                    gt = gt.half()
                    offsets = offsets.half()

                # plt.imshow(input[0, 0, :, :])
                # plt.show()

                # plt.imshow(input[0, 1, :, :])
                # plt.show()

                # plt.imshow(mask[0, 0, :, :])
                # plt.show()
                # print(np.sum(np.array((input != input).cpu()).astype(np.int), dtype=np.int32))

                if phase == 'train':
                    gt_class_inds = torch.clamp((gt * class_count).type(torch.int64), 0, class_count - 1)
                    class_output, regression_output, mask_output, latent = model(image_r, gt_class_inds)

                    #for half precision we probably need autocast
                    #with torch.cuda.amp.autocast():
                    if (i_batch - 1) % batch_accumulation == 0 or half_precision:
                        optimizer.zero_grad()

                    loss_class, loss_reg, loss_mask, loss_depth, loss_disp = \
                        combo_loss(class_output, regression_output, mask_output, mask_gt.cuda(), gt.cuda(),
                                   enable_masking=enable_mask, class_count=class_count, half_res=half_res)

                    loss = loss_class + alpha_regression * loss_reg + alpha_mask * loss_mask

                    if half_precision:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                else:
                    with torch.no_grad():
                        class_output, regression_output, mask_output, latent = model(image_r)
                        loss_class, loss_reg, loss_mask, loss_depth, loss_disp = \
                            combo_loss(class_output, regression_output, mask_output, mask_gt.cuda(), gt.cuda(),
                                       enable_masking=enable_mask, class_count=class_count, half_res=half_res)
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
                                      regression_output, class_count)
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

                    depth_1 = calc_depth_right(gt[:, [0], :, :], half_res=half_res)[0, 0, 0, :].cpu().detach().numpy()
                    depth_2 = torch.argmax(class_output, dim=1) * (1.0 / class_count)
                    depth_2 = depth_2.unsqueeze(dim=1)
                    depth_2 = calc_depth_right(depth_2, half_res=half_res)[0, 0, 0, :].cpu().detach().numpy()
                    depth_3 = calc_depth_right(disp, half_res=half_res)[0, 0, 0, :].cpu().detach().numpy()

                    fig = plt.figure()
                    plt.plot(x, depth_1, x, gt_d[0, 0, 0, :].cpu().detach().numpy())
                    plt.legend(["groundtruth", "groundtruth 2"])
                    plt.show()

                    fig = plt.figure()
                    plt.plot(x, depth_1, x, depth_2, x, depth_3, x, gt_d[0, 0, 0, :].cpu().detach().numpy())
                    plt.legend(["groundtruth", "class", "class+regression", "groundtruth 2"])
                    plt.show()

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
            # print(net.conv_dwn_0_1.weight.grad == 0)
            # store at the end of a epoch
            if phase == 'val' and unconditional_chckpts:
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module, model_path_unconditional)
                else:
                    torch.save(model, model_path_unconditional)
            if phase == 'val' and unconditional_chckpts:
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
