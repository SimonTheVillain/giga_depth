import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss
import torch.optim as optim
from dataset.dataset_rendered_stereo import DatasetRenderedStereo
from model.model_1 import Model1
from torch.utils.data import DataLoader
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

#TODO: train via using grid_sample
# https://pytorch.org/docs/stable/nn.functional.html?highlight=grid_sample#torch.nn.functional.grid_sample

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


def sqr_loss(output, mask, gt, alpha, enable_mask, use_smooth_l1=True):
    width = 1280
    subpixel = 30  # targeted subpixel accuracy
    l1_scale = width * subpixel
    if enable_mask:
        mean = torch.mean(mask) + 0.0001  # small epsilon for not crashing!
        if use_smooth_l1:
            loss = (alpha / l1_scale) * torch.mean(smooth_l1(output[:, [0], :, :] * l1_scale, gt * l1_scale) * mask)
        else:
            loss = alpha * torch.mean(((output[:, 0, :, :] - gt) * mask) ** 2)
        loss_unweighted = loss.item()
        loss_disparity = loss.item() / mean.item()
        loss_mask = torch.mean(torch.abs(output[:, 1, :, :] - mask))  # todo: maybe replace this with the hinge loss
        loss += loss_mask
        loss_mask = loss_mask.item()
        loss_unweighted += loss_mask
    else:
        # loss = alpha * torch.mean((output[:, 0, :, :] - gt) ** 2)
        if use_smooth_l1:
            loss = (alpha / l1_scale) * torch.mean(smooth_l1(output[:, [0], :, :] * l1_scale, gt * l1_scale))
        else:
            loss = alpha * torch.mean((output[:, 0, :, :] - gt) ** 2)
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
    # plt.imshow((mask[0, 0, :] == 0).float().cpu())

    # TODO: find out why this is all zero
    # print(output.shape)
    # plt.imshow(output[0, 0, :].cpu().detach())

    # print(gt.shape)
    # plt.imshow((gt[0, :]).cpu())
    # plt.imshow((gt[0, 0, :] == 0).cpu())
    # plt.show()

    return loss, loss_unweighted, loss_disparity, loss_mask


def reprj_loss(left, right, half_res=False, debug_right=None, debug_gt_l=None, debug_gt_r=None, gt_l_d=None, gt_r_d=None):
    device = left.device
    fxr = 1115.44
    cxr = 604.0
    fxl = 1115.44
    cxl = 604.0
    cxr = cxl = 608.0 #1216/2 (lets put it right in the center since we are working on
    fxp = 1115.44
    cxp = 640.0# put the center right in the center
    b1 = 0.0634
    b2 = 0.07501
    epsilon = 0.01 #very small pixel offsets should be forbidden
    if half_res:
        fxr = fxr * 0.5
        cxr = cxr * 0.5
        fxl = fxl * 0.5
        cxl = cxl * 0.5
        #fxp = fxp * 0.5
        #cxp = cxp * 0.5
    xp = right[:, [0], :, :] * 1280.0#float(right.shape[3])
    #xp = debug_gt_r * 1280.0 #debug
    xr = np.asmatrix(np.array(range(0, right.shape[3]))).astype(np.float32)
    xr = torch.tensor(np.matlib.repeat(xr, right.shape[2], 0), device=device)
    y = np.asmatrix(np.array(range(0, right.shape[2]))).astype(np.float32)
    y = torch.tensor(np.transpose(np.matlib.repeat(y, right.shape[3], 0)), device=device)
    #print(xp.shape)
    #print(y.shape)
    #print(xr.shape)
    #print((1280 - cxp) * fxr)
    #print((1128 - cxr) * fxp)
    #print((1280 - cxp) * fxr - (1128 - cxr) * fxp)
    z_ = (xp - cxp) * fxr - (xr - cxr) * fxp
    #z_ = -z_ #todo: remove

    loss1 = -z_ + epsilon #torch.div(-1.0, z_)# loss one is for where the depth estimate is negative
    loss = loss1 * 0.001
    z = torch.div(b1 * fxp * fxr,  z_)

    xr_space = z * (xr - cxr) * (1.0 / fxr) #x position in cartesioan space in front of right camera
    xl = torch.div((xr_space + b2), z) * fxl + cxl #position to read out pixel in the left image

    y_ex = y.unsqueeze(0).repeat([left.shape[0], 1, 1, 1])
    y_ex = y_ex * (2.0 / float(left.shape[2]-1)) - 1.0
    xl = xl * (2.0 / float(left.shape[3]-1)) - 1.0
    #print(y_ex.shape)
    #print(xl.shape)
    grid = torch.cat((xl.squeeze(1).unsqueeze(-1), y_ex.squeeze(1).unsqueeze(-1)), -1)
    #print(grid.shape)
    ansn = torch.ones(xl.shape, dtype=torch.float32, device=device)
    weights = F.grid_sample(ansn[:, [0], :, :], grid, padding_mode='zeros')
    xp_l = F.grid_sample(left[:, [0], :, :], grid, padding_mode='border') * 1280#float(right.shape[3])
    loss2 = torch.abs(xp_l - xp)
    #loss2 = torch.mul(torch.abs(xp_l - xp), weights) #loss 2 is for wherever the depth estimate makes the slightest of sense
    loss[z_ > epsilon] = loss2[z_ > epsilon]
    mask = loss * 0.0# create a mask with zeros
    mask[z_ > epsilon] = 1
    #print(torch.any(torch.isnan(loss1)))
    #print(torch.any(torch.isnan(loss2)))
    #print(torch.any(torch.isnan(loss)))
    debug = True
    if debug:
        fig = plt.figure()
        count_y = 4
        count_x = 2
        fig.add_subplot(count_y, count_x, 1)
        plt.imshow(debug_right[0, 0, :, :].detach().cpu(), vmin=0, vmax=0.3, cmap='gist_gray')
        #plt.imshow(debug_right[0, 0, :, :].detach().cpu(), vmin=0, vmax=1)
        plt.title("Intensity")
        fig.add_subplot(count_y, count_x, 2)
        plt.imshow(z[0, 0, :, :].detach().cpu(), vmin=0, vmax=10)
        plt.title("depth")
        fig.add_subplot(count_y, count_x, 3)
        plt.imshow(xl[0, 0, :, :].detach().cpu(), vmin=0, vmax=right.shape[3])
        plt.title("x readout position")
        fig.add_subplot(count_y, count_x, 4)
        plt.imshow(loss1[0, 0, :, :].detach().cpu())
        plt.title("loss1")
        fig.add_subplot(count_y, count_x, 5)
        plt.imshow(loss2[0, 0, :, :].detach().cpu())
        plt.title("loss2")
        fig.add_subplot(count_y, count_x, 6)
        plt.imshow(loss[0, 0, :, :].detach().cpu())
        plt.title("combined_loss")
        fig.add_subplot(count_y, count_x, 7)
        plt.imshow(mask[0, 0, :, :].detach().cpu())
        plt.title("mask")
        fig.add_subplot(count_y, count_x, 8)
        plt.imshow(weights[0, 0, :, :].detach().cpu())
        plt.title("weights")
        #fig.add_subplot(count_y, count_x, 9)
        #plt.imshow(xp_l[0, 0, :, :].detach().cpu())
        #plt.title("reprojected")
        #fig.add_subplot(count_y, count_x, 10)
        #plt.imshow(gt_r_d[0, :, :].detach().cpu())
        #plt.title("reprojected")
        plt.show()

    loss = torch.mean(loss)
    return loss



def train():
    dataset_path = "/media/simon/ssd_data/data/reduced_0_08"
    dataset_path = "D:/dataset_filtered"
    writer = SummaryWriter('tensorboard/experiment12')

    model_path_src = "trained_models/model_1_5.pt"
    load_model = True
    model_path_dst = "trained_models/model_1_5_reproj.pt"
    crop_div = 4
    crop_res = (896/crop_div, 1216)
    store_checkpoints = True
    num_epochs = 500
    batch_size = 1
    num_workers = 1
    show_images = False
    shuffle = False
    half_res = True
    enable_mask = False
    alpha = 10
    use_smooth_l1 = True

    min_test_batch_loss = 100000.0


    if load_model:
        model = torch.load(model_path_src)
        model.eval()
    else:
        model = Model1()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    # for param_tensor in net.state_dict():
    #    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=0.001)

    #the filtered dataset
    datasets = {
        'train': DatasetRenderedStereo(dataset_path, 0, 8000, crop_res),
        'val': DatasetRenderedStereo(dataset_path, 8000, 9000, crop_res),
        'test': DatasetRenderedStereo(dataset_path, 9000, 10000, crop_res)
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

            running_loss = 0.0
            subbatch_loss = 0.0
            for i_batch, sampled_batch in enumerate(dataloaders[phase]):
                step = step + 1
                l, r, y, gt_l, gt_r, gt_l_d, gt_r_d = \
                    sampled_batch["image_left"], sampled_batch["image_right"], \
                    sampled_batch["vertical"], sampled_batch["gt_l"], sampled_batch["gt_r"], \
                    sampled_batch["gt_l_d"], sampled_batch["gt_r_d"]

                if torch.cuda.device_count() == 1:
                    l = l.cuda()
                    r = r.cuda()
                    y = y.cuda()
                    gt_l = gt_l.cuda()
                    gt_r = gt_r.cuda()
                l = l[:, None, :, :]
                r = r[:, None, :, :]
                y = y[:, None, :, :]
                print(l.shape)
                print(r.shape)
                l_cat = torch.cat((l, y), 1)
                print(l_cat.shape)
                r_cat = torch.cat((r, y), 1)
                # plt.imshow(input[0, 0, :, :])
                # plt.show()

                # plt.imshow(input[0, 1, :, :])
                # plt.show()

                # plt.imshow(mask[0, 0, :, :])
                # plt.show()
                # print(np.sum(np.array((input != input).cpu()).astype(np.int), dtype=np.int32))

                if phase == 'train':

                    outputs_l, latent_l = model(l_cat)
                    outputs_r, latent_r = model(r_cat)
                    optimizer.zero_grad()
                    loss = reprj_loss(outputs_l, outputs_r, half_res, r_cat, gt_l, gt_r, gt_l_d, gt_l_d)
                    #loss, loss_unweighted, loss_disparity, loss_mask = \
                    #    sqr_loss(outputs, mask.cuda(), gt.cuda(),
                    #             alpha=alpha, enable_mask=enable_mask, use_smooth_l1=use_smooth_l1)
                    loss.backward()
                    # print(net.conv_dwn_0_1.bias)
                    # print(net.conv_dwn_0_1.bias.grad == 0)
                    # print(net.conv_dwn_0_1.weight.grad == 0)
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs_l, latent_l = model(l_cat)
                        outputs_r, latent_r = model(r_cat)
                    loss = reprj_loss(outputs_l, outputs_r, half_res, r_cat)

                writer.add_scalar('loss_disparity/' + phase, loss.item(), step)


                subbatch_loss += loss.item()
                # if i_batch == 0:

                if show_images:
                    fig = plt.figure()

                    fig.add_subplot(2, 3, 1)
                    plt.imshow(input[0, 0, :, :].cpu().detach().numpy(), vmin=0, vmax=1)

                    fig.add_subplot(2, 3, 4)
                    plt.imshow(input[0, 1, :, :].cpu().detach().numpy(), vmin=0, vmax=1)

                    fig.add_subplot(2, 3, 2)
                    # plt.imshow(gt[0, 0, :, :].cpu().detach().numpy(), vmin=0, vmax=1)
                    plot_disp(gt[0, 0, :, :].cpu().detach().numpy(), -0.04, 0.04)

                    fig.add_subplot(2, 3, 5)
                    plt.imshow(mask[0, 0, :, :].cpu().detach().numpy(), vmin=0, vmax=1)

                    fig.add_subplot(2, 3, 3)
                    # plt.imshow(outputs[0, 0, :, :].cpu().detach().numpy(), vmin=0, vmax=1)
                    plot_disp(outputs[0, 0, :, :].cpu().detach().numpy(), -0.04, 0.04,
                              mask=mask[0, 0, :, :].cpu().detach().numpy())

                    fig.add_subplot(2, 3, 6)
                    plt.imshow(outputs[0, 1, :, :].cpu().detach().numpy(), vmin=0, vmax=1)

                    plt.show()
                # print("FUCK YEAH")
                if i_batch % 100 == 99:
                    print("batch {} loss {}".format(i_batch, str(subbatch_loss / 100)))
                    writer.add_scalar('loss/' + phase, subbatch_loss / 100, step)
                    subbatch_loss = 0
                    # pass
                    # print(outputs.shape)
                    # test = torch.cat((input[0, :, :, :], input[0, :, :, :], input[0, :, :, :]),0)
                    # plt.imshow(input[0, 0, :, :].detach().cpu())
                    # plt.imshow(outputs[0, 1, :, :].detach().cpu())
                    # writer.add_image('images', torch.cat((input[0, :, :, :], input[0, :, :, :], input[0, :, :, :]), 0))
                    # writer.add_graph(net, input.cuda())
                    # plt.show()
                # plt.show(block=False)
                # plt.pause(0.001)

                # print("another 100 loss = ", str(loss))
                # for var_name in optimizer.state_dict():
                #    print(var_name, "\t", optimizer.state_dict()[var_name])
                # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

                running_loss += loss.item() * batch_size  # loss.item() * input.size(0)

            print("size of dataset " + str(dataset_sizes[phase]))
            epoch_loss = running_loss / dataset_sizes[phase]  # dataset_sizes[phase]

            writer.add_scalar('epoch_loss/' + phase, epoch_loss, step)
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # print(net.conv_dwn_0_1.bias)
            # print(net.conv_dwn_0_1.bias.grad == 0)
            # print(net.conv_dwn_0_1.weight.grad == 0)

            # store at the end of a epoch
            if phase == 'val' and store_checkpoints:
                if epoch_loss < min_test_batch_loss:
                    print("storing network")
                    min_test_batch_loss = epoch_loss
                    if isinstance(model, nn.DataParallel):
                        torch.save(model.module, model_path_dst)
                    else:
                        torch.save(model, model_path_dst)

    writer.close()

    # TODO: test at end of epoch

    # TODO: at the end of all validate


if __name__ == '__main__':
    train()
