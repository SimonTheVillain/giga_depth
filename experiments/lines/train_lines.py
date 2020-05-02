import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.optim as optim
from experiments.lines.dataset_lines import DatasetLines
from experiments.lines.model_lines_1 import Model_Lines_1
from experiments.lines.model_lines_2 import Model_Lines_2
from experiments.lines.model_lines_3 import Model_Lines_3
from torch.utils.data import DataLoader
import numpy as np
import os

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


def calc_depth_right(right, half_res=False):
    device = right.device
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


    xp = right[:, [0], :, :] * 1280.0#float(right.shape[3])
    #xp = debug_gt_r * 1280.0 #debug
    xr = np.asmatrix(np.array(range(0, right.shape[3]))).astype(np.float32)
    xr = torch.tensor(np.matlib.repeat(xr, right.shape[2], 0), device=device)
    xr = xr.unsqueeze(0).unsqueeze(0).repeat((right.shape[0], 1, 1, 1))
    z_ = (xp - cxp) * fxr - (xr - cxr) * fxp

    z = torch.div(b1 * fxp * fxr, z_)
    return z

def l1_and_mask_loss(output, mask, gt, enable_masking=True, use_smooth_l1=False, debug_depth=None):
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

    loss_depth = torch.mean(torch.abs(calc_depth_right(output[:, [0], :, :]) -
                                      calc_depth_right(gt)))
    debug = False
    if debug:
        z_gt = calc_depth_right(gt)
        z = calc_depth_right(output[:, [0], :, :])
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

def train():
    dataset_path = 'D:/dataset_filtered_strip_100_31'

    if os.name == 'nt':
        dataset_path = 'D:/dataset_filtered_strip_100_31'

    writer = SummaryWriter('tensorboard/train_lines3')

    model_path_src = "../../trained_models/model_stripe_3.pt"
    load_model = True
    model_path_dst = "../../trained_models/model_stripe_3.pt"
    store_checkpoints = True
    num_epochs = 500
    batch_size = 16# 6
    num_workers = 4# 8
    show_images = True
    shuffle = False
    enable_mask = True
    alpha = 0.1
    use_smooth_l1 = False
    learning_rate = 0.01# formerly it was 0.001 but alpha used to be 10 # maybe after this we could use 0.01 / 1280.0
    #learning_rate = 0.00001# should be about 0.001 for disparity based learning
    momentum = 0.90
    projector_width = 1280
    batch_accumulation = 1


    min_test_epoch_loss = 100000.0


    if load_model:
        model = torch.load(model_path_src)
        model.eval()
    else:
        model = Model_Lines_3()



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    # for param_tensor in net.state_dict():
    #    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = optim.Adam(net.parameters(), lr=0.001)

    #the whole unity rendered dataset

    #the filtered dataset
    datasets = {
        'train': DatasetLines(dataset_path, 0, 8000),
        'val': DatasetLines(dataset_path, 8000, 9000),
        'test': DatasetLines(dataset_path, 9000, 9999)
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

            loss_disparity_running = 0.0
            loss_mask_running = 0.0
            loss_depth_running = 0.0

            loss_mask_subepoch = 0.0
            loss_depth_subepoch = 0.0
            loss_disparity_subepoch = 0.0
            for i_batch, sampled_batch in enumerate(dataloaders[phase]):
                step = step + 1
                image_r, mask, gt = sampled_batch["image"], sampled_batch["mask"], sampled_batch["gt"]
                if torch.cuda.device_count() == 1:
                    image_r = image_r.cuda()
                    mask = mask.cuda()
                    gt = gt.cuda()

                # plt.imshow(input[0, 0, :, :])
                # plt.show()

                # plt.imshow(input[0, 1, :, :])
                # plt.show()

                # plt.imshow(mask[0, 0, :, :])
                # plt.show()
                # print(np.sum(np.array((input != input).cpu()).astype(np.int), dtype=np.int32))

                if phase == 'train':
                    outputs, latent = model(image_r)
                    if (i_batch - 1) % batch_accumulation == 0:
                        optimizer.zero_grad()
                    loss_disparity, loss_mask, loss_depth = \
                        l1_and_mask_loss(outputs, mask.cuda(), gt.cuda(),
                                enable_masking=enable_mask, use_smooth_l1=use_smooth_l1)
                    loss = loss_disparity + alpha * loss_mask
                        #loss = loss_mask
                    loss.backward()
                    # print(net.conv_dwn_0_1.bias)
                    # print(net.conv_dwn_0_1.bias.grad == 0)
                    # print(net.conv_dwn_0_1.weight.grad == 0)
                    if i_batch % batch_accumulation == 0:
                        optimizer.step()
                else:
                    with torch.no_grad():
                        outputs, latent = model(image_r.cuda())
                        loss_disparity, loss_mask, loss_depth = \
                            l1_and_mask_loss(outputs, mask.cuda(), gt.cuda(),
                                     enable_masking=enable_mask, use_smooth_l1=use_smooth_l1)
                        loss = loss_disparity + alpha * loss_mask

                writer.add_scalar('batch_{}/loss_combined'.format(phase), loss.item() , step)
                writer.add_scalar('batch_{}/loss_disparity'.format(phase),
                                  loss_disparity.item() * projector_width, step)
                writer.add_scalar('batch_{}/loss_mask'.format(phase), loss_mask.item(), step)
                writer.add_scalar('batch_{}/loss_depth'.format(phase), loss_depth.item(), step)

                #print("DEBUG: batch {} loss {}".format(i_batch, str(loss_unweighted)))

                #print("batch {} loss {} , mask_loss {} , depth_loss {}".format(i_batch, loss_disparity.item(),
                #                                                                loss_mask.item(), loss_depth.item()))
                loss_mask_subepoch += loss_mask.item()
                loss_depth_subepoch += loss_depth.item()
                loss_disparity_subepoch += loss_disparity.item()
                # if i_epoch == 0:

                if show_images:
                    fig = plt.figure()
                    fig.add_subplot(2, 1, 1)
                    plt.imshow(image_r[0, 0, :, :].cpu().detach().numpy(), vmin=0, vmax=1)

                    fig.add_subplot(2, 1, 2)
                    data_1 = gt[0, 0, 0, :].cpu().detach().numpy().flatten()
                    data_2 = outputs[0, 0, 0, :].cpu().detach().numpy()
                    x = np.array(range(0, data_1.shape[0]))
                    plt.plot(x, data_2, x, data_1)
                    # plt.show()

                    depth_1 = calc_depth_right(gt[:, [0], :, :], half_res=False)[0, 0, 0, :].cpu().detach().numpy()
                    depth_2 = calc_depth_right(outputs[:, [0], :, :])[0, 0, 0, :].cpu().detach().numpy()
                    fig = plt.figure()
                    plt.plot(x, depth_1, x, depth_2)
                    plt.show()

                # print("FUCK YEAH")
                if i_batch % 100 == 99:
                    print("batch {} loss {}".format(i_batch, loss_disparity_subepoch / 100))
                    writer.add_scalar('subepoch_{}/loss_disparity'.format(phase),
                                      loss_disparity_subepoch / 100.0 * projector_width, step)
                    writer.add_scalar('subepoch_{}/loss_mask'.format(phase), loss_mask_subepoch / 100.0, step)
                    writer.add_scalar('subepoch_{}/loss_deph'.format(phase), loss_depth_subepoch / 100, step)
                    loss_disparity_subepoch = 0
                    loss_depth_subepoch = 0
                    loss_mask_subepoch = 0
                    subepoch_loss = 0
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

                loss_disparity_running += loss_disparity.item() * image_r.size(0)
                loss_mask_running += loss_mask.item() * image_r.size(0)
                loss_depth_running += loss_depth.item() * image_r.size(0)

            print("size of dataset " + str(dataset_sizes[phase]))

            epoch_loss = (loss_disparity_running + loss_mask_running) / dataset_sizes[phase]
            writer.add_scalar('epoch_{}/loss_disparity'.format(phase),
                              loss_disparity_running / dataset_sizes[phase] * projector_width, step)
            writer.add_scalar('epoch_{}/loss_mask'.format(phase), loss_mask_running / dataset_sizes[phase], step)
            writer.add_scalar('epoch_{}/loss_depth'.format(phase), loss_depth_running / dataset_sizes[phase], step)
            print('{} Loss: {:.4f}'.format( phase, loss_disparity_running / dataset_sizes[phase]))

            # print(net.conv_dwn_0_1.bias)
            # print(net.conv_dwn_0_1.bias.grad == 0)
            # print(net.conv_dwn_0_1.weight.grad == 0)

            # store at the end of a epoch
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



