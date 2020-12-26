import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.optim as optim
from dataset.dataset_rendered import DatasetRendered
from model.regressor_2Stage import Regressor2Stage
from model.backbone_6_64 import Backbone6_64
from torch.utils.data import DataLoader
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter




def train():
    dataset_path = "/home/simon/datasets/structure_core_unity"

    experiment_name = "bb64_2stage_simple"

    writer = SummaryWriter(f"tensorboard/{experiment_name}")

    model_path_src = ""
    model_path_dst = f"trained_models/{experiment_name}.pt"

    num_epochs = 5000
    batch_size = 1 # batch size of 1 to begin with!!!!
    num_workers = 8
    alpha = 0.1
    learning_rate = 0.01# formerly it was 0.001 but alpha used to be 10 # maybe after this we could use 0.01 / 1280.0
    #learning_rate = 0.00001# should be about 0.001 for disparity based learning
    momentum = 0.90


    depth_loss = False
    if depth_loss:
        learning_rate = 0.00001# should be about 0.001 for disparity based learning
        momentum = 0.9


    min_test_epoch_loss = 100000.0

    if model_path_src != "":
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

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = optim.Adam(net.parameters(), lr=0.001)

    #the whole unity rendered dataset

    #the filtered dataset
    datasets = {
        'train': DatasetRendered(dataset_path, 0, 8000, half_res, crop_res),
        'val': DatasetRendered(dataset_path, 8000, 9000, half_res, crop_res),
        'test': DatasetRendered(dataset_path, 9000, 10000, half_res, crop_res)
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
                input, mask, gt, gt_d, offsets = sampled_batch["image"], sampled_batch["mask"], \
                                                 sampled_batch["gt"], sampled_batch["gt_d"], sampled_batch["offset"]
                offsets = offsets.cuda()
                if torch.cuda.device_count() == 1:
                    input = input.cuda()
                    mask = mask.cuda()
                    gt = gt.cuda()
                    offsets = offsets.cuda()

                # plt.imshow(input[0, 0, :, :])
                # plt.show()

                # plt.imshow(input[0, 1, :, :])
                # plt.show()

                # plt.imshow(mask[0, 0, :, :])
                # plt.show()
                # print(np.sum(np.array((input != input).cpu()).astype(np.int), dtype=np.int32))

                if phase == 'train':

                    outputs, latent = model(input)
                    optimizer.zero_grad()
                    if depth_loss:
                        loss, loss_unweighted, loss_disparity, loss_mask = \
                            depth_loss_func(outputs, mask.cuda(), alpha, gt_d.cuda(), half_res)
                        pass
                    else:
                        loss_disparity, loss_mask, loss_depth = \
                            l1_and_mask_loss(outputs, mask.cuda(), gt.cuda(), offsets,
                                    enable_masking=enable_mask, use_smooth_l1=use_smooth_l1, debug_depth=gt_d)
                        loss = loss_disparity + alpha * loss_mask
                        #loss = loss_mask

                    loss.backward()
                    # print(net.conv_dwn_0_1.bias)
                    # print(net.conv_dwn_0_1.bias.grad == 0)
                    # print(net.conv_dwn_0_1.weight.grad == 0)
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs, latent = model(input.cuda())
                        if depth_loss:
                            loss_disparity, loss_mask, loss_depth = \
                                depth_loss_func(outputs, mask.cuda(), alpha, gt_d.cuda(), half_res)
                        else:
                            loss_disparity, loss_mask, loss_depth = \
                                l1_and_mask_loss(outputs, mask.cuda(), gt.cuda(), offsets,
                                         enable_masking=enable_mask, use_smooth_l1=use_smooth_l1, debug_depth=gt_d)
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

                loss_disparity_running += loss_disparity.item() * input.size(0)
                loss_mask_running += loss_mask.item() * input.size(0)
                loss_depth_running += loss_depth.item() * input.size(0)

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
