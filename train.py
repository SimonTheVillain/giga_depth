import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.optim as optim
from dataset.dataset_rendered import DatasetRendered
from model.model_1 import Model1
from torch.utils.data import DataLoader
import numpy as np

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


def train():
    dataset_path = "/media/simon/ssd_data/data/reduced_0_08_2"
    dataset_path = "D:/dataset_filtered"
    writer = SummaryWriter('tensorboard/experiment13')

    model_path_src = "trained_models/model_1_3.pt"
    load_model = True
    model_path_dst = "trained_models/model_1_4.pt"
    crop_div = 2
    crop_res = (896/crop_div, 1216/crop_div)
    store_checkpoints = True
    num_epochs = 500
    batch_size = 2# 6
    num_workers = 4# 8
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

    #the whole unity rendered dataset

    #the filtered dataset
    datasets = {
        'train': DatasetRendered(dataset_path, 0, 12000, half_res, crop_res),
        'val': DatasetRendered(dataset_path, 12000, 13000, half_res, crop_res),
        'test': DatasetRendered(dataset_path, 13000, 14000, half_res, crop_res)
    }
    datasets = {
        'train': DatasetRendered(dataset_path, 0, 10000, half_res, crop_res),
        'val': DatasetRendered(dataset_path, 10000, 11000, half_res, crop_res),
        'test': DatasetRendered(dataset_path, 11000, 12000, half_res, crop_res)
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
                input, mask, gt = sampled_batch["image"], sampled_batch["mask"], sampled_batch["gt"]
                if torch.cuda.device_count() == 1:
                    input = input.cuda()
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

                    outputs, latent = model(input)
                    optimizer.zero_grad()
                    loss, loss_unweighted, loss_disparity, loss_mask = \
                        sqr_loss(outputs, mask.cuda(), gt.cuda(),
                                 alpha=alpha, enable_mask=enable_mask, use_smooth_l1=use_smooth_l1)
                    loss.backward()
                    # print(net.conv_dwn_0_1.bias)
                    # print(net.conv_dwn_0_1.bias.grad == 0)
                    # print(net.conv_dwn_0_1.weight.grad == 0)
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs, latent = model(input.cuda())
                        loss, loss_unweighted, loss_disparity, loss_mask = \
                            sqr_loss(outputs, mask.cuda(), gt.cuda(),
                                     alpha=alpha, enable_mask=enable_mask, use_smooth_l1=use_smooth_l1)

                if loss_unweighted > 100:
                    print("{} loss on batch {}".format(loss_unweighted, i_batch))

                writer.add_scalar('loss_disparity/' + phase, loss_disparity, step)
                writer.add_scalar('loss_mask/' + phase, loss_mask, step)


                subbatch_loss += loss_unweighted
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

                running_loss += loss_unweighted * input.size(0)  # loss.item() * input.size(0)

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
