import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
#from torch.cuda.amp.autocast import autocast
from dataset.dataset_rendered import DatasetRendered

from model.regressor_v1 import Regressor1
from model.regressor_v2 import Regressor2
from model.regressor_v3 import Regressor3
from model.backbone_v1 import Backbone1
from model.backbone_v2 import Backbone2
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from os.path import expanduser

import matplotlib
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

import gc
def print_cuda_data():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print("{},{}".format(type(obj), obj.size()))
        except:
            pass

#if not torch.cuda.is_available():
#    raise Exception("No GPU found, please run without --cuda")

class CompositeModel(nn.Module):
    def __init__(self, backbone, regressor):
        super(CompositeModel, self).__init__()
        self.backbone = backbone
        self.regressor = regressor

    def forward(self, x, x_gt=None):
        x = self.backbone(x)
        x, mask, class_loss = self.regressor(x, x_gt)
        return x, mask, class_loss

class CompositeLossModel(nn.Module):
    def __init__(self, backbone, regressor, mask_loss=True):
        super(CompositeLossModel, self).__init__()
        self.backbone = backbone
        self.regressor = regressor
        self.mask_loss = mask_loss

    def forward(self, x, x_gt=None, mask_gt=None, use_gt_classes=True):
        x = self.backbone(x)
        if x_gt == None or mask_gt == None:
            x, mask, _ = self.regressor(x)
            return x, mask
        else:
            if use_gt_classes:
                x, mask, loss_class = self.regressor(x, x_gt)
            else:
                x, mask, loss_class = self.regressor(x, None)
            loss_mask = torch.abs(mask - mask_gt)
            loss_reg = torch.abs(x - x_gt)
            if self.mask_loss:
                if use_gt_classes:
                    loss_class = torch.mean(loss_class * mask_gt)
                else:
                    loss_class = None
                #loss_class = loss_class * mask_gt
                loss_reg = loss_reg * mask_gt

            #print(torch.max(mask))
            #print(torch.max(mask_gt))
            #print(torch.max(loss_mask))

            #print(torch.min(mask))
            #print(torch.min(mask_gt))
            #print(torch.min(loss_mask))
            loss_mask = torch.mean(loss_mask)
            loss_reg = torch.mean(loss_reg)
            #loss_class = torch.mean(loss_class)
            return loss_class, loss_reg, loss_mask


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




def train():
    path = expanduser("~/giga_depth_results/")
    path = "/workspace/giga_depth_results/"

    half_precision = False
    single_gpu = False
    dataset_path = "/media/simon/ssd_data/data/dataset_reduced_0_08"
    dataset_path = "/home/simon/datasets/dataset_filtered"
    dataset_path = "/workspace/datasets/dataset_filtered"

    if os.name == 'nt':
        dataset_path = "D:/dataset_filtered"

    if torch.cuda.device_count() == 1:
        # Mainly this means we are working on my laptop and not in a docker
        path = expanduser("~/giga_depth_results/")
    writer = SummaryWriter(path + 'tensorboard/b2r1')
    #writer = SummaryWriter('tensorboard/dump')

    model_path_src = path + "trained_models/b2r3.pt"
    load_model = True
    model_path_dst = path + "trained_models/b2r1.pt"
    model_path_unconditional = path + "trained_models/b2r1_chckpt.pt"
    unconditional_chckpts = True
    crop_res = (896, 1216)# - 432)
    shuffle = False
    half_res = True
    dataset_noise = 0.2
    vert_pix_jitter = 2
    #crop_res = (100, 1216)
    store_checkpoints = True
    num_epochs = 5000
    batch_size = 6
    num_workers = 16# 8
    show_images = False
    mask_loss = False
    alpha_mask = 0.1
    alpha_regression = 100 # 10 is a good value to improve subpixel accuracy only 1 is tested to train classification
    alpha_regression = 1#TODO: go back to a weight of 100 for the regression part
    learning_rate = 0.1 #0.1 for the coarse part
    momentum = 0.90
    projector_width = 1280
    batch_accumulation = 1
    class_count = 128
    single_slice = False
    log_steps = False # log every step with tensorboard... is overkill in the most cases
    if torch.cuda.device_count() == 1:
        # Mainly this means we are working on my laptop and not in a docker
        path = expanduser("~/giga_depth_results/")
        dataset_path = "/home/simon/datasets/dataset_filtered"
        batch_size = 2
        model_path_src = path + "trained_models/b2r3.pt"
        load_model = False
        model_path_dst = path + "trained_models/b2r1.pt"
        model_path_unconditional = path + "trained_models/b2r1_chckpt.pt"


    core_image_height = crop_res[0]
    if single_slice:
        dataset_path = "/media/simon/ssd_data/data/dataset_filtered_slice_142"
        slices = 1
        padding = 0 #Model_CR10_3_hsn.padding()
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
        #model = Model_CR10_3_hsn(class_count, core_image_height)
        #model.copy_backbone(model_old)
        #model_old = None

        backbone = Backbone2()
        regressor = Regressor1(128, int(crop_res[0]/2), int(crop_res[1]/2))

        model_old = model
        model = CompositeLossModel(backbone, regressor)
        #copy over some of the pretrained bits
        model.backbone = model_old.backbone
        model.regressor.conv_c = model_old.regressor.conv_c
    else:
        backbone = Backbone2()
        regressor = Regressor1(128, int(crop_res[0]/2), int(crop_res[1]/2))

        model = CompositeLossModel(backbone, regressor)

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
    #cuda_devices = ["cuda:{}".format(i) for i in range(torch.cuda.device_count())]
    #no cpu allowed in dataparallel !?
    #model = nn.DataParallel(model, device_ids=cuda_devices, output_device=torch.device("cpu"))

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
        'train': DatasetRendered(dataset_path, 0, 8000, half_res, crop_res, dataset_noise, vert_pix_jitter),
        'val': DatasetRendered(dataset_path, 8000, 9000, half_res, crop_res, dataset_noise, vert_pix_jitter),
        'test': DatasetRendered(dataset_path, 9000, 10000, half_res, crop_res, dataset_noise, vert_pix_jitter)
    }
    if single_slice:

        datasets = {
            'train': DatasetRendered(dataset_path, 0, 8000, half_res, crop_res, dataset_noise, vert_pix_jitter,
                                     single_slice, crop_top, crop_bottom),
            'val': DatasetRendered(dataset_path, 8000, 9000, half_res, crop_res, dataset_noise, vert_pix_jitter,
                                   single_slice, crop_top, crop_bottom),
            'test': DatasetRendered(dataset_path, 9000, 10000, half_res, crop_res, dataset_noise, vert_pix_jitter,
                                    single_slice, crop_top, crop_bottom)
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
            loss_reg_relative_running = 0.0
            loss_reg_absolute_running = 0.0
            loss_mask_running = 0.0
            loss_running = 0.0

            loss_class_subepoch = 0.0
            loss_reg_relative_subepoch = 0.0
            loss_reg_absolute_subepoch = 0.0
            loss_mask_subepoch = 0.0
            loss_subepoch = 0.0
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

                #offsets = offsets.cuda()
                #mask_gt = mask_gt.cuda()
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

                # plt.imshow(image_r[0, 0, :, :].cpu().detach())
                # plt.show()

                # plt.imshow(input[0, 1, :, :])
                # plt.show()

                # plt.imshow(mask[0, 0, :, :])
                # plt.show()
                # print(np.sum(np.array((input != input).cpu()).astype(np.int), dtype=np.int32))

                if phase == 'train':
                    #print(image_r.device)
                    #print(gt.device)
                    loss_class, loss_reg_relative, loss_mask = model(image_r, gt, mask_gt)

                    #print(loss_class.shape)
                    #print(loss_mask.shape)
                    #print(loss_reg_relative.shape)
                    loss_class = torch.mean(loss_class)
                    loss_mask = torch.mean(loss_mask)
                    loss_reg_relative = torch.mean(loss_reg_relative)

                    #print_cuda_data()
                    #for half precision we probably need autocast
                    #with torch.cuda.amp.autocast():
                    if (i_batch - 1) % batch_accumulation == 0 or half_precision:
                        optimizer.zero_grad()


                    loss = loss_class + alpha_regression * loss_reg_relative + alpha_mask * loss_mask

                    if half_precision:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        #print("going backward")
                        loss.backward()
                        #print_cuda_data()
                        optimizer.step()

                else: # evaluation
                    with torch.no_grad():
                        loss_class, loss_reg_relative, loss_mask = model(image_r, gt, mask_gt)
                        _, loss_reg_absolute, _ = model(image_r, gt, mask_gt, False)# the same without using the GT classes


                        loss_class = torch.mean(loss_class)
                        loss_mask = torch.mean(loss_mask)
                        loss_reg_relative = torch.mean(loss_reg_relative)
                        loss_reg_absolute = torch.mean(loss_reg_absolute)

                        loss = loss_class + alpha_regression * loss_reg_relative + alpha_mask * loss_mask

                        if log_steps:
                            writer.add_scalar('batch_{}/loss_reg_absolute'.format(phase), loss_reg_absolute.item() * projector_width, step)
                        loss_reg_absolute_subepoch += loss_reg_absolute.item()

                if log_steps:
                    writer.add_scalar('batch_{}/loss_combined'.format(phase), loss.item(), step)
                    writer.add_scalar('batch_{}/loss_class'.format(phase), loss_class.item(), step)
                    writer.add_scalar('batch_{}/loss_mask'.format(phase), loss_mask.item(), step)
                    writer.add_scalar('batch_{}/loss_reg_relative'.format(phase), loss_reg_relative.item() * projector_width, step)
                loss_reg_relative_subepoch += loss_reg_relative.item()

                if False:
                    print("combined = {}, class = {}, regression = {}, mask = {}".format(loss.item(),
                                                                                         loss_class.item(),
                                                                                         loss_reg.item(),
                                                                                         loss_mask.item()))
                loss_class_subepoch += loss_class.item()
                loss_mask_subepoch += loss_mask.item()
                loss_reg_relative_subepoch += loss_reg_relative.item()
                loss_subepoch += loss.item()

                if show_images:
                    fig = plt.figure()
                    fig.add_subplot(2, 1, 1)
                    plt.imshow(image_r[0, 0, :, :].cpu().detach().numpy(), vmin=0, vmax=1)
                    plot.show()

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
                    print("batch {} loss {}".format(i_batch, loss_subepoch / 100))

                    writer.add_scalar('subepoch_{}/loss_class'.format(phase),
                                      loss_class_subepoch / 100.0, step)

                    writer.add_scalar('subepoch_{}/loss_reg_relative'.format(phase),
                                          loss_reg_relative_subepoch / 100.0 * projector_width, step)
                    if phase == "eval":
                        writer.add_scalar('subepoch_{}/loss_reg_absolute'.format(phase),
                                          loss_reg_absolute_subepoch / 100.0 * projector_width, step)
                    writer.add_scalar('subepoch_{}/loss_mask'.format(phase), loss_mask_subepoch / 100.0, step)
                    writer.add_scalar('subepoch_{}/loss_combined'.format(phase), loss_subepoch / 100.0, step)
                    loss_class_subepoch = 0.0
                    loss_reg_relative_subepoch = 0.0
                    loss_reg_absolute_subepoch = 0.0
                    loss_mask_subepoch = 0.0
                    loss_subepoch = 0.0


                loss_running += loss.item() * batch_size
                loss_class_running += loss_class.item() * batch_size
                loss_mask_running += loss_mask.item() * batch_size
                loss_running += loss.item() * batch_size
                loss_reg_relative_running += loss_reg_relative.item() * batch_size
                if phase == "val":
                    loss_reg_absolute_running += loss_reg_absolute.item() * batch_size

            writer.add_scalar('epoch_{}/loss_class'.format(phase), loss_class_running / dataset_sizes[phase], epoch)
            writer.add_scalar('epoch_{}/loss_reg_relative'.format(phase),
                              loss_reg_relative_running / dataset_sizes[phase] * projector_width, epoch)
            if phase == "val":
                writer.add_scalar('epoch_{}/loss_reg_absolute'.format(phase),
                                loss_reg_absolute_running / dataset_sizes[phase] * projector_width, epoch)

            writer.add_scalar('epoch_{}/loss_mask'.format(phase), loss_mask_running / dataset_sizes[phase], epoch)

            writer.add_scalar('epoch_{}/loss_combined'.format(phase),
                              loss_running / dataset_sizes[phase], epoch)
            print('{} Loss: {:.4f}'.format(phase, loss_running / dataset_sizes[phase]))
            # print(net.conv_dwn_0_1.weight.grad == 0)
            # store at the end of a epoch
            if phase == 'val' and unconditional_chckpts:
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module, model_path_unconditional)
                else:
                    torch.save(model, model_path_unconditional)
            if phase == 'val' and unconditional_chckpts:
                if loss_running < min_test_epoch_loss:
                    print("storing network")
                    min_test_epoch_loss = loss_running
                    if isinstance(model, nn.DataParallel):
                        torch.save(model.module, model_path_dst)
                    else:
                        torch.save(model, model_path_dst)

    writer.close()


if __name__ == '__main__':
    train()
