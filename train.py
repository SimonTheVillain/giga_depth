import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.optim as optim
from dataset.dataset_rendered_2 import DatasetRendered2
from model.regressor_2Stage import Regressor2Stage
from model.regressor_1Stage import Regressor1Stage
from model.backbone_6_64 import Backbone6_64
from torch.utils.data import DataLoader
import numpy as np
import os
import math

import matplotlib
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter


class CompositeModel(nn.Module):
    def __init__(self, backbone, regressor):
        super(CompositeModel, self).__init__()
        self.backbone = backbone
        self.regressor = regressor

    def forward(self, x, x_gt=None, mask_gt=None):
        x = self.backbone(x)
        return self.regressor(x, x_gt, mask_gt)

def train():
    dataset_path = "/home/simon/datasets/structure_core_unity"

    experiment_name = "bb64_2stage_simple"

    writer = SummaryWriter(f"tensorboard/{experiment_name}")

    #todo: plit loading and storing of models for Backbone and Regressor
    load_regressor = ""
    load_backbone = ""
    model_path_dst = f"trained_models/{experiment_name}.pt"

    num_epochs = 5000
    batch_size = 2
    num_workers = 8
    alpha = 0.1
    learning_rate = 0.0001# formerly it was 0.001 but alpha used to be 10 # maybe after this we could use 0.01 / 1280.0
    momentum = 0.90
    shuffle = True

    if load_regressor != "":
        regressor = torch.load(load_regressor)
        regressor.eval()
    else:
        regressor = Regressor2Stage()

    if load_backbone != "":
        backbone = torch.load(load_backbone)
        backbone.eval()
    else:
        backbone = Backbone6_64()

    model = CompositeModel(backbone, regressor)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    # for param_tensor in net.state_dict():
    #    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #the whole unity rendered dataset

    #the filtered dataset
    datasets = {
        'train': DatasetRendered2(dataset_path, 0, 8000),
        'val': DatasetRendered2(dataset_path, 8000, 9000),
        'test': DatasetRendered2(dataset_path, 9000, 10000)
    }

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                                  shuffle=shuffle, num_workers=num_workers)
                   for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}
    min_test_epoch_loss = 100000.0
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

            loss_disparity_acc = 0
            loss_class_acc = 0
            loss_disparity_acc_sub = 0
            loss_class_acc_sub = 0

            for i_batch, sampled_batch in enumerate(dataloaders[phase]):
                step = step + 1
                input, x_gt, mask_gt = sampled_batch
                #print(torch.min(x_gt)) # todo: remove this debug!!!!!
                #print(torch.max(x_gt))
                if torch.cuda.device_count() == 1:
                    input = input.cuda()
                    mask_gt = mask_gt.cuda()
                    x_gt = x_gt.cuda()

                if phase == 'train':
                    torch.autograd.set_detect_anomaly(True)
                    x, mask, class_losses = model(input, x_gt, mask_gt)
                    optimizer.zero_grad()
                    loss = torch.mean(torch.abs(x-x_gt) * mask_gt)
                    loss_disparity_acc += loss.item()
                    loss_disparity_acc_sub += loss.item()
                    loss = loss * alpha
                    for class_loss in class_losses:
                        loss += class_loss
                        loss_class_acc += class_loss.item()
                        loss_class_acc_sub += class_loss.item()

                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        x, mask = model(input)
                        loss = torch.mean(torch.abs(x-x_gt) * mask_gt)
                        loss_disparity_acc += loss.item()
                        loss_disparity_acc_sub += loss.item()


                # print("FUCK YEAH")
                if i_batch % 100 == 99:
                    print("batch {} loss {}".format(i_batch, (loss_disparity_acc_sub + loss_class_acc_sub) / 100))
                    writer.add_scalar(f'{phase}_subepoch/disparity(loss)',
                                      loss_disparity_acc_sub / 100.0 * 1024, step)
                    if loss_class_acc != 0:
                        writer.add_scalar(f'{phase}_subepoch/class_loss', loss_class_acc_sub / 100, step)
                    loss_disparity_acc_sub = 0
                    loss_class_acc_sub = 0

            epoch_loss = loss_disparity_acc / dataset_sizes[phase] * alpha
            writer.add_scalar(f"{phase}/disparity(loss)",
                              loss_disparity_acc / dataset_sizes[phase] * 1024, step)
            if loss_class_acc_sub != 0:
                epoch_loss += loss_class_acc / dataset_sizes[phase]
                writer.add_scalar(f"{phase}/class_loss", loss_class_acc / dataset_sizes[phase], step)


            print(f"{phase} Loss: {epoch_loss}")



            # store at the end of a epoch
            if phase == 'val' and not math.isnan(loss_disparity_acc) and not math.isinf(loss_disparity_acc):
                if isinstance(model, nn.DataParallel):
                    module = model.module
                else:
                    module = model

                print("storing network")
                torch.save(module.backbone, f"trained_models/{experiment_name}_backbone_chk.pt")
                torch.save(module.regressor, f"trained_models/{experiment_name}_regressor_chk.pt")

                if epoch_loss < min_test_epoch_loss:
                    print("storing network")
                    min_test_epoch_loss = epoch_loss
                    torch.save(module.backbone, f"trained_models/{experiment_name}_backbone.pt")#maybe use type(x).__name__()
                    torch.save(module.regressor, f"trained_models/{experiment_name}_regressor.pt")

    writer.close()


if __name__ == '__main__':
    train()
