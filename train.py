import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.optim as optim
from dataset.dataset_rendered_2 import DatasetRendered2
from model.regressor_2Stage import Regressor2Stage
from model.regressor_1Stage import Regressor1Stage
from model.regressor_1branch import Regressor1Branch
from model.regressor_branchless import RegressorBranchless
from model.backbone_6_64 import Backbone6_64
from model.backbone import Backbone
from torch.utils.data import DataLoader
import math
import argparse
import os

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
    parser = argparse.ArgumentParser()
    #parser.add_argument("-V", "--version", help="show program version", action="store_true")
    parser.add_argument("-d", "--dataset_path", dest="path", help="Path to the dataset.", action="store",
                        default=os.path.expanduser("~/datasets/structure_core_unity"))

    args = parser.parse_args()

    experiment_name = "single_slice_branched_sgd"

    writer = SummaryWriter(f"tensorboard/{experiment_name}")

    #slit loading and storing of models for Backbone and Regressor
    load_regressor = "trained_models/single_slice_branched_sgd_regressor_chk.pt"
    load_backbone = "trained_models/single_slice_branched_sgd_backbone_chk.pt"

    # not loading any pretrained part of any model whatsoever
    #load_regressor = ""
    #load_backbone = ""

    num_epochs = 5000
    #todo: either do only 100 lines at a time, or go for
    tgt_res = (1216, 896)
    is_npy = True
    slice_in = (100, 100 + 17*2+1)
    slice_gt = (50+8, 50+8+1)
    batch_size = 256
    num_workers = 8
    alpha = 0.01# usually this is 0.1
    learning_rate = 0.1 #0.001 for the branchless regressor (smaller since we feel it's not entirely stable)
    momentum = 0.90
    shuffle = True

    if load_regressor != "":
        regressor = torch.load(load_regressor)
        regressor.eval()
    else:
        regressor = RegressorBranchless(height=1)
        #regressor = Regressor2Stage()
        regressor = Regressor1Stage(height=1)
        #regressor = Regressor1Branch(height=1)

    if load_backbone != "":
        backbone = torch.load(load_backbone)
        backbone.eval()
        #fix parameters in the backbone (or maybe not!)
        #for param in backbone.parameters():
        #    param.requires_grad = False
    else:
        backbone = Backbone()

    model = CompositeModel(backbone, regressor)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    # for param_tensor in net.state_dict():
    #    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #the whole unity rendered dataset

    #the filtered dataset
    datasets = {
        'train': DatasetRendered2(args.path, 0, 40000, tgt_res=tgt_res, is_npy=True),
        'val': DatasetRendered2(args.path, 40000, 41000, tgt_res=tgt_res, is_npy=True),
        'test': DatasetRendered2(args.path, 41000, 42000, tgt_res=tgt_res, is_npy=True)
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

            #TODO: accumulate classification losses for each classification stage!!!
            loss_disparity_acc = 0
            loss_class_acc = []
            loss_disparity_acc_sub = 0
            loss_class_acc_sub = []

            for i_batch, sampled_batch in enumerate(dataloaders[phase]):
                step = step + 1
                input, x_gt, mask_gt = sampled_batch
                if not is_npy:
                    input = input[:, :, slice_in[0]:slice_in[1], :]
                    x_gt = x_gt[:, :, slice_gt[0]:slice_gt[1], :]
                    mask_gt = mask_gt[:, :, slice_gt[0]:slice_gt[1], :]
                #print(torch.min(x_gt)) # todo: remove this debug!!!!!
                #print(torch.max(x_gt))
                if torch.cuda.device_count() >= 1:
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
                    if len(loss_class_acc) == 0:
                        loss_class_acc = [0] * len(class_losses)
                        loss_class_acc_sub = [0] * len(class_losses)
                    for i, class_loss in enumerate(class_losses):
                        loss += torch.mean(class_loss)
                        loss_class_acc[i] += torch.mean(class_loss).item()
                        loss_class_acc_sub[i] += torch.mean(class_loss).item()

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
                    print("batch {} loss {}".format(i_batch, (loss_disparity_acc_sub * alpha + sum(loss_class_acc_sub)) / 100))
                    writer.add_scalar(f'{phase}_subepoch/disparity(loss)',
                                      loss_disparity_acc_sub / 100.0 * 1024, step)
                    for i, class_loss in enumerate(loss_class_acc_sub):
                        writer.add_scalar(f'{phase}_subepoch/class_loss_{i}', class_loss / 100, step)
                        loss_class_acc_sub[i] = 0

                    loss_disparity_acc_sub = 0

            epoch_loss = loss_disparity_acc / dataset_sizes[phase] * batch_size * alpha
            writer.add_scalar(f"{phase}/disparity(loss)",
                              loss_disparity_acc / dataset_sizes[phase] * batch_size * 1024, step)
            for i, class_loss in enumerate(loss_class_acc):
                epoch_loss += class_loss / dataset_sizes[phase] * batch_size
                writer.add_scalar(f"{phase}/class_loss{i}", class_loss / dataset_sizes[phase] * batch_size, step)


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
