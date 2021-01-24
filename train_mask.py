import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from experiments.lines.model_lines_CR8_n import *
from torch.utils.data import DataLoader
import math
import argparse

from torch.utils.tensorboard import SummaryWriter


def sigma_loss(sigma, x, x_gt): #sigma_sq is also called "variance"
    if torch.any(torch.isnan(x)):
        print("x: found nan")
    if torch.any(torch.isinf(x)):
        print("x: found inf")
    if torch.any(torch.isnan(x_gt)):
        print("x_gt: found nan")
    if torch.any(torch.isinf(x_gt)):
        print("x_gt: found inf")
    delta = torch.abs(x - x_gt)
    if torch.any(torch.isnan(delta)):
        print("delta: found nan")
    if torch.any(torch.isinf(delta)):
        print("delta: found inf")
    if torch.any(torch.isnan(sigma)):
        print("sigma: found nan")
    if torch.any(torch.isinf(sigma)):
        print("sigma: found inf")
    eps = 0.01 # sqrt(0.00001) = 0.0003 ... that means we actually have approx 0.3 pixel of basic offset for sigma
    term1 = torch.div(torch.square((x - x_gt)*1024.0), sigma*sigma + eps)
    term2 = torch.log(sigma * sigma + eps)
    # term 3 is to stay positive(after all it is sigma^2)
    term3 = 0#F.relu(-sigma)
    loss = term1 + term2 + term3
    #print("term1")
    #print(term1)
    #print("term2")
    #print(term2)
    #print("max term1")
    #print(torch.max(term1))
    #print("max term2")
    #print(torch.max(term2))
    #print("max sigma:")
    #print(torch.max(torch.abs(sigma_sq)))
    #print("term 1 den:")
    #print((2*torch.min(torch.abs(sigma_sq)) + eps))
    if torch.any(torch.isnan(term1)):
        print("term1: found nan")
    if torch.any(torch.isinf(term1)):
        print("term1: found inf")

    if torch.any(torch.isnan(term2)):
        print("term2: found nan")
    if torch.any(torch.isinf(term2)):
        print("term2: found inf")
    return torch.mean(loss) #torch.tensor(0)#


def train():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-V", "--version", help="show program version", action="store_true")
    parser.add_argument("-d", "--dataset_path", dest="path", help="Path to the dataset.", action="store",
                        default=os.path.expanduser("~/datasets/structure_core_unity"))
    parser.add_argument("-r", "--dataset_path_results", dest="path_results",
                        help="Path to the results of a network running trough a dataset.",
                        action="store",
                        default="")

    args = parser.parse_args()

    experiment_name = "sigma_mask_scaled"

    writer = SummaryWriter(f"tensorboard/{experiment_name}")

    # slit loading and storing of models for Backbone and Regressor
    load_model = "trained_models/sigma_mask.pt"

    # not loading any pretrained part of any model whatsoever
    load_model = ""

    num_epochs = 5000
    # todo: either do only 100 lines at a time, or go for
    tgt_res = (1216, 896)
    is_npy = True
    slice_in = (100, 100 + 17 * 2 + 1)
    slice_gt = (50 + 8, 50 + 8 + 1)
    batch_size = 32
    num_workers = 8
    alpha = 1.0
    alpha_sigma = 1e-10  # how much weight do we give correct confidence measures
    learning_rate = 0.2  # 0.001 for the branchless regressor (smaller since we feel it's not entirely stable)
    momentum = 0.90
    shuffle = True

    if load_model != "":
        model = torch.load(load_model)
        model.eval()
    else:
        model = CR8_mask_var()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    # for param_tensor in net.state_dict():
    #    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # the whole unity rendered dataset

    # the filtered dataset
    datasets = {
        'train': DatasetRendered2(args.path, 0, 40000, tgt_res=tgt_res, is_npy=is_npy, result_dir=args.path_results),
        'val': DatasetRendered2(args.path, 40000, 41000, tgt_res=tgt_res, is_npy=is_npy, result_dir=args.path_results),
        'test': DatasetRendered2(args.path, 41000, 42000, tgt_res=tgt_res, is_npy=is_npy, result_dir=args.path_results)
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

            # TODO: accumulate classification losses for each classification stage!!!
            loss_mask_acc = 0
            loss_sigma_acc = 0
            loss_mask_acc_sub = 0
            loss_sigma_acc_sub = 0

            for i_batch, sampled_batch in enumerate(dataloaders[phase]):
                step = step + 1
                ir, x_gt, mask_gt, x_real = sampled_batch
                if not is_npy:
                    ir = ir[:, :, slice_in[0]:slice_in[1], :]
                    x_gt = x_gt[:, :, slice_gt[0]:slice_gt[1], :]
                    mask_gt = mask_gt[:, :, slice_gt[0]:slice_gt[1], :]
                # print(torch.min(x_gt)) # todo: remove this debug!!!!!
                # print(torch.max(x_gt))
                if torch.cuda.device_count() >= 1:
                    ir = ir.cuda()
                    mask_gt = mask_gt.cuda()
                    x_gt = x_gt.cuda()
                    x_real = x_real.cuda()

                #overwrite the mask_gt!
                mask_gt = (torch.abs(x_real-x_gt) < 20/1024).type(torch.float32)

                if phase == 'train':
                    torch.autograd.set_detect_anomaly(True)
                    mask, sigma = model(ir)
                    x_real = x_real.detach()

                    optimizer.zero_grad()
                    loss = torch.mean(torch.abs(mask - mask_gt))
                    loss_mask_acc += loss.item()
                    loss_mask_acc_sub += loss.item()

                    loss_sigma = sigma_loss(sigma, x_real, x_gt)
                    loss_sigma_acc += loss_sigma.item()
                    loss_sigma_acc_sub += loss_sigma.item()
                    loss = loss * alpha
                    #print(loss.item())
                    loss += loss_sigma * alpha_sigma


                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        mask, sigma = model(ir)
                        loss = torch.mean(torch.abs(mask - mask_gt))
                        loss_mask_acc += loss.item()
                        loss_mask_acc_sub += loss.item()

                        loss_sigma = sigma_loss(sigma, x_real, x_gt)
                        loss_sigma_acc += loss_sigma.item()
                        loss_sigma_acc_sub += loss_sigma.item()

                if i_batch % 100 == 99:
                    combo_loss = loss_mask_acc_sub * alpha + \
                                 loss_sigma_acc_sub * alpha_sigma
                    print("batch {} loss {}".format(i_batch, combo_loss / 100))
                    writer.add_scalar(f'{phase}_subepoch/mask(loss)',
                                      loss_mask_acc_sub / 100.0, step)

                    writer.add_scalar(f'{phase}_subepoch/sigma(loss)',
                                      loss_sigma_acc_sub / 100.0, step)

                    loss_mask_acc_sub = 0
                    loss_sigma_acc_sub = 0

            epoch_loss = loss_mask_acc / dataset_sizes[phase] * batch_size * alpha
            writer.add_scalar(f"{phase}/mask(loss)",
                              loss_mask_acc / dataset_sizes[phase] * batch_size * 1024, step)

            writer.add_scalar(f"{phase}/sigma(loss)",
                              loss_sigma_acc / dataset_sizes[phase] * batch_size, step)

            print(f"{phase} Loss: {epoch_loss}")

            # store at the end of a epoch
            if phase == 'val' and not math.isnan(loss_mask_acc) and not math.isinf(loss_sigma_acc):
                if isinstance(model, nn.DataParallel):
                    module = model.module
                else:
                    module = model

                print("storing network")
                torch.save(module, f"trained_models/{experiment_name}_chk.pt")

                if epoch_loss < min_test_epoch_loss:
                    print("storing network")
                    min_test_epoch_loss = epoch_loss
                    torch.save(module, f"trained_models/{experiment_name}.pt")

    writer.close()


if __name__ == '__main__':
    train()
