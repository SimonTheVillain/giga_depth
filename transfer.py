import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.optim as optim
from dataset.dataset_rendered_2 import *
from dataset.dataset_captured import *
from torch.utils.data import DataLoader
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from model.discriminator import DiscriminatorSliced

from torch.utils.tensorboard import SummaryWriter


def train():
    gpu_list = [1]
    main_device = f"cuda:{gpu_list[0]}"  # torch.cuda.device(args.gpu_list[0])
    dataset_path_source = "/media/simon/ssd_data/data/datasets/structure_core_unity_3"
    dataset_path_tgt = "/media/simon/ssd_data/data/datasets/structure_core/single_shots"

    path_model = "trained_models/2stage_class_52_backbone.pt"
    path_target_model = ""
    discriminator_path = ""
    discriminator_pretrain_epochs = 10
    epochs = 20
    batch_size = 2
    num_workers = 8

    width = 640
    height = 448
    slices = 4# todo get slices from loaded model

    lr = 0.01
    weight_decay = 1e-5

    #load source model
    sourceModel = torch.load(path_model)
    sourceModel.eval()

    # clone or
    targetModel = sourceModel.clone()
    if path_target_model != "":
        targetModel = torch.load(path_target_model)
        targetModel.eval()

    #create or load disciriminator
    discriminator = DiscriminatorSliced(64, 640, 448, 4)
    if discriminator_path != "":
        discriminator = torch.load(discriminator_path)
        discriminator.eval()

    #the optimizer for the target
    optimizerD = optim.Adam(discriminator.parameters(),
                           lr=lr,
                           weight_decay=weight_decay)

    #the optimizer for the target model
    optimizerMt = optim.Adam(targetModel.parameters(),
                           lr=lr,
                           weight_decay=weight_decay)


    #todo: load dataset
    datasetsSrc = GetDataset(dataset_path_tgt, version=3)
    datasetTgt = GetDataset(dataset_path_tgt, version=3)

    dataloader = torch.utils.data.DataLoader(datasetsSrc['train'], batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)

    #loop to pretrain the discriminator
    for epoch in range(1, discriminator_pretrain_epochs):
        optimizerD.zero_grad()



    for epoch in range(1, epochs):
        #TODO: implement this!
        # https://github.com/Carl0520/ADDA-pytorch/blob/master/core/adapt.py


        optimizerD.zero_grad()
        optimizerMt.zero_grad()




if __name__ == '__main__':
    train()
