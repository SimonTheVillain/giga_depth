import torch
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast
import numpy
from torch.utils.tensorboard import SummaryWriter
from model.uNet import UNet
from dataset.dataset_captured_ambient import DatasetCapturedAmbient
import cv2
from common.common import LCN_tensors
import math
import torch.nn as nn
import torch.nn.functional as F


class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def get_gray(self,x):
        '''
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        # x_list = []
        # for i in range(x.shape[1]):
        #     x_i = x[:, i]
        #     x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
        #     x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
        #     x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
        #     x_list.append(x_i)

        # x = torch.cat(x_list, dim=1)
        if x.shape[1] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x

class GradLoss(nn.Module):

    def __init__(self):
        super(GradLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.grad_layer = GradLayer()

    def forward(self, output, gt_img):
        output_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(gt_img)
        return self.loss(output_grad, gt_grad)

def train():
    experiment_name = "amb_estimator2"
    optimizer = "adam"
    momentum = 0.9
    weight_decay = 1e-5
    learning_rate = 1e-4
    half_precision = False
    shuffle = True
    num_workers = 8
    batch_size = 2
    epochs = 10
    dataset_path = "/home/simon/datasets/structure_core/sequences_combined_ambient"
    weight_LCN_loss = 1.0
    weight_sobel = 1.0
    writer = SummaryWriter(f"tensorboard/{experiment_name}")

    model = UNet(1, 1, channel_size_scale=0.25)
    model.cuda()
    loss_sobel = GradLoss().cuda()
    model = torch.load("trained_models/amb_estimator2.pt")
    model.eval()
    model.cuda()

    # for param_tensor in net.state_dict():
    #    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate,
                              momentum=momentum,
                              weight_decay=weight_decay)
    else:
        if optimizer == "adam":
            optimizer = optim.Adam(model.parameters(),
                                   lr=learning_rate,
                                   weight_decay=weight_decay)
        else:
            print("Optimizer argument must either be sgd or adam!")
            return

    if half_precision:
        scaler = GradScaler()

    datasets = {'train': DatasetCapturedAmbient(dataset_path, 'train'),
                'val': DatasetCapturedAmbient(dataset_path, 'val')}

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                                  shuffle=shuffle, num_workers=num_workers)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    best_loss = 100000.0
    step = -1
    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs}')
        print('-' * 10)
        for phase in ['train', 'val']:
            print(phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            model.zero_grad()
            loss_acc = 0
            loss_acc_sub = 0
            for i_batch, sampled_batch in enumerate(dataloaders[phase]):

                step = step + 1
                ir, ambient_gt = sampled_batch
                ir = ir.cuda()
                ambient_gt = ambient_gt.cuda()

                if False:
                    cv2.imshow("ir", ir[0, 0, :, :].detach().cpu().numpy())
                    cv2.imshow("ambient_gt", ambient_gt[0, 0, :, :].detach().cpu().numpy())
                    cv2.waitKey()

                if phase == 'train':
                    optimizer.zero_grad()
                    torch.autograd.set_detect_anomaly(True)
                    if half_precision:
                        with autocast():
                            ambient_est = model(ir)

                    else:
                        ambient_est = model(ir)
                        delta = torch.abs(ambient_est - ambient_gt).mean()
                    loss = delta.mean()
                    loss += (delta*delta).mean()
                    # lcn1 = LCN_tensors(ambient_gt)[0]
                    # cv2.imshow("lcn_gt", lcn1[0,0,:,:].detach().cpu().numpy()*0.1+0.5)
                    # cv2.waitKey()
                    loss += weight_sobel * loss_sobel(ambient_est, ambient_gt)
                    loss += weight_LCN_loss * torch.abs(LCN_tensors(ambient_est)[0] - LCN_tensors(ambient_gt)[0]).mean()

                    if False:
                        cv2.imshow("ir", ir[0, 0, :, :].detach().cpu().numpy())
                        cv2.imshow("ambient_gt", ambient_gt[0, 0, :, :].detach().cpu().numpy())
                        cv2.imshow("ambient_est", ambient_est[0, 0, :, :].type(torch.float32).detach().cpu().numpy())
                        cv2.waitKey()

                    if half_precision:
                        scaler.scale(loss).backward()

                        scaler.step(optimizer)
                        scaler.update()

                    else:
                        loss.backward()
                        optimizer.step()

                else:  # val
                    with torch.no_grad():
                        if half_precision:
                            with autocast():
                                ambient_est = model(ir)
                        else:
                            ambient_est = model(ir)

                    delta = torch.abs(ambient_est - ambient_gt).mean()
                    loss = delta.mean()
                    loss += (delta*delta).mean()
                    loss += weight_sobel * loss_sobel(ambient_est, ambient_gt)
                    loss += weight_LCN_loss * torch.abs(
                        LCN_tensors(ambient_est.type(torch.float32))[0] - LCN_tensors(ambient_gt)[0]).mean()

                loss_acc += loss.item()
                loss_acc_sub += loss.item()

                # print progress every 100 steps!
                if i_batch % 100 == 0:
                    cv2.imwrite(f"tmp/{phase}_input.png",
                                ir[0, 0, :, :].type(torch.float32).detach().cpu().numpy() * 255)
                    cv2.imwrite(f"tmp/{phase}_ambient_gt.png",
                                ambient_gt[0, 0, :, :].type(torch.float32).detach().cpu().numpy() * 255)
                    cv2.imwrite(f"tmp/{phase}_ambient_est.png",
                                ambient_est[0, 0, :, :].type(torch.float32).detach().cpu().numpy() * 255)

                if i_batch % 100 == 99:
                    writer.add_scalar(f'{phase}_subepoch/loss',
                                      loss_acc_sub / 100.0, step)
                    print(f"loss at iteration/batch {i_batch}: {loss_acc_sub / 100.0}")
                    loss_acc_sub = 0

        loss_acc = loss_acc / dataset_sizes[phase]
        print(f"loss of epoch {epoch} ({phase}): {loss_acc}")

        writer.add_scalar(f'{phase}/loss', loss_acc, step)
        if phase == 'val':
            # store at the end of a validation phase of this epoch!
            if not math.isnan(best_loss) and not math.isinf(loss_acc):
                print("storing network")
                torch.save(model, f"trained_models/{experiment_name}_chk.pt")
                if loss_acc < best_loss:
                    print("storing network (best)")
                    best_loss = loss_acc
                    torch.save(model, f"trained_models/{experiment_name}.pt")

    writer.close()


if __name__ == '__main__':
    train()
