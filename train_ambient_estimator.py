import torch
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
import numpy
from torch.utils.tensorboard import SummaryWriter
from model.uNet import UNet
from dataset.dataset_captured_ambient import DatasetCapturedAmbient
import cv2
from common.common import LCN_tensors
import math

def train():
    experiment_name = "amb_estimator1"
    optimizer = "adam"
    momentum = 0.9
    weight_decay = 1e-5
    learning_rate = 1e-4
    half_precision = False
    shuffle = True
    num_workers = 8
    batch_size = 1
    epochs = 100
    dataset_path = "/home/simon/datasets/structure_core/sequences_combined_ambient"

    writer = SummaryWriter(f"tensorboard/{experiment_name}")

    model = UNet(1, 1, channel_size_scale=0.25)
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
                    cv2.imshow("ir", ir[0,0,:,:].detach().cpu().numpy())
                    cv2.imshow("ambient_gt", ambient_gt[0,0,:,:].detach().cpu().numpy())
                    cv2.waitKey()


                if phase == 'train':
                    torch.autograd.set_detect_anomaly(True)
                    if half_precision:
                        with autocast():
                            ambient_est = model(ir)

                    else:
                        ambient_est = model(ir)

                    if False:
                        cv2.imshow("x_gt", x_gt[0, 0, :, :].detach().cpu().numpy())
                        cv2.imshow("x_out", x_real[0, 0, :, :].detach().cpu().numpy())
                        cv2.imshow("mask_gt", mask_gt[0, 0, :, :].detach().cpu().numpy())
                        cv2.imshow("ir", ir[0, 0, :, :].detach().cpu().numpy())
                        cv2.waitKey()


                    optimizer.zero_grad()
                    delta = torch.abs(ambient_est - ambient_gt).mean()
                    loss = delta.mean()
                    loss += torch.abs(LCN_tensors(ambient_est)[0] - LCN_tensors(ambient_gt)[0]).mean()
                    if half_precision:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                        model.zero_grad()

                else: # val
                    with torch.no_grad():
                        ambient_est= model(ir)

                    delta = torch.abs(ambient_est - ambient_gt).mean()
                    loss = delta.mean()
                    loss += torch.abs(LCN_tensors(ambient_est)[0] - LCN_tensors(ambient_gt)[0]).mean()

                loss_acc += loss.item()
                loss_acc_sub += loss.item()

                # print progress every 99 steps!
                if i_batch % 100 == 99:
                    cv2.imwrite(f"tmp/{phase}input.png", ir[0, 0, :, :].detach().cpu().numpy()*255)
                    cv2.imwrite(f"tmp/{phase}ambient_gt.png", ambient_gt[0, 0, :, :].detach().cpu().numpy()*255)
                    cv2.imwrite(f"tmp/{phase}ambient_est.png", ambient_est[0, 0, :, :].detach().cpu().numpy()*255)
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
