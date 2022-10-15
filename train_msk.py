import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
from dataset.datasets import GetDataset
from model.composite_model import CompositeModel, GetModel
from model.backbone import *
from model.backboneSliced import *
from model.regressor import Reg_3stage
from torch.utils.data import DataLoader
import math
from params import parse_args
from model.invalidator import InvalidationModel, InvalidationModelU

from torch.utils.tensorboard import SummaryWriter



def train():
    # TODO: DO SOMETHING U-NET to get the artifacts on the left of the image out!
    # or give the x position as fourier features?
    # some data augmentation too!
    experiment_name = "mask_training_7"
    writer = SummaryWriter(f"tensorboard/{experiment_name}")

    dataset_path = "/media/simon/sandisk2/dataset_collection/synthetic_test"

    model_path = "trained_models/full_76_lcn_j4_c1280_chk.pt"

    weight_invalid = 5.0
    learning_rate = 1e-3
    weight_decay = 1e-4
    batch_size = 4
    num_workers = 8
    epochs = 40
    half_precision = True

    main_device = torch.cuda.current_device()
    model_depth = torch.load(model_path)
    model_depth.cuda()
    model_depth.eval()

    #model_invalidation = InvalidationModel([4, 8, 32, 32, 1])
    #model_invalidation = InvalidationModelU()
    #maybe this is not enough to detect the artifacts on the left of the image
    model_invalidation = InvalidationModel([4, 8, 16, 16, 1])
    model_invalidation.cuda()
    model_invalidation.eval()

    loss_function = torch.nn.BCEWithLogitsLoss(reduction='none')

    optimizer = optim.Adam(model_invalidation.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)

    datasets, _, _, _, _, tgt_res = \
        GetDataset(dataset_path,
                   vertical_jitter=4,
                   tgt_res=[1216, 896],
                   version="structure_core_unity",
                   left_only=True)


    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
                   for x in ['train', 'val']}

    #TODO: load the original training set to perform testing.

    if half_precision:
        scaler = GradScaler()

    min_loss = 4
    step = -1
    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs - 1}')
        print('-' * 10)
        for phase in ['train', 'val']:
            print(phase)
            if phase == 'train':
                model_invalidation.train()  # Set model to training mode
            else:
                model_invalidation.eval()  # Set model to evaluate mode

            loss_accu = 0
            discarded_valid_accu = 0
            undetected_invalid_accu = 0
            discarded_valid_handcrafted_accu = 0
            undetected_invalid_handcrafted_accu = 0
            invalid_accu = 0
            loss_accu_epoch = 0
            step_batch = 0

            model_invalidation.zero_grad()#TODO: necessary??
            for i_batch, sampled_batch in enumerate(dataloaders[phase]):
                step = step + 1
                step_batch = step_batch + 1
                if len(sampled_batch) == 4:
                    ir, x_gt, mask_gt, edge_mask = sampled_batch
                    #rendered_msk = torch.ones((ir.shape[0], 1, 1, 1), dtype=torch.float32)
                else:
                    ir, x_gt, mask_gt, edge_mask, _ = sampled_batch

                ir = ir.to(main_device)
                mask_gt = mask_gt.to(main_device)
                x_gt = x_gt.to(main_device)
                #rendered_msk = rendered_msk.to(main_device)

                mask_gt[torch.logical_or(x_gt < 0.0, x_gt > 1.0)] = 0

                with torch.no_grad():
                    x, entropy1, entropy2, entropy3 = model_depth(ir.type(torch.half), output_entropies=True)
                    invalid_pixel_gt = torch.ones_like(mask_gt)
                    invalid_pixel_gt[torch.abs(x - x_gt) < (
                                5 / 1280)] = 0.0  # invalidate every pixel we are more than 10 pixel away
                    focal = 1.1154399414062500e+03
                    baseline = 0.0634
                    x_0 = torch.arange(0, x.shape[3]).reshape([1, 1, 1, x.shape[3]]).cuda()
                    disp = x * x.shape[3] - x_0
                    disp *= -2.0  # scale the disparity
                    #disp[disp < 0] = 0
                    #disp[disp > 1000] = 0
                    depth = focal*baseline / disp
                    depth[torch.isnan(depth)] = 0
                    depth[depth < 0] = 0
                    depth[depth > 20] = 0

                    invalid_pixel_handcrafted = ((entropy1 + entropy2 + entropy3) > 3.0).float()
                    discarded_valid = torch.logical_and(invalid_pixel_handcrafted > 0.5, invalid_pixel_gt < 0.5)
                    discarded_valid = torch.mean(discarded_valid.float())
                    discarded_valid_handcrafted_accu += discarded_valid

                    undetected_invalid = torch.logical_and(invalid_pixel_handcrafted < 0.5, invalid_pixel_gt > 0.5)
                    undetected_invalid = torch.mean(undetected_invalid.float())
                    undetected_invalid_handcrafted_accu += undetected_invalid.item()

                    if False:
                        cv2.imshow("ir", ir[0, 0, :, :].detach().cpu().numpy())
                        cv2.imshow("x", x[0, 0, :, :].detach().cpu().numpy())
                        cv2.imshow("disp", disp[0, 0, :, :].detach().cpu().numpy()/100)
                        cv2.imshow("depth", depth[0, 0, :, :].detach().cpu().numpy()/5)
                        cv2.imshow("invalid_pixel_gt", invalid_pixel_gt[0, 0, :, :].detach().cpu().numpy())
                        cv2.imshow("entropy1", entropy1[0, 0, :, :].detach().cpu().numpy())
                        cv2.imshow("entropy2", entropy2[0, 0, :, :].detach().cpu().numpy())
                        cv2.imshow("entropy3", entropy3[0, 0, :, :].detach().cpu().numpy())
                        cv2.imshow("invalid_pixel_test", invalid_pixel_handcrafted[0, 0, :, :].detach().cpu().numpy())
                        cv2.waitKey()
                    features = torch.cat([depth, entropy1, entropy2, entropy3], dim=1)


                if phase == 'train':
                    torch.autograd.set_detect_anomaly(True)
                    optimizer.zero_grad()
                    if half_precision:
                        with torch.cuda.amp.autocast():
                            invalid_pixel_estimate = model_invalidation(features)
                            loss = loss_function(invalid_pixel_estimate, invalid_pixel_gt)
                            loss[invalid_pixel_gt > 0.5] *= weight_invalid
                            loss = torch.mean(loss)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        invalid_pixel_estimate = model_invalidation(features)
                        loss = loss_function(invalid_pixel_estimate, invalid_pixel_gt)
                        loss[invalid_pixel_gt > 0.5] *= weight_invalid
                        loss = torch.mean(loss)
                        loss.backward()
                        optimizer.step()

                else:# phase == 'val'
                    with torch.no_grad():
                        if half_precision:
                            with torch.cuda.amp.autocast():
                                invalid_pixel_estimate = model_invalidation(features)
                        else:
                            invalid_pixel_estimate = model_invalidation(features)
                        loss = loss_function(invalid_pixel_estimate, invalid_pixel_gt)
                        loss = torch.mean(loss)

                discarded_valid = torch.logical_and(invalid_pixel_estimate > 0.5, invalid_pixel_gt < 0.5)
                discarded_valid = torch.mean(discarded_valid.float())
                discarded_valid_accu += discarded_valid

                undetected_invalid = torch.logical_and(invalid_pixel_estimate < 0.5, invalid_pixel_gt > 0.5)
                undetected_invalid = torch.mean(undetected_invalid.float())
                undetected_invalid_accu += undetected_invalid.item()

                invalid_accu += torch.mean(invalid_pixel_gt).mean().item()

                loss_accu += loss.item()
                loss_accu_epoch += loss.item()

                if step_batch % 100 == 99:
                    #TODO false positives, false negatives
                    bce = loss_accu / 100
                    discarded_valid = discarded_valid_accu / (100.0 - invalid_accu)
                    undetected_invalid = undetected_invalid_accu / invalid_accu
                    print(f"batch{step_batch} binary cross entropy {bce}, discarded valid {discarded_valid}, "
                          f"undetected invalid {undetected_invalid}")

                    writer.add_scalar(f'{phase}_subepoch/binary_cross_entropy',
                                      loss_accu / 100.0, step)
                    writer.add_scalar(f'{phase}_subepoch/discarded_valid',
                                      discarded_valid, step)
                    writer.add_scalar(f'{phase}_subepoch/undetected_invalid',
                                      undetected_invalid, step)

                    #output the same metrics for the handcrafted features:
                    discarded_valid_handcrafted = discarded_valid_handcrafted_accu / (100.0 - invalid_accu)
                    undetected_invalid_handcrafted = undetected_invalid_handcrafted_accu / invalid_accu
                    print(f"compared_to_handcrafted: discarded valid {discarded_valid_handcrafted}, "
                          f"undetected invalid {undetected_invalid_handcrafted}")

                    writer.add_scalar(f'{phase}_subepoch/discarded_valid_relative',
                                      discarded_valid/discarded_valid_handcrafted, step)
                    writer.add_scalar(f'{phase}_subepoch/undetected_invalid_relative',
                                      undetected_invalid/undetected_invalid_handcrafted, step)

                    loss_accu = 0
                    discarded_valid_accu = 0
                    undetected_invalid_accu = 0
                    discarded_valid_handcrafted_accu = 0
                    undetected_invalid_handcrafted_accu = 0
                    invalid_accu = 0

            loss = loss_accu_epoch / step_batch
            writer.add_scalar(f'{phase}/binary_cross_entropy',
                              loss_accu_epoch / step_batch, step)

            if phase == 'val':
                print("storing network")
                torch.save(model_invalidation, f"trained_models/{experiment_name}_chk.pt")
                if loss < min_loss:
                    print("storing new best weights")
                    min_loss = loss
                    torch.save(model_invalidation, f"trained_models/{experiment_name}.pt")

    writer.close()


if __name__ == '__main__':
    train()
