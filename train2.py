import model.composite_model
import torch
from params import parse_args
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
from model.losses import *
from model.composite_model import GetModel
from torch.utils.tensorboard import SummaryWriter
from dataset.datasets import GetDataset





def train():
    args, config = parse_args()

    model = GetModel(args, config)

    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    writer = SummaryWriter(f"tensorboard/{args.experiment_name}")

    scheduler_gamma = 0.5
    step_scale = 3
    scheduler_milestones = [int(20000 * step_scale),
                            int(30000 * step_scale),
                            int(40000 * step_scale),
                            int(50000 * step_scale)]
    overall_steps = int(60000 * step_scale)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_milestones,
                                               gamma=scheduler_gamma)

    if args.half_precision:
        scaler = GradScaler()

    assert args.dataset_type == "structure_core_combo", "Dataset type needs to be structure_core_combo"
    datasets, _, _, _, _, tgt_res = \
        GetDataset(args.path,
                   vertical_jitter=config["dataset"]["vertical_jitter"],
                   tgt_res=config["dataset"]["tgt_res"],
                   version=args.dataset_type)

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size,
                                                  shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    epochs = overall_steps // dataset_sizes["train"] + 1

    loss_min = 10000000
    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            for i_batch, sampled_batch in enumerate(dataloaders[phase]):
                print(sampled_batch)
                step = step + 1
                ir, irr, gt_x, gt_msk, gt_edge, has_gt = sampled_batch


                if phase == 'train':
                    x, sigma, class_losses, x_real, debug = model(ir, gt_x)







if __name__ == '__main__':
    train()