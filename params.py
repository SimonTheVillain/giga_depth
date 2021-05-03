import yaml
import argparse
import os
import torch


def parse_args():
    with open("configs/default.yaml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", dest="config", action="store",
                        help="Load the config file with parameters.",
                        default="")
    args, additional_args = parser.parse_known_args()
    if args.config != "":
        with open(args.config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)
            # todo: recursively merge both config structures!!!!!!!
    # parser.add_argument("-V", "--version", help="show program version", action="store_true")
    parser.add_argument("-d", "--dataset_path", dest="path", action="store",
                        help="Path to the dataset.",
                        default=os.path.expanduser(config["dataset"]["path"]))
    parser.add_argument("-npy", "--npy_dataset", dest="is_npy", action="store_const",
                        help="Loads data directly form numpy files",
                        default=bool(config["dataset"]["is_npy"]), const=True)
    parser.add_argument("-b", "--batch_size", dest="batch_size", action="store",
                        help="The batch size during training",
                        type=int,
                        default=int(config["training"]["batch_size"]))
    parser.add_argument("-e", "--epochs", dest="epochs", action="store",
                        help="The number of epochs before quitting training.",
                        default=config["training"]["epochs"])
    parser.add_argument("-n", "--experiment_name", dest="experiment_name", action="store",
                        help="The name of this training for tensorboard and checkpoints.",
                        default=config["training"]["name"])
    parser.add_argument("-w", "--num_workers", dest="num_workers", action="store",
                        help="The number of threads working on loading and preprocessing data.",
                        type=int,
                        default=config["dataset"]["workers"])
    parser.add_argument("-g", "--gpu_list", dest="gpu_list", action="store",
                        nargs="+", type=int,
                        default=list(range(0, torch.cuda.device_count())))
    parser.add_argument("-r", "--learning_rate", dest="learning_rate", action="store",
                        help="Learning rate for gradient descent algorithm.",
                        type=float,
                        default=config["training"]["learning_rate"])
    parser.add_argument("-m", "--momentum", dest="momentum", action="store",
                        help="Momentum for gradient descent algorithm.",
                        type=float,
                        default=config["training"]["momentum"])
    parser.add_argument("-wd", "--weight_decay", dest="weight_decay", action="store",
                        help="Weight decay, effectively this is an l2 loss for the weights.",
                        type=float,
                        default=config["training"]["weight_decay"] if "weight_decay" in config["training"] else 0)
    default_acc = config["training"]["accumulation_steps"] if "accumulation_steps" in config["training"] else 1
    parser.add_argument("-acc", "--accumulation_steps", dest="accumulation_steps", action="store",
                        help="Accumulate gradient for a few steps before updating weights.",
                        type=float,
                        default=default_acc)
    parser.add_argument("-o", "--optimizer", dest="optimizer", action="store",
                        help="The optimizer used for training sgd or adam",
                        type=str,
                        default=config["training"]["optimizer"] if "optimizer" in config["training"] else "sgd")
    default_precision = bool(config["training"]["half_precision"]) if "half_precision" in config["training"] else False
    parser.add_argument("-hp", "--half_precision", dest="half_precision", action="store_const",
                        help="Utilize half precision for the backbone of the network.",
                        default=default_precision,
                        const=True)
    parser.add_argument("-a", "--alpha_reg", dest="alpha_reg", action="store",
                        help="The factor with which the regression error is incorporated into the loss.",
                        type=float,
                        nargs="+",
                        default=config["training"]["alpha_reg"])
    parser.add_argument("-as", "--alpha_sigma", dest="alpha_sigma", action="store",
                        help="The factor with which mask error is incorporated into the loss.",
                        type=float,
                        nargs="+",
                        default=config["training"]["alpha_sigma"])
    parser.add_argument("-ot", "--outlier_thresholds", dest="outlier_thresholds", action="store",
                        help="The thresholds for which the outlier ratios will be logged.",
                        type=float,
                        nargs="+",
                        default=[0.5, 1, 2, 5])
    parser.add_argument("-otr", "--relative_outlier_thresholds", dest="relative_outlier_thresholds", action="store",
                        help="The thresholds for which the outlier ratios will be logged. "
                             "The first value is the one every other is relative to.",
                        type=float,
                        nargs="+",
                        default=[5, 0.1, 0.2, 0.3, 0.4])

    parser.add_argument("-lcn", "--local_contrast_normalization", dest="LCN", action="store_const",
                        help="Use Local Contrast Normalization to increase signal at the input. ",
                        default=bool(config["backbone"]["local_contrast_norm"]),
                        const=True)

    parser.add_argument("-ew", "--edge_weight", dest="edge_weight", action="store",
                        help="Giving the depth estimate more weight at the edges.",
                        nargs="+",
                        default=config["training"]["edge_weight"])

    args = parser.parse_args(additional_args)

    return args, config
