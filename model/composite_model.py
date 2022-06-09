import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from model.backbone import *
from model.backboneSliced import *
from model.regressor import *
from model.uNet import UNet
import time


class CompositeModel(nn.Module):
    def __init__(self, backbone, regressor, half_precision=False, apply_lcn=False):
        super(CompositeModel, self).__init__()
        self.half_precision = half_precision
        self.backbone = backbone
        self.regressor = regressor
        self.apply_lcn = apply_lcn

    def forward(self, x, x_gt=None, output_entropies=False, measure_time=False):
        if hasattr(self.backbone, 'LCN'):
            # TODO: REMOVE THIS BRANCH AS SOON AS WE KNOW THAT THE OLD EXPERIMENTS CAN BE DELETED!
            if self.backbone.LCN:
                lcn, _, _ = LCN_tensors(x)
                x = torch.cat((x, lcn), 1)
        else:
            if self.apply_lcn:
                lcn, _, _ = LCN_tensors(x)
                x = torch.cat((x, lcn), 1)

        if x_gt != None:
            if self.half_precision:
                with autocast():
                    x, debugs = self.backbone(x, True)
                x = x.type(torch.float32)
            else:
                x, debugs = self.backbone(x, True)

            results = self.regressor(x, x_gt)
            # todo: batch norm the whole backbone and merge two dicts:
            # https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python-taking-union-o
            # z = {**x, **y}
            for key, val in debugs.items():
                results[-1][key] = val

            return results
        else:
            #measure_time = True

            if measure_time:
                self.half_precision = False
                tsince = int(round(time.time() * 1000))
            if self.half_precision:
                with autocast():
                    x = self.backbone(x)
                x = x.type(torch.float32)
            else:
                x = self.backbone(x)
            if measure_time:
                torch.cuda.synchronize()
                ttime_elapsed = int(round(time.time() * 1000)) - tsince
                print(f"backbone time elapsed {ttime_elapsed} ms")

                tsince = int(round(time.time() * 1000))
            results = self.regressor(x, x_gt,
                                     output_entropies=output_entropies)
            if measure_time:
                torch.cuda.synchronize()
                ttime_elapsed = int(round(time.time() * 1000)) - tsince
                print(f"regressor time elapsed {ttime_elapsed} ms")

            return results



def GetModel(args, config):

    if config["training"]["load_model"] != "":
        model = torch.load(config["training"]["load_model"])
        model.eval()
        return model

    if config["regressor"]["load_file"] != "":
        regressor = torch.load(config["regressor"]["load_file"])
        regressor.eval()
    else:

        in_channels = 1
        if args.LCN:
            in_channels = 2

        # https://stackoverflow.com/questions/334655/passing-a-dictionary-to-a-function-as-keyword-parameters
        if config["regressor"]["name"] == "Regressor":
            regressor = Regressor(  height=config["regressor"]["lines"],
                                    classes=config["regressor"]["classes"],
                                    class_pad=config["regressor"]["class_padding"],
                                    class_ch_in_offset=config["regressor"]["class_ch_in_offset"],
                                    class_ch_in=config["regressor"]["class_ch_in"],
                                    class_ch=config["regressor"]["class_ch"],
                                    reg_data_start=config["regressor"]["reg_data_start"],
                                    reg_ch_in=config["regressor"]["reg_ch_in"],
                                    reg_ch=config["regressor"]["reg_ch"],
                                    reg_superclasses=config["regressor"]["reg_superclasses"],
                                    reg_overlap_neighbours=config["regressor"]["reg_overlap_neighbours"], # take channels
                                    reg_shared_over_lines=config["regressor"]["reg_shared_over_lines"],
                                    reg_pad=config["regressor"]["reg_pad"])
        elif config["regressor"]["name"] == "None":
            regressor = RegressorNone( )
        elif config["regressor"]["name"] == "Lines":
            regressor = RegressorLinewise(lines=config["regressor"]["lines"],
                                          ch_in=config["regressor"]["ch_in"],
                                          ch_latent=config["regressor"]["ch_latent"])
        else:
            #TODO: REMOVE EVERYTHING Reg_3stage related!!!!
            print("THIS IS DEPRECATED!!!!! DON'T USE ANYMORE")
            regressor = Reg_3stage(ch_in=config["regressor"]["ch_in"],
                                   height=config["regressor"]["lines"],  # 64,#448,
                                   ch_latent=config["regressor"]["bb"],
                                   # [128, 128, 128],#todo: make this of variable length
                                   superclasses=config["regressor"]["superclasses"],
                                   ch_latent_r=config["regressor"]["ch_reg"],
                                   # 64/64 # in the current implementation there is only one stage between input
                                   ch_latent_msk=config["regressor"]["msk"],
                                   classes=config["regressor"]["classes"],
                                   pad=config["regressor"]["padding"],
                                   ch_latent_c=config["regressor"]["class_bb"],  # todo: make these of variable length
                                   regress_neighbours=config["regressor"]["regress_neighbours"],
                                   reg_line_div=config["regressor"]["reg_line_div"],
                                   c3_line_div=config["regressor"]["c3_line_div"],
                                   close_far_separation=config["regressor"]["close_far_separation"],
                                   sigma_mode=config["regressor"]["sigma_mode"],
                                   vertical_slices=config["backbone"]["slices"],
                                   pad_proj=args.pad_proj)

    if config["backbone"]["load_file"] != "":
        backbone = torch.load(config["backbone"]["load_file"])
        backbone.eval()
    else:
        # todo: remove numpy support!!!!
        name = config["backbone"]["name"].replace("Sliced", "")

        if name == "Backbone3":
            backboneType = Backbone3Slice
            constructor = lambda pad, channels, downsample: Backbone3Slice(
                channels=config["backbone"]["channels"],
                channels_sub=config["backbone"]["channels2"],
                use_bn=True,
                pad=pad, channels_in=channels, downsample=downsample)

        if name == "Backbone":
            backboneType = BackboneSlice
            constructor = lambda pad, channels, downsample: BackboneSlice(
                channels=config["backbone"]["channels"],
                kernel_sizes=config["backbone"]["kernel_sizes"],
                channels_sub=config["backbone"]["channels_sub"],
                kernel_sizes_sub=config["backbone"]["kernel_sizes_sub"],
                use_bn=True,
                pad=pad, channels_in=channels)
        if name == "UNet":
            backboneType = UNet
            constructor = lambda pad, channels, downsample: UNet(
                cin=config["backbone"]["input_channels"],
                cout=config["backbone"]["output_channels"],
                half_out=True,
                channel_size_scale=config["backbone"]["unet_channel_count_scale"]
            )

        if name == "BackboneU5":
            assert args.downsample_output, "For the U shaped network it is required to downsample the network"
            constructor = lambda pad, channels, downsample: BackboneU5Slice(pad=pad, in_channels=channels)
            backboneType = BackboneU5Slice

        if config["backbone"]["name"].endswith("Sliced"):
            backbone = BackboneSlicer(backboneType, constructor,
                                      config["backbone"]["slices"],
                                      in_channels=in_channels,
                                      downsample_output=args.downsample_output)
        else:
            backbone = constructor('both', in_channels, args.downsample_output)


    model = CompositeModel(backbone, regressor, args.half_precision, apply_lcn=args.LCN)
    return model