import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from model.backbone import *
from model.backboneSliced import *
from model.regressor import Reg_3stage, RegressorConv


class CompositeModel(nn.Module):
    def __init__(self, backbone, regressor, regressor_conv=None, half_precision=False):
        super(CompositeModel, self).__init__()
        self.half_precision = half_precision
        self.backbone = backbone
        self.regressor = regressor
        self.regressor_conv = regressor_conv

        # TODO: remove this debug(or at least make it so it can run with other than 64 channels
        # another TODO: set affine parameters to false!
        # self.bn = nn.BatchNorm2d(64, affine=False)

    def forward(self, x, x_gt=None):

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

            if self.regressor_conv != None:
                result = self.regressor_conv(x)
                results = results + result
            return results
        else:

            if self.half_precision:
                with autocast():
                    x = self.backbone(x)
                x = x.type(torch.float32)
            else:
                x = self.backbone(x)

            results = self.regressor(x, x_gt)
            if self.regressor_conv:
                result = self.regressor_conv(x)
                results = results + result
            return results



def GetModel(args, config):

    if config["regressor"]["load_file"] != "":
        regressor = torch.load(config["regressor"]["load_file"])
        regressor.eval()
    else:
        # https://stackoverflow.com/questions/334655/passing-a-dictionary-to-a-function-as-keyword-parameters
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
        # fix parameters in the backbone (or maybe not!)
        # for param in backbone.parameters():
        #    param.requires_grad = False
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
                channels_sub=config["backbone"]["channels2"],
                use_bn=True,
                pad=pad, channels_in=channels)

        if name == "BackboneU5":
            assert args.downsample_output, "For the U shaped network it is required to downsample the network"
            constructor = lambda pad, channels, downsample: BackboneU5Slice(pad=pad, in_channels=channels)
            backboneType = BackboneU5Slice

        if config["backbone"]["name"].endswith("Sliced"):
            backbone = BackboneSlicer(backboneType, constructor,
                                      config["backbone"]["slices"],
                                      lcn=args.LCN,
                                      downsample_output=args.downsample_output)
        else:
            in_channels = 1
            if args.LCN:
                in_channels = 2
            backbone = constructor('both', in_channels, args.downsample_output)

        if False:
            if config["backbone"]["name"] == "BackboneNoSlice3":
                print("BackboneNoSlice3")
                backbone = BackboneNoSlice3(height=config["dataset"]["slice_in"]["height"],
                                            channels=config["backbone"]["channels"],
                                            channels_sub=config["backbone"]["channels2"],
                                            use_bn=True, lcn=args.LCN)
            if config["backbone"]["name"] == "Backbone3Sliced":
                constructor = lambda pad, channels, downsample: Backbone3Slice(
                    channels=config["backbone"]["channels"],
                    channels_sub=config["backbone"]["channels2"],
                    use_bn=True,
                    pad=pad, channels_in=channels, downsample=downsample)

                backbone = BackboneSlicer(Backbone3Slice, constructor,
                                          config["backbone"]["slices"],
                                          lcn=args.LCN,
                                          downsample_output=args.downsample_output)
            if config["backbone"]["name"] == "BackboneU1":
                print("BACKBONEU1")
                backbone = BackboneU1()
            if config["backbone"]["name"] == "BackboneU2":
                print("BACKBONEU2")
                backbone = BackboneU2()
            if config["backbone"]["name"] == "BackboneU3":
                print("BACKBONEU3")
                backbone = BackboneU3()

            if config["backbone"]["name"] == "BackboneU4":
                print("BACKBONEU4")
                backbone = BackboneU4()

            if config["backbone"]["name"] == "BackboneU5":
                print("BACKBONEU5")
                backbone = BackboneU5(norm=config["backbone"]["norm"], lcn=args.LCN)

            if config["backbone"]["name"] == "BackboneU5Sliced":
                print("BACKBONEU5Sliced")
                #backbone = BackboneU5Sliced(slices=config["backbone"]["slices"], lcn=args.LCN)
                assert args.downsample_output, "For the U shaped network it is required to downsample the network"
                constructor = lambda pad, channels, downsample: BackboneU5Slice(pad=pad, in_channels=channels)
                backbone = BackboneSlicer(BackboneU5Slice, constructor,
                                          config["backbone"]["slices"],
                                          lcn=args.LCN,
                                          downsample_output=args.downsample_output)

    regressor_conv = None
    if "regressor_conv" in config:
        if config["regressor_conv"]["load_file"] != "":
            regressor_conv = torch.load(config["regressor_conv"]["load_file"])
            regressor_conv.eval()
        else:
            regressor_conv = RegressorConv(
                lines=config["regressor"]["lines"],
                ch_in=config["regressor"]["ch_in"],
                ch_latent=config["regressor_conv"]["layers"],
                ch_latent_msk=config["regressor_conv"]["layers_msk"],
                slices=config["regressor_conv"]["slices"],
                vertical_fourier_encoding=config["regressor_conv"]["vertical_fourier_encoding"],
                batch_norm=config["regressor_conv"]["batch_norm"])
    model = CompositeModel(backbone, regressor, regressor_conv, args.half_precision)
    return model