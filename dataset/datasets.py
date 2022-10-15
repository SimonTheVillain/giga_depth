from dataset.dataset_rendered_shapenet import *
from dataset.dataset_rendered import DatasetRenderedSequences
from dataset.combined_dataset import DatasetCombined
from pathlib import Path
import re

def GetDataset(path, tgt_res, vertical_jitter=1, version="unity_4", debug=False, left_only=False):
    if version == "depth_in_space":

        datasets = {"train": DatasetRenderedShapenet(path, "train", debug=debug, vertical_jitter= vertical_jitter),
                    "val": DatasetRenderedShapenet(path, "val", debug=debug, vertical_jitter= vertical_jitter),
                    "test": DatasetRenderedShapenet(path, "test", debug=debug, vertical_jitter= vertical_jitter)}

        # for the captured: 512, 432
        # for rendered_default: 512, 432
        # for rendered_kinect: 512, 432
        # for rendered_real: 512, 432
        src_res = (432, 512) # 512 vertically, 432 horizontally
        principal = (432.0/2.0, 512.0/2.0)
        focal = (570, 570) # same focal length in both directions
        baselines = {'left': -0.075}
        has_lr = False
        return datasets, baselines, has_lr, focal, principal, src_res

    if version == "shapenet_half_res":

        datasets = {"train": DatasetRenderedShapenet(path, "train", debug=debug),
                    "val": DatasetRenderedShapenet(path, "val", debug=debug),
                    "test": DatasetRenderedShapenet(path, "test", debug=debug)}

        src_res = (640, 480)
        principal = (324.7, 250.1)
        focal = (567.6, 570.2) # same focal length in both directions
        baselines = {'left': -0.075}
        has_lr = False
        return datasets, baselines, has_lr, focal, principal, src_res
    if version == "shapenet_full_res":

        datasets = {"train": DatasetRenderedShapenet(path, "train", full_res=True, use_npy=False, debug=debug),
                    "val": DatasetRenderedShapenet(path, "val", full_res=True, use_npy=False, debug=debug),
                    "test": DatasetRenderedShapenet(path, "test", full_res=True, use_npy=False, debug=debug)}

        src_res = (640 * 2, 480 * 2)
        principal = (324.7 * 2, 250.1 * 2)
        focal = (567.6 * 2, 570.2 * 2) # same focal length in both directions
        baselines = {'left': -0.075}
        has_lr = False
        return datasets, baselines, has_lr, focal, principal, src_res

    if version == "structure_core_unity_sequences":
        sequences = os.listdir(path)

        paths = [Path(path) / x for x in sequences if os.path.isdir(Path(path) / x)]
        paths.sort()

        paths_train = paths[:len(paths) - 64]
        paths_val = paths[len(paths) - 64:]

        datasets = {
            'train': DatasetRenderedSequences(paths_train, vertical_jitter=vertical_jitter,
                                      tgt_res=tgt_res, debug=debug, left_only=left_only),
            'val': DatasetRenderedSequences(paths_val, vertical_jitter=vertical_jitter,
                                    tgt_res=tgt_res, use_all_frames=True, debug=debug, left_only=left_only)
        }

        src_res = (1216, 896)
        principal = (604, 457)
        focal = (1.1154399414062500e+03, 1.1154399414062500e+03) # same focal length in both directions
        has_lr = True
        baselines = {"right": 0.07501 - 0.0634, "left": -0.0634}
        return datasets, baselines, has_lr, focal, principal, src_res

    if version == "structure_core_unity":
        files = os.listdir(path)
        indices = set()
        for file in files:
            result = re.search(r"\d+", file)
            indices.add(result.group(0))
        indices = list(indices)
        indices.sort()
        indices = [str(Path(path) / x) for x in indices if os.path.isfile(f"{Path(path) / x}_left.jpg")]

        indices_train = indices[:len(indices) - 256]
        indices_val = indices[len(indices) - 256:]
        datasets = {
            'train': DatasetRenderedSequences(indices_train, vertical_jitter=vertical_jitter,
                                      tgt_res=tgt_res, debug=debug, left_only=left_only),
            'val': DatasetRenderedSequences(indices_val, vertical_jitter=vertical_jitter,
                                    tgt_res=tgt_res, use_all_frames=True, debug=debug, left_only=left_only)
        }

        src_res = (1216, 896)
        principal = (604, 457)
        focal = (1.1154399414062500e+03, 1.1154399414062500e+03) # same focal length in both directions
        has_lr = True
        baselines = {"right": 0.07501 - 0.0634, "left": -0.0634}
        return datasets, baselines, has_lr, focal, principal, src_res

    if version == "structure_core_combo":
        datasets = {
            'train': DatasetCombined(path, vertical_jitter=vertical_jitter, type='train'),
            'val': DatasetCombined(path, vertical_jitter=vertical_jitter, type='val')
        }

        src_res = (1216, 896)
        principal = (604, 457)
        focal = (1.1154399414062500e+03, 1.1154399414062500e+03)  # same focal length in both directions
        has_lr = True
        baselines = {"right": 0.0634, "left": 0.0634}
        return datasets, baselines, has_lr, focal, principal, src_res





