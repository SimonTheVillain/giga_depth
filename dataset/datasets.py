from dataset.dataset_rendered_2 import *
from dataset.dataset_rendered_shapenet import *
from dataset.dataset_rendered_3 import DatasetRenderedSequences
from pathlib import Path

def GetDataset(path, tgt_res, vertical_jitter=1, version="unity_4", debug=False):
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

    if version == "structure_core_unity_4":
        files = os.listdir(path)
        #print(files)
        keys = []
        for file in files:
            if os.path.isfile(f"{path}/{file}"):
                keys.append(file.split("_")[0])

        keys = list(set(keys))
        #print(keys)
        # split up training and validation set:
        keys_train = keys[0:int((len(keys) * 95) / 100)]
        keys_val = keys[int((len(keys) * 95) / 100):]

        datasets = {
            'train': DatasetRendered4(path, keys_train, vertical_jitter=vertical_jitter,
                                      tgt_res=tgt_res, debug=debug),
            'val': DatasetRendered4(path, keys_val, vertical_jitter=vertical_jitter,
                                    tgt_res=tgt_res, debug=debug)
        }

        src_res = (1216, 896)
        principal = (604, 457)
        focal = (1.1154399414062500e+03, 1.1154399414062500e+03) # same focal length in both directions
        has_lr = True
        baselines = {"right": 0.07501 - 0.0634, "left": -0.0634}
        return datasets, baselines, has_lr, focal, principal, src_res

    if version == "structure_core_unity_sequences":
        sequences = os.listdir(path)

        paths = [Path(path) / x for x in sequences if os.path.isdir(Path(path) / x)]
        paths.sort()

        if False:#TODO: remove!!!
            dataset_paths = ["/media/simon/WD/datasets_raw/structure_core_unity_1",
                             "/media/simon/WD/datasets_raw/structure_core_unity_2",
                             "/media/simon/WD/datasets_raw/structure_core_unity_3",
                             "/media/simon/LaCie/datasets_raw/structure_core_unity_4",
                             "/media/simon/LaCie/datasets_raw/structure_core_unity_5",
                             "/media/simon/LaCie/datasets_raw/structure_core_unity_6",
                             "/media/simon/WD/datasets_raw/structure_core_unity_7"]
            paths = []
            for dataset_path in dataset_paths:
                folders = os.listdir(dataset_path)
                folders.sort()
                folders = [Path(dataset_path) / x for x in folders if os.path.isdir(Path(dataset_path) / x)]
                paths += folders

        paths_train = paths[:len(paths) - 64]
        paths_val = paths[len(paths) - 64:]

        datasets = {
            'train': DatasetRenderedSequences(paths_train, vertical_jitter=vertical_jitter,
                                      tgt_res=tgt_res, debug=debug),
            'val': DatasetRenderedSequences(paths_val, vertical_jitter=vertical_jitter,
                                    tgt_res=tgt_res, use_all_frames=True, debug=debug)
        }

        src_res = (1216, 896)
        principal = (604, 457)
        focal = (1.1154399414062500e+03, 1.1154399414062500e+03) # same focal length in both directions
        has_lr = True
        baselines = {"right": 0.07501 - 0.0634, "left": -0.0634}
        return datasets, baselines, has_lr, focal, principal, src_res





