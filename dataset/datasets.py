from dataset.dataset_rendered_2 import *
from dataset.dataset_rendered_shapenet import *

def GetDataset(path, is_npy, tgt_res, version="unity_4", debug=False):
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
            'train': DatasetRendered4(path, keys_train, tgt_res=tgt_res, debug=debug),
            'val': DatasetRendered4(path, keys_val, tgt_res=tgt_res, debug=debug)
        }

        src_res = (1216, 896)
        principal = (604, 457)
        focal = (1.1154399414062500e+03, 1.1154399414062500e+03) # same focal length in both directions
        has_lr = True
        baselines = {"right": 0.07501 - 0.0634, "left": -0.0634}
        return datasets, baselines, has_lr, focal, principal, src_res
    if version == 3: #todo: remove deprecated!!!
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
            'train': DatasetRendered3(path, keys_train, tgt_res=tgt_res, debug=debug),
            'val': DatasetRendered3(path, keys_val, tgt_res=tgt_res, debug=debug)
        }
        return datasets
    if version == 2: # todo: remove deprecated
        max_ind = 0
        files = os.listdir(path)
        for file in files:
            if os.path.isfile(f"{path}/{file}"):
                test = int(re.search(r'\d+', file).group())
                if test > max_ind:
                    max_ind = test

        datasets = {
            'train': DatasetRendered2(path, 0, int(max_ind*9/10) - 1, tgt_res=tgt_res, is_npy=is_npy, debug=debug),
            'val': DatasetRendered2(path, int(max_ind*9/10), max_ind, tgt_res=tgt_res, is_npy=is_npy, debug=debug)
        }
        return datasets


