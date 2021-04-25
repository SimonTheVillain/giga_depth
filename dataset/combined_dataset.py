import torch
import torch.utils.data as data

class DatasetCombined(data.Dataset):

    def __init__(self, datasetSrc, datasetDst):
        self.datasetSrc = datasetSrc
        self.datasetDst = datasetDst

    def __len__(self):
        return len(self.datasetSrc) + len(self.datasetDst)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        size1 = len(self.datasetSrc)
        if idx < size1:
            image, _, _ = self.datasetSrc[idx]
            domain = 0.0
        else:
            image = self.datasetDst[idx-size1]
            domain = 1.0

        sample = {'image': image, 'domain': domain}
        return sample

