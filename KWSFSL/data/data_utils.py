import torch
from torch.utils.data import Dataset


class SetDataset:
    def __init__(self, dl_list):
        self.sub_dataloader = dl_list

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)


class SimpleDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __getitem__(self, index):
        item = self.data_list[index]
        if self.transform:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.data_list)
