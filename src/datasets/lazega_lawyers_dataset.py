import torch
from torch_geometric.data import Dataset
from torch_geometric.data import download_url
import os.path as osp


class LazegaLawyersDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['adv', 'fri', 'work']

    @property
    def processed_file_names(self):
        return ['data_lawyers_adv.pt', 'data_lawyers_fri.pt', 'data_lawyers_work.pt']

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        dict_name = {0: 'adv', 1: 'fri', 2: 'work'}
        idx = dict_name[idx]
        data = torch.load(osp.join(self.processed_dir, f'data_lawyers_{idx}.pt'))
        return data
