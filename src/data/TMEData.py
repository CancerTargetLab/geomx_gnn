from src.data.GeoMXData import GeoMXDataset
from torch_geometric.transforms import RootedRWSubgraph
import torch

class TMEDataset(GeoMXDataset):
    def __init__(self, root_dir='data/', raw_subset_dir='',
                 train_ratio = 0.6, val_ratio = 0.2, walk_length=3,
                 repeat=1):
        self.walk_length = walk_length
        self.repeat = repeat
        self.RootedRWSubgraph = RootedRWSubgraph(walk_length=self.walk_length, repeat=self.repeat)
        super().__init__(root_dir=root_dir, raw_subset_dir=raw_subset_dir,
                 train_ratio=train_ratio, val_ratio=val_ratio, transform=self.transform)
    
    def transform(self, data):
        subs1 = self.RootedRWSubgraph(data)
        subs2 = self.RootedRWSubgraph(data)
        subs1.sub_edge_index = torch.cat((subs1.sub_edge_index, subs2.sub_edge_index), dim=1)
        subs1.n_id = torch.cat((subs1.n_id, subs2.n_id), dim=0)
        subs1.e_id = torch.cat((subs1.e_id, subs2.e_id), dim=0)
        subs1.n_sub_batch = torch.cat((subs1.n_sub_batch, subs2.n_sub_batch), dim=0)
        subs1.e_sub_batch = torch.cat((subs1.e_sub_batch, subs2.e_sub_batch), dim=0)
        return subs1