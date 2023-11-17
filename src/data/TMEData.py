from src.data.GeoMXData import GeoMXDataset
from torch_geometric.transforms import RootedRWSubgraph
from torch_geometric.data import Data
import torch
import os


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        if key == 'n_sub_batch_s':
            return 0
        if key == 'n_sub_batch_t':
            return 0
        return super().__inc__(key, value, *args, **kwargs)


class TMEDataset(GeoMXDataset):
    def __init__(self, 
                 root_dir='data/', 
                 raw_subset_dir='',
                 train_ratio=0.6, 
                 val_ratio=0.2,
                 label_data='OC1_all.csv',
                 walk_length=3,
                 repeat=1):
        
        self.walk_length = walk_length
        self.repeat = repeat
        self.RootedRWSubgraph = RootedRWSubgraph(walk_length=self.walk_length, repeat=self.repeat)
        super().__init__(root_dir=root_dir, raw_subset_dir=raw_subset_dir, label_data=label_data,
                 train_ratio=train_ratio, val_ratio=val_ratio, transform=self.transform)
    
    def transform(self, data):
        subs1 = self.RootedRWSubgraph(data)
        subs2 = self.RootedRWSubgraph(data)
        data = PairData(x_s=subs1.x[subs1.n_id], edge_index_s=subs1.sub_edge_index,
                        edge_attr_s=subs1.edge_attr[subs1.e_id], n_sub_batch_s=subs1.n_sub_batch,
                        x_t=subs2.x[subs2.n_id], edge_index_t=subs2.sub_edge_index,
                        edge_attr_t=subs2.edge_attr[subs2.e_id], n_sub_batch_t=subs2.n_sub_batch,)
        return data
    
    def get(self, idx):
        if self.mode == self.train:
            return torch.load(os.path.join(self.processed_dir, self.data[self.train_map][idx]))
        elif self.mode == self.val:
            return torch.load(os.path.join(self.processed_dir, self.data[self.val_map][idx]))
        elif self.mode == self.test:
            return torch.load(os.path.join(self.processed_dir, self.data[self.test_map][idx]))
        else:
            return torch.load(os.path.join(self.processed_dir, self.data[idx]))
    
    def subgraph_batching(self, batch):
        max_val = 0
        for n_batch in torch.unique(batch.x_s_batch):
            batch.n_sub_batch_s[batch.x_s_batch==n_batch] += max_val
            max_val = torch.max(batch.n_sub_batch_s[batch.x_s_batch==n_batch]) + 1
        for n_batch in torch.unique(batch.x_t_batch):
            batch.n_sub_batch_t[batch.x_t_batch==n_batch] += max_val
            max_val = torch.max(batch.n_sub_batch_t[batch.x_t_batch==n_batch]) + 1
        batch = Data(x=torch.cat((batch.x_s, batch.x_t)),
                     edge_index=torch.cat((batch.edge_index_s, batch.edge_index_t), dim=1),
                     edge_attr=torch.cat((batch.edge_attr_s, batch.edge_attr_t)),
                     batch=torch.cat((batch.n_sub_batch_s, batch.n_sub_batch_t)))
        return batch

