from src.data.GeoMXData import GeoMXDataset
from torch_geometric.transforms import RootedEgoNets
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge, dropout_node
from torch_geometric.transforms import RemoveIsolatedNodes, AddRemainingSelfLoops
from torch import Tensor
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
                 node_dropout=0.2,
                 edge_dropout=0.3,
                 label_data='OC1_all.csv',
                 num_hops=1,
                 subgraphs_per_graph=10):
        
        self.num_hops = num_hops
        self.subgraphs_per_graph = subgraphs_per_graph
        self.RootedEgoNets = RootedEgoNets(num_hops=1)
        self.node_dropout = node_dropout
        self.edge_dropout = edge_dropout
        super().__init__(root_dir=root_dir, raw_subset_dir=raw_subset_dir, label_data=label_data,
                 train_ratio=train_ratio, val_ratio=val_ratio, transform=self.transform)
    
    def transform(self, data):
        subs1 = self.RootedEgoNets(data)
        subs1 = Data(x=subs1.x[subs1.n_id], edge_index=subs1.sub_edge_index,
                        edge_attr=subs1.edge_attr[subs1.e_id], n_sub_batch=subs1.n_sub_batch)
        subs2 = subs1.clone()
        num_nodes = subs1.num_nodes 
        egonets = torch.unique(subs1.n_sub_batch)

        node_map = dropout_node(subs1.edge_index, p=self.node_dropout, training=self.mode==self.train)
        subs1.edge_index, subs1.edge_attr = node_map[0], subs1.edge_attr[node_map[1]]
        subs1.x, subs1.n_sub_batch = subs1.x[node_map[2]], subs1.n_sub_batch[node_map[2]]
        edge_map = dropout_edge(subs1.edge_index, p=self.edge_dropout, force_undirected=True, training=self.mode==self.train)
        subs1.edge_index, subs1.edge_attr = edge_map[0], subs1.edge_attr[edge_map[1]]

        node_map = dropout_node(subs2.edge_index, p=self.node_dropout, training=self.mode==self.train)
        subs2.edge_index, subs2.edge_attr = node_map[0], subs2.edge_attr[node_map[1]]
        subs2.x, subs2.n_sub_batch = subs2.x[node_map[2]], subs2.n_sub_batch[node_map[2]]
        edge_map = dropout_edge(subs2.edge_index, p=self.edge_dropout, force_undirected=True, training=self.mode==self.train)
        subs2.edge_index, subs2.edge_attr = edge_map[0], subs2.edge_attr[edge_map[1]]

        if self.mode==self.train:
            sub = torch.randint(0, data.num_nodes, (int(torch.min(torch.Tensor([data.num_nodes, self.subgraphs_per_graph]))),), device=data.edge_index.device)
            subs1.n_id, subs1.n_sub_batch = subs1.n_id[torch.isin(subs1.n_sub_batch, sub)], subs1.n_sub_batch[torch.isin(subs1.n_sub_batch, sub)]
            subs1.e_id, subs1.e_sub_batch, subs1.sub_edge_index = subs1.e_id[torch.isin(subs1.e_sub_batch, sub)], subs1.e_sub_batch[torch.isin(subs1.e_sub_batch, sub)], subs1.sub_edge_index[:,torch.isin(subs1.e_sub_batch, sub)]
            subs2.n_id, subs2.n_sub_batch = subs2.n_id[torch.isin(subs2.n_sub_batch, sub)], subs2.n_sub_batch[torch.isin(subs2.n_sub_batch, sub)]
            subs2.e_id, subs2.e_sub_batch, subs2.sub_edge_index = subs2.e_id[torch.isin(subs2.e_sub_batch, sub)], subs2.e_sub_batch[torch.isin(subs2.e_sub_batch, sub)], subs2.sub_edge_index[:,torch.isin(subs2.e_sub_batch, sub)]

        data_s, data_t = RemoveIsolatedNodes()(data_s), RemoveIsolatedNodes()(data_t)
        data_s, data_t = AddRemainingSelfLoops(data_s), AddRemainingSelfLoops(data_t)

        data = PairData(x_s=data_s.x, edge_index_s=data_s.edge_index,
                        edge_attr_s=data_s.edge_attr[subs1.e_id], n_sub_batch_s=data_s.n_sub_batch,
                        x_t=data_t.x, edge_index_t=data_t.edge_index,
                        edge_attr_t=data_t.edge_attr, n_sub_batch_t=data_t.n_sub_batch)
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

