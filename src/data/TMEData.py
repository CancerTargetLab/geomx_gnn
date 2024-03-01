from src.data.GeoMXData import GeoMXDataset
from torch_geometric.transforms import RootedEgoNets
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge, dropout_node, k_hop_subgraph
from torch_geometric.transforms import RemoveIsolatedNodes, AddRemainingSelfLoops
import torch


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
        self.RemoveIsolatedNodes = RemoveIsolatedNodes()
        self.AddRemainingSelfLoops = AddRemainingSelfLoops(attr='edge_attr', fill_value=0.1)
        self.node_dropout = node_dropout
        self.edge_dropout = edge_dropout
        super().__init__(root_dir=root_dir, raw_subset_dir=raw_subset_dir, label_data=label_data,
                 train_ratio=train_ratio, val_ratio=val_ratio, transform=self.transform)
    
    def transform(self, data):
        sub = torch.randint(0, data.num_nodes, (int(torch.min(torch.Tensor([data.num_nodes, self.subgraphs_per_graph]))),), device=data.edge_index.device)
        new_data = None
        for node_i in list(range(sub.shape[0])):
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(sub[node_i].item(), self.num_hops, data.edge_index, relabel_nodes=True, directed=False)
            subgraph = Data(x=data.x[subset], edge_index=edge_index, edge_attr=data.edge_attr[edge_mask])

            valid_dropout = True
            while valid_dropout:
                subs1 = subgraph.clone()
                node_map = dropout_node(subs1.edge_index, p=self.node_dropout)[1]
                subs1.edge_index, subs1.edge_attr = subs1.edge_index[:,node_map], subs1.edge_attr[node_map]
                edge_map = dropout_edge(subs1.edge_index, p=self.edge_dropout)[1]
                subs1.edge_index, subs1.edge_attr = subs1.edge_index[:,edge_map], subs1.edge_attr[edge_map]
                subs1 = self.RemoveIsolatedNodes(subs1)
                subs1 = self.AddRemainingSelfLoops(subs1)
                valid_dropout =  0 == subs1.x.shape[0]

            valid_dropout = True
            while valid_dropout:
                subs2 = subgraph.clone()
                node_map = dropout_node(subs2.edge_index, p=self.node_dropout)[1]
                subs2.edge_index, subs2.edge_attr = subs2.edge_index[:,node_map], subs2.edge_attr[node_map]
                edge_map = dropout_edge(subs2.edge_index, p=self.edge_dropout)[1]
                subs2.edge_index, subs2.edge_attr = subs2.edge_index[:,edge_map], subs2.edge_attr[edge_map]
                subs2 = self.RemoveIsolatedNodes(subs2)
                subs2 = self.AddRemainingSelfLoops(subs2)
                valid_dropout =  0 == subs2.x.shape[0]

            if new_data is not None:
                new_data.edge_index_s = torch.cat((new_data.edge_index_s, subs1.edge_index+new_data.x_s.shape[0]), dim=1)
                new_data.edge_index_t = torch.cat((new_data.edge_index_t, subs2.edge_index+new_data.x_t.shape[0]), dim=1)
                new_data.x_s, new_data.x_t = torch.cat((new_data.x_s, subs1.x)), torch.cat((new_data.x_t, subs2.x))
                new_data.edge_attr_s = torch.cat((new_data.edge_attr_s, subs1.edge_attr))
                new_data.edge_attr_t = torch.cat((new_data.edge_attr_t, subs2.edge_attr))
                new_data.n_sub_batch_s= torch.cat((new_data.n_sub_batch_s, torch.tensor([node_i]*subs1.num_nodes, dtype=int)))
                new_data.n_sub_batch_t = torch.cat((new_data.n_sub_batch_t, torch.tensor([node_i]*subs2.num_nodes, dtype=int)))
                if not torch.unique(new_data.n_sub_batch_s).shape[0] == torch.unique(new_data.n_sub_batch_t).shape[0]:
                    raise Exception('Data is unequal!!')
            else:
                new_data = PairData(x_s=subs1.x, edge_index_s=subs1.edge_index,
                                    edge_attr_s=subs1.edge_attr, n_sub_batch_s=torch.tensor([node_i]*subs1.num_nodes, dtype=int),
                                    x_t=subs2.x, edge_index_t=subs2.edge_index,
                                    edge_attr_t=subs2.edge_attr, n_sub_batch_t=torch.tensor([node_i]*subs2.num_nodes, dtype=int))
                if not torch.unique(new_data.n_sub_batch_s).shape[0] == torch.unique(new_data.n_sub_batch_t).shape[0]:
                    raise Exception('Data is unequal!!')

        return new_data
    
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

