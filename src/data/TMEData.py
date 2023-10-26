from src.data.GeoMXData import GeoMXDataset
import torch_geometric
import torch

class TMEDataset(GeoMXDataset):
    def __init__(self, root_dir='data/', raw_subset_dir='',
                 train_ratio = 0.6, val_ratio = 0.2):
        super().__init__(root_dir=root_dir, raw_subset_dir=raw_subset_dir,
                 train_ratio=train_ratio, val_ratio=val_ratio, transform=self.transform)
    
    def transform(self, data):
        subs = torch_geometric.transforms.RootedRWSubgraph(walk_length=3, repeat=1)(data)
        data.x = data.x[subs]
        return data