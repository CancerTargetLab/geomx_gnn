import torch
import os
import torchvision.transforms as T
import random
from src.data.GeoMXData import GeoMXDataset

class ImageGraphData(GeoMXDataset):
    def __init__(self, 
                 root_dir='data/',
                 raw_subset_dir='',
                 train_ratio=0.6,
                 val_ratio=0.2,
                 node_dropout=0.2,
                 edge_dropout=0.3,
                 pixel_pos_jitter=40,
                 n_knn=6,
                 subgraphs_per_graph=0,
                 num_hops=10,
                 label_data='label_data.csv',
                 transform=None):
        super().__init__(root_dir=root_dir,
                        raw_subset_dir=raw_subset_dir,
                        train_ratio=train_ratio,
                        val_ratio=val_ratio,
                        node_dropout=node_dropout,
                        edge_dropout=edge_dropout,
                        pixel_pos_jitter=pixel_pos_jitter,
                        n_knn=n_knn,
                        subgraphs_per_graph=subgraphs_per_graph,
                        num_hops=num_hops,
                        label_data=label_data,
                        transform=transform,
                        use_embed_image=False)
        self.data = [torch.load(os.path.join(self.processed_dir, self.data[graph])) for graph in self.data]
    
    def transform(self, data):
        x_lower = int(self.crop_factor * data.shape[-1])
        y_lower = int(self.crop_factor * data.shape[-2])
        # Generate random integers for x and y within the specified range
        random_x = random.randint(x_lower, data.shape[-1])
        random_y = random.randint(y_lower, data.shape[-2])
        
        compose = T.Compose([
            T.RandomCrop((random_y, random_x)),
            T.Resize((data.shape[-1], data.shape[-2]), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=90),
            #T.GaussianBlur(kernel_size=(3,3), sigma=(0.0, 5.))
        ])
        if self.mode == self.train:
            data.x = compose(data.x.to(torch.float32))
        
        return super().transform(data)

    def get(self, idx):
        if self.mode == self.train:
            return self.data[self.train_map][idx]
        elif self.mode == self.val:
            return self.processed_dir, self.data[self.val_map][idx]
        elif self.mode == self.test:
            return self.processed_dir, self.data[self.test_map][idx]
        else:
            return self.processed_dir, self.data[idx]