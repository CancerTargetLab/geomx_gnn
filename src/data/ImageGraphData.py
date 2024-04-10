import torch
import numpy as np
import os
from tqdm import tqdm
import torchvision.transforms as T
import random
from src.data.GeoMXData import GeoMXDataset

class ImageGraphDataset(GeoMXDataset):
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
                 transform=None,
                 crop_factor=0.5):
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
        self.crop_factor = crop_factor
        self.data_path = self.data
        self.data_idx = np.array(list(range(self.data.shape[0])))
        self.data = [torch.load(os.path.join(self.processed_dir, graph)) for graph in self.data]
    
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
            return self.data[self.data_idx[self.train_map][idx]]
        elif self.mode == self.val:
            return self.data[self.data_idx[self.val_map][idx]]
        elif self.mode == self.test:
            return self.data[self.data_idx[self.test_map][idx]]
        else:
            return self.data[idx]
    
    def embed(self, model, path, device='cpu', batch_size=256, return_mean=False):
        del self.data
        self.data = np.ndarray((len(self.data_path)), dtype=str)  #Needed as parent class uses self.data np array of str paths to load data

        self.graph_embed_path = os.path.join(self.processed_path, 'embed')
        if not (os.path.exists(self.graph_embed_path) and os.path.isdir(self.graph_embed_path)):
            os.makedirs(self.graph_embed_path)

        with torch.no_grad():
            with tqdm(self.data_path, total=len(self.data_path), desc='Save Image embedings') as data_path:
                for i, dpath in enumerate(data_path):
                    data = torch.load(os.path.join(dpath))
                    embed = torch.zeros((data.x.shape[0], model.embed_size), dtype=torch.float32)
                    num_batches = (data.shape[0] // batch_size) + 1
                    for batch_idx in range(num_batches):
                        if batch_idx < num_batches - 1:
                            embed[batch_idx*batch_size:batch_idx*batch_size+batch_size] = model.image.forward(data.x[batch_idx*batch_size:batch_idx*batch_size+batch_size].to(device, torch.float32)).to('cpu')
                        else:
                            embed[batch_idx*batch_size:] = model(data.x[batch_idx*batch_size:].to(device, torch.float32)).to('cpu')
                    data.x = embed
                    torch.save(data, os.path.join(self.graph_embed_path, dpath.split('/')[-1].split('.')[0]+'_embed.pt'))
                    self.data[i] = os.path.join(self.graph_embed_path, dpath.split('/')[-1].split('.')[0]+'_embed.pt')
        super().embed(model=model.graph, path=path, device=device, return_mean=return_mean)

