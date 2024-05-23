import torch
import numpy as np
import os
from tqdm import tqdm
import torchvision.transforms as T
import random
from src.data.GeoMXData import GeoMXDataset
from src.data.CellContrastData import AddGaussianNoiseToRandomChannels

class ImageGraphDataset(GeoMXDataset):
    """
    Datset of Cell Graphs containing cell cutouts per ROI/Graph.
    """
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
                 crop_factor=0.5,
                 embed=False):
        """
        Init dataset.

        root_dir (str): Path to dir containing raw/ and processed dir
        raw_subset_dir (str): Name of dir in raw/ and processed/ containing  per ROI visual cell representations(in raw/)
        train_ratio (float): Ratio of IDs used for training
        val_ratio (float): Ratio of IDs used for validation
        node_dropout (float): Chance of node dropout during training
        edge_dropout (float): Chance of edge dropout during training
        pixel_pos_jitter (int): Positional jittering of nodes during training
        n_knn (int): Number of Nearest Neighbours to calculate for each cell and create edges to
        subgraphs_per_graph (int): Number of ~equally distributed subgraphs per ROI to create, use when observable SC data exists
        num_hops (int): Number of hops to create subgraphs from centoid cell
        label_data (str): .csv name in raw/ dir contaiing ROI label data
        transform (None): -
        crop_factor (float): Min cell cut out image crop for transformation
        embed (bool): Wether or not to load graphs during runtime or have them loaded in memory
        """
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
        if not embed:
            self.data = [torch.load(os.path.join(self.processed_dir, graph)) for graph in self.data]
    
    def transform(self, data):
        """"
        Transform graph if training.

        Paramters:
        data (torch_geometric.data.Data): Graph

        Returns:
        torch_geometric.data.Data: Graph
        """
        def img_transform(data):
            """"
            Create transformed views of all Images.

            Paramters:
            data (torch.Tensor): Cell Images

            Returns:
            torch.Tensor: Cell Images transformed
            """
            gausblur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 3.))
            rnd_gausblur = T.RandomApply([gausblur], p=0.5)
            for img in range(data.shape[0]):
                compose = T.Compose([
                    T.RandomResizedCrop(size=(data.shape[-1], data.shape[-2]), scale=(self.crop_factor, 1.0), antialias=True),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomErasing(value=0),
                    #AddGaussianNoiseToRandomChannels(),
                    rnd_gausblur
                ])
                data[img] = compose(data[img])
            return data
        data.x = data.x.to(torch.float32)
        if self.mode == self.train:
            data.x = img_transform(data.x)
        
        return super().transform(data)

    def get(self, idx):
        """
        Get Graph self.data[idx] depending on mode.

        Parameters:
        idx (int): index

        Returns:
        torch_geometric.data.Data, Cell Graph
        """
        if self.mode == self.train:
            return self.data[self.data_idx[self.train_map][idx]]
        elif self.mode == self.val:
            return self.data[self.data_idx[self.val_map][idx]]
        elif self.mode == self.test:
            return self.data[self.data_idx[self.test_map][idx]]
        else:
            return self.data[idx]
    
    def embed(self, model, path, device='cpu', batch_size=256, return_mean=False):
        """
        Save model sc expression of all cells per ROI and Image Representations.

        model (torch.Module): model
        path (str): Dir to save ROI sc expression to
        device (str): device to operate on
        batch_size (int): Number of cells to extract representations from at once
        return_mean (bool): Wether or not ZINB/NB models return predicted mean of Genes/Proteins per cell
        """
        del self.data
        path_list = []  #Needed as parent class uses self.data np array of str paths to load data

        self.graph_embed_path = os.path.join(self.processed_path, 'embed')
        if not (os.path.exists(self.graph_embed_path) and os.path.isdir(self.graph_embed_path)):
            os.makedirs(self.graph_embed_path)

        model = model.to(device)

        with torch.no_grad():
            with tqdm(self.data_path, total=len(self.data_path), desc='Save Image embedings') as data_path:
                for i, dpath in enumerate(data_path):
                    data = torch.load(os.path.join(self.processed_dir, dpath))
                    embed = torch.zeros((data.x.shape[0], model.image.embed_size), dtype=torch.float32)
                    num_batches = (data.x.shape[0] // batch_size) + 1
                    for batch_idx in range(num_batches):
                        if batch_idx < num_batches - 1:
                            embed[batch_idx*batch_size:batch_idx*batch_size+batch_size] = model.image.forward(data.x[batch_idx*batch_size:batch_idx*batch_size+batch_size].to(device, torch.float32)).to('cpu')
                        else:
                            embed[batch_idx*batch_size:] = model.image.forward(data.x[batch_idx*batch_size:].to(device, torch.float32)).to('cpu')
                    data.x = embed
                    torch.save(data, os.path.join(self.graph_embed_path, dpath.split('/')[-1].split('.')[0]+'_embed.pt'))
                    path_list.append(os.path.join(self.graph_embed_path, dpath.split('/')[-1].split('.')[0]+'_embed.pt'))
                self.data = np.array(path_list)
        super().embed(model=model.graph, path=path, device='cpu', return_mean=return_mean)

