from torch_geometric.data import Dataset, Data
import torch
import torch_geometric
import os
import squidpy as sq
import pandas as pd
import numpy as np
from anndata import AnnData
from skimage import io
import torchvision.transforms as T

class GeoMXDataset(Dataset):
    def __init__(self, root_dir='data/', pre_transform=None, pre_filter=None):
        self.root_dir = os.path.join(os.getcwd(), root_dir)
        self.raw_path = os.path.join(self.root_dir, 'raw/TMA1_prepocessed')
        self.processed_path = os.path.join(self.root_dir, 'processed')


        if os.path.exists(self.raw_path) and os.path.isdir(self.raw_path):
            self.raw_files = [os.path.join(self.raw_path, p) for p in os.listdir(self.raw_path) if p.endswith('_embed.pt')]
            self.raw_files.sort()
            self.cell_pos = [os.path.join(self.raw_path, p) for p in os.listdir(self.raw_path) if p.endswith('.csv')][0]
        
        self.string_labels_map = {}

        super().__init__(self.root_dir, self.transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self.raw_files
    
    @property
    def processed_file_names(self):
        """ return list of files should be in processed dir, if found - skip processing."""
        processed_filename = []
        for i, _ in enumerate(self.raw_files):
            processed_filename.append(f"graph_{i:03}.pt")
        processed_filename.sort()
        return processed_filename
    
    def transform(self, data):
        return data   
    
    def download(self):
        pass

    def process(self):
        label = pd.read_csv(self.raw_dir, header=0, sep=',')
        df = pd.read_csv(self.cell_pos, header=0, sep=",")
        df['Centroid X px'] = df['Centroid X px'].round().astype(int)
        df['Centroid Y px'] = df['Centroid Y px'].round().astype(int)
        for file in self.raw_paths:
            self._process_one_step(file, df, label)

    def _process_one_step(self, file, df, label):
        file_prefix = file.split('/')[-1].split('_')[0]
        df = df[df['Image']==file_prefix+'.tiff']
        df = df.drop("Image", axis=1)

        counts = np.zeros(size=(df.shape[0], 1))
        coordinates = np.column_stack((df["Centroid X px"].to_numpy(), df["Centroid Y px"].to_numpy()))
        adata = AnnData(counts, obsm={"spatial": coordinates})
        sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
        edge_matrix = adata.obsp["spatial_connectivities"] * (adata.obsp["spatial_connectivities"] <= 50)
        edge_index, edge_attr = torch_geometric.utils.convert.from_scipy_sparse_matrix(edge_matrix)

        # label = df["Class"].values
        # string_labels = list(df["Class"].unique())
        # if len(self.string_labels_map.keys()) == 0:
        #     int_labels = list(range(len(string_labels)))
        #     self.string_labels_map = dict(zip(string_labels, int_labels))
        #     label = torch.tensor(np.vectorize(self.string_labels_map.get)(label))
        # else:
        #     keys = list(self.string_labels_map.keys())
        #     for l in string_labels:
        #         if l not in keys:
        #             self.string_labels_map[l] = len(keys)
        #             keys.append(l)
        #     label = torch.tensor(np.vectorize(self.string_labels_map.get)(label))
        

        node_features =torch.load(file)

        label = label[label['ROI']==file_prefix.lstrip('0')]
        label = torch.from_numpy(label.iloc[:,2:].sum().to_numpy())
        
        data = Data(x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=label
                    )
        torch.save(data, os.path.join(self.processed_path, f"graph_{file_prefix}.pt"))


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        return data

