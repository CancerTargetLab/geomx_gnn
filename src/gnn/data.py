from torch_geometric.data import Dataset, Data
import torch
import torch_geometric
import os
import squidpy as sq
import pandas as pd
import numpy as np
from anndata import AnnData
from tqdm import tqdm

class GeoMXDataset(Dataset):
    def __init__(self, root_dir='data/', raw_subset_dir='',
                 train_ratio = 0.6, val_ratio = 0.2, node_dropout=0.2,
                 edge_dropout=0.3):
        self.root_dir = os.path.join(os.getcwd(), root_dir)
        self.raw_path = os.path.join(self.root_dir, 'raw', raw_subset_dir)
        self.processed_path = os.path.join(self.root_dir, 'processed')

        self.node_dropout = node_dropout
        self.edge_dropout = edge_dropout

        if os.path.exists(self.raw_path) and os.path.isdir(self.raw_path):
            self.raw_files = [os.path.join(self.raw_path, p) for p in os.listdir(self.raw_path) if p.endswith('_embed.pt')]
            self.raw_files.sort()
            self.cell_pos = [os.path.join(self.raw_path, p) for p in os.listdir(self.raw_path) if p.endswith('.csv')][0]
        
        self.string_labels_map = {}

        super().__init__(self.root_dir, self.transform, None, None)

        self.data = np.array(self.processed_file_names)

        total_samples = self.data.shape[0]
        train_size = int(train_ratio * total_samples)
        val_size = int(val_ratio * total_samples)
        test_size = total_samples - train_size - val_size

        # Use random_split to split the data tensor
        train_map, val_map, test_map = torch.utils.data.random_split(torch.arange(total_samples), [train_size, val_size, test_size])
        self.train_map, self.val_map, self.test_map = train_map.indices, val_map.indices, test_map.indices


        self.mode = 'TRAIN'
        self.train = 'TRAIN'
        self.val = 'VAL'
        self.test = 'TEST'

    @property
    def raw_file_names(self):
        return self.raw_files
    
    @property
    def processed_file_names(self):
        """ return list of files should be in processed dir, if found - skip processing."""
        processed_filename = []
        for i, _ in enumerate(self.raw_files):
            i += 1
            processed_filename.append(f"graph_{i:03}.pt")
        processed_filename.sort()
        return processed_filename
    
    def transform(self, data):
        node_map = torch_geometric.utils.dropout_node(data.edge_index, p=self.node_dropout, training=self.mode==self.train)[1]
        data.edge_index, data.edge_attr = data.edge_index[:,node_map], data.edge_attr[node_map]
        edge_map = torch_geometric.utils.dropout_edge(data.edge_index, p=self.edge_dropout, training=self.mode==self.train)[1]
        data.edge_index, data.edge_attr = data.edge_index[:,edge_map], data.edge_attr[edge_map]
        data = torch_geometric.transforms.RemoveIsolatedNodes()(data)
        return data   
    
    def download(self):
        pass

    def process(self):
        label = pd.read_csv(os.path.join(self.raw_dir, 'OC1_all.csv'), header=0, sep=',')
        df = pd.read_csv(self.cell_pos, header=0, sep=",")
        df['Centroid X px'] = df['Centroid X px'].round().astype(float)
        df['Centroid Y px'] = df['Centroid Y px'].round().astype(float)
        with tqdm(self.raw_paths, total=len(self.raw_paths), desc='Preprocessing Graphs') as raw_paths:
            for file in raw_paths:
                self._process_one_step(file, df, label)

    def _process_one_step(self, file, df, label):
        file_prefix = file.split('/')[-1].split('_')[0]
        df = df[df['Image']==file_prefix+'.tiff']
        # Deduplicate identical cell position: ~ is not op, first selects duplicates, second selects non first duplicates, | is or op
        mask = ~df.duplicated(subset=['Centroid X px', 'Centroid Y px'], keep=False) | ~df.duplicated(subset=['Centroid X px', 'Centroid Y px'], keep='first')
        df = df[mask]

        counts = np.zeros((df.shape[0], 1))
        coordinates = np.column_stack((df["Centroid X px"].to_numpy(), df["Centroid Y px"].to_numpy()))
        adata = AnnData(counts, obsm={"spatial": coordinates})
        sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
        edge_matrix = adata.obsp["spatial_distances"]
        edge_matrix[edge_matrix > 60] = 0.
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
        

        node_features =torch.load(file)[torch.from_numpy(mask.values)]

        label = label[label['ROI']==int(file_prefix.lstrip('0'))]
        label = torch.from_numpy(label.iloc[:,2:].sum().to_numpy())
        
        data = Data(x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=label
                    )
        data = torch_geometric.transforms.AddRemainingSelfLoops(attr='edge_attr', fill_value=0.1)(data)
        torch.save(data, os.path.join(self.processed_path, f"graph_{file_prefix}.pt"))

    def setMode(self, mode):
        if mode.upper() in [self.train, self.val, self.test]:
            self.mode = mode.upper()
        else:
            print(f'Mode {mode} not suported, has to be one of .train, .val .test or .embed')

    def len(self):
        if self.mode == self.train:
            return len(self.train_map)
        elif self.mode == self.val:
            return len(self.val_map)
        elif self.mode == self.test:
            return len(self.test_map)
        else:
            return self.data.shape[0]

    def get(self, idx):
        if self.mode == self.train:
            return torch.load(os.path.join(self.processed_dir, self.data[self.train_map][idx]))
        elif self.mode == self.val:
            return torch.load(os.path.join(self.processed_dir, self.data[self.val_map][idx]))
        elif self.mode == self.test:
            return torch.load(os.path.join(self.processed_dir, self.data[self.test_map][idx]))
        else:
            return torch.load(os.path.join(self.processed_dir, self.data[idx]))

