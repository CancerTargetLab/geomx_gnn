from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import RandomJitter, KNNGraph, Distance, LocalCartesian, RootedEgoNets
import torch
import torch_geometric
import os
import squidpy as sq
import pandas as pd
import numpy as np
from anndata import AnnData
from tqdm import tqdm

class GeoMXDataset(Dataset):
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
        self.root_dir = os.path.join(os.getcwd(), root_dir)
        self.raw_path = os.path.join(self.root_dir, 'raw', raw_subset_dir)
        self.processed_path = os.path.join(self.root_dir, 'processed', raw_subset_dir)
        self.label_data = label_data
        self.raw_subset_dir = raw_subset_dir

        self.node_dropout = node_dropout
        self.edge_dropout = edge_dropout
        self.pixel_pos_jitter = pixel_pos_jitter
        self.n_knn = n_knn
        self.subgraphs_per_graph = subgraphs_per_graph
        self.num_hops = num_hops

        self.RandomJitter = RandomJitter(self.pixel_pos_jitter)
        self.KNNGraph = KNNGraph(k=self.n_knn, force_undirected=True)
        self.Distance = Distance(norm=False, cat=False)
        self.LocalCartesian = LocalCartesian()

        if not (os.path.exists(self.processed_path) and os.path.isdir(self.processed_path)):
            os.makedirs(self.processed_path)

        if os.path.exists(self.raw_path) and os.path.isdir(self.raw_path):
            self.cell_pos = [os.path.join(self.raw_path, p) for p in os.listdir(self.raw_path) if p.endswith('.csv')][0]
            self.raw_files = pd.read_csv(self.cell_pos, header=0, sep=',')['Image'].apply(lambda x: x.split('.')[0]+'_cells_embed.pt').unique().tolist()
            self.raw_files = [os.path.join(self.raw_path, p) for p in self.raw_files]
            self.raw_files.sort()
        
        image_name_split = pd.read_csv(self.cell_pos, header=0, sep=',')['Image'].iloc[0].split('.')
        self.image_ending = ''
        for i in range(len(image_name_split)-1):
            self.image_ending = self.image_ending + '.' + image_name_split[i+1]

        super().__init__(self.root_dir, self.transform if transform is None else transform, None, None)

        self.data = np.array(self.processed_file_names)

        df = pd.read_csv(os.path.join(self.raw_dir, self.label_data), header=0, sep=',')
        IDs = np.array(df[~df.duplicated(subset=['ROI'], keep=False) | ~df.duplicated(subset=['ROI'], keep='first')].sort_values(by=['ROI'])['Patient_ID'].values)
        un_IDs = np.unique(IDs)

        total_samples = un_IDs.shape[0]
        train_size = int(train_ratio * total_samples)
        val_size = int(val_ratio * total_samples)
        test_size = total_samples - train_size - val_size

        train_map, val_map, test_map = torch.utils.data.random_split(torch.arange(total_samples), [train_size, val_size, test_size])
        self.train_map, self.val_map, self.test_map = np.argwhere(np.isin(IDs, un_IDs[train_map.indices])).squeeze().tolist(), np.argwhere(np.isin(IDs, un_IDs[val_map.indices])).squeeze().tolist(), np.argwhere(np.isin(IDs, un_IDs[test_map.indices])).squeeze().tolist()

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
        for i, path in enumerate(self.raw_files):
            i += 1
            appendix = path.split('/')[-1].split('_')[0]
            if len(self.raw_subset_dir) > 0:
                processed_filename.append(f'{self.raw_subset_dir}/graph_{appendix}.pt')
            else:
                processed_filename.append(f'graph_{appendix}.pt')
        processed_filename.sort()
        return processed_filename
    
    def transform(self, data):
        if self.subgraphs_per_graph > 0:
            sub = torch.randint(0, data.num_nodes, (int(torch.min(torch.Tensor([data.num_nodes, self.subgraphs_per_graph]))),), device=data.edge_index.device)
            subset, edge_index, mapping, edge_mask = torch_geometric.utils.k_hop_subgraph(sub, self.num_hops, data.edge_index, relabel_nodes=True, directed=False)
            data = Data(x=data.x[subset], edge_index=edge_index, edge_attr=data.edge_attr[edge_mask], pos=data.pos[subset], y=torch.sum(data.cellexpr[subset], axis=0))
        if self.mode==self.train:
            y = data.y
            data.edge_index = torch.Tensor([])
            data = self.RandomJitter(data)
            data = self.KNNGraph(data)
            data = self.Distance(data)
            node_map = torch_geometric.utils.dropout_node(data.edge_index, p=self.node_dropout, training=self.mode==self.train)[1]
            data.edge_index, data.edge_attr = data.edge_index[:,node_map], data.edge_attr[node_map]
            edge_map = torch_geometric.utils.dropout_edge(data.edge_index, p=self.edge_dropout, training=self.mode==self.train)[1]
            data.edge_index, data.edge_attr = data.edge_index[:,edge_map], data.edge_attr[edge_map]
            data = torch_geometric.transforms.AddRemainingSelfLoops(attr='edge_attr', fill_value=0.0)(data)
            data.y = y
        #data = self.LocalCartesian(data)
        return data   
    
    def download(self):
        pass

    def process(self):
        label = pd.read_csv(os.path.join(self.raw_dir, self.label_data), header=0, sep=',')
        df = pd.read_csv(self.cell_pos, header=0, sep=',')
        df['Centroid.X.px'] = df['Centroid.X.px'].astype(np.float32)
        df['Centroid.Y.px'] = df['Centroid.Y.px'].astype(np.float32)
        with tqdm(self.raw_paths, total=len(self.raw_paths), desc='Preprocessing Graphs') as raw_paths:
            for file in raw_paths:
                self._process_one_step(file, df, label)

    def _process_one_step(self, file, df, label):
        file_prefix = file.split('/')[-1].split('_')[0]
        df = df[df['Image']==file_prefix+self.image_ending]
        # Deduplicate identical cell position: ~ is not op, first selects duplicates, second selects non first duplicates, | is or op
        mask = ~df.duplicated(subset=['Centroid.X.px', 'Centroid.Y.px'], keep=False) | ~df.duplicated(subset=['Centroid.X.px', 'Centroid.Y.px'], keep='first')
        df = df[mask]

        counts = np.zeros((df.shape[0], 1))
        coordinates = np.column_stack((df["Centroid.X.px"].to_numpy(), df["Centroid.Y.px"].to_numpy()))
        adata = AnnData(counts, obsm={"spatial": coordinates})
        sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=self.n_knn)
        edge_matrix = adata.obsp["spatial_distances"]
        edge_index, edge_attr = torch_geometric.utils.convert.from_scipy_sparse_matrix(edge_matrix)

        node_features =torch.load(file)[torch.from_numpy(mask.values)]

        label = label[label['ROI']==file_prefix]
        label = torch.from_numpy(label.iloc[:,2:].sum().to_numpy()).to(torch.float32)
        cellexpr = label.clone()
        if df.columns.shape[0] > 4:
            cellexpr = torch.from_numpy(df[df.columns[4:].values].values).to(torch.float32)
        if torch.sum(label) > 0:
            if 'Class' in df.columns:
                data = Data(x=node_features,
                        edge_index=edge_index,
                        edge_attr=edge_attr.to(torch.float32),
                        y=label,
                        pos=torch.from_numpy(coordinates).to(torch.float32),
                        Class=df['Class'].values,
                        cellexpr=cellexpr)
            else:
                data = Data(x=node_features,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            pos=torch.from_numpy(coordinates).to(torch.float32),
                            y=label,
                            cellexpr=cellexpr)
            data = torch_geometric.transforms.AddRemainingSelfLoops(attr='edge_attr', fill_value=0.0)(data)
            torch.save(data, os.path.join(self.processed_path, f"graph_{file_prefix}.pt"))
        else: 
            raise Exception(f'File {file} has no Expression data in {self.label_data}!!!')

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
    
    def embed(self, model, path, device='cpu', return_mean=False):
        with torch.no_grad():
            model = model.to(device)
            with tqdm(self.data.tolist(), total=self.data.shape[0], desc='Creating ROI embeddings') as data:
                for graph_path in data:
                    graph = torch.load(os.path.join(self.processed_dir, graph_path))
                    roi_pred = model(graph.to(device))
                    roi_pred = roi_pred[0] if isinstance(roi_pred, tuple) else roi_pred
                    cell_pred = model(graph.to(device), return_cells=True, return_mean=return_mean)
                    torch.save(roi_pred, os.path.join(path, 'roi_pred_'+graph_path.split('/')[-1]))
                    torch.save(cell_pred, os.path.join(path, 'cell_pred_'+graph_path.split('/')[-1]))

