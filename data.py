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
    def __init__(self, dir="data/", pre_transform=None, pre_filter=None, load=None):
        self.path = os.path.join(os.getcwd(), dir)
        self.raw_path = os.path.join(self.path, "raw")
        self.processed_path = os.path.join(self.path, "processed")

        if load is not None and os.path.exists(load):
            self.load(load)
        else:
            if os.path.exists(self.raw_path) and os.path.isdir(self.raw_path):
                self.filenames = [] # List of raw files, in your case point cloud
                for file in os.listdir(self.raw_path):
                    self.filenames.append(file)
                self.raw_files = []
                self.img_file = []
                for file in self.filenames:
                    if file.endswith(".csv"):
                        self.raw_files.append(file)
                    elif file.endswith(".tiff"):
                        self.img_file.append(os.path.join(self.raw_path, file))
                self.raw_files.sort()
                self.img_file.sort()
            
            self.string_labels_map = {}
            self.cell_image_dirs = []
            self.cell_embed_dirs = []
            self.max_image_int = 0
            self.img_z = 3
            self.mean = 0
            self.std = 1
            self.num_tiffs = 0
            self.running_mean = torch.Tensor([0]*self.img_z)
            self.running_std = torch.Tensor([0]*self.img_z)

        super().__init__(self.path, self.transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self.raw_files
    
    @property
    def processed_file_names(self):
        """ return list of files should be in processed dir, if found - skip processing."""
        processed_filename = []
        for i, _ in enumerate(self.raw_files):
            processed_filename.append(f"graph_{i}.pt")
        processed_filename.sort()
        return processed_filename
    
    def load(self, load):
        attrs = torch.load(load)
        self.raw_files = attrs["raw_files"]
        self.img_file = attrs["img_file"]
        self.string_labels_map = attrs["string_labels_map"]
        self.cell_image_dirs = attrs["cell_image_dirs"]
        self.cell_embed_dirs = attrs["cell_embed_dirs"]
        self.max_image_int = attrs["max_image_int"]
        self.img_z = attrs["img_z"]
        self.mean = attrs["mean"]
        self.std = attrs["std"]
        self.num_tiffs = attrs["num_tiffs"]
        self.running_mean = attrs["running_mean"]
        self.running_std  = attrs["running_std"]

    def save(self):
        torch.save({
            "raw_files": self.raw_files,
            "img_file": self.img_file,
            "string_labels_map": self.string_labels_map,
            "cell_image_dirs": self.cell_image_dirs,
            "cell_embed_dirs": self.cell_embed_dirs,
            "max_image_int": self.max_image_int,
            "img_z": self.img_z,
            "mean": self.mean,
            "std": self.std,
            "num_tiffs": self.num_tiffs,
            "running_mean": self.running_mean,
            "running_std": self.running_std,
        }, os.path.join(self.processed_dir, 'dataset.pt'))
    
    def transform(self, data):
        nodes = data.x
        nodes = nodes/self.max_image_int
        nodes = self._image_transform()(nodes)
        data.x = nodes
        return data
    
    def _image_transform(self):
        transform = T.Compose([
            T.Normalize(self.mean/self.max_image_int, self.std/self.max_image_int),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5)
        ])
        return transform
    
    def _mean_std_calc(self):
        self.mean = self.running_mean/self.num_tiffs
        self.std = self.running_std/self.num_tiffs
    
    def cell_images(self, tiff_image_path, df):
        img = io.imread(tiff_image_path, plugin='tifffile').astype("float32")
        self.max_image_int = int(np.max(img)) if int(np.max(img)) > self.max_image_int else self.max_image_int
        img = torch.from_numpy(img)
        self.running_mean = self.running_mean + torch.mean(img, dim=(0,1))
        self.running_std = self.running_std + torch.std(img, dim=(0,1))
        self.num_tiffs += 1

        x = df["Centroid X px"].round().astype(int).values
        y = df["Centroid Y px"].round().astype(int).values

        cell_file = tiff_image_path.split('.')[0]+'.pt'
        
        all_cells = torch.Tensor()
        for cell in list(range(x.shape[0])):
            print(cell)
            delta_x1 = 10 if x[cell] >= 10 else x[cell]
            delta_y1 = 10 if y[cell] >= 10 else y[cell]
            delta_x2 = 10 if img.shape[1]-x[cell] >= 10 else img.shape[1]-x[cell]
            delta_y2 = 10 if img.shape[0]-y[cell] >= 10 else img.shape[0]-y[cell]
            cell_img = img[y[cell]-delta_y1:y[cell]+delta_y2,x[cell]-delta_x1:x[cell]+delta_x2,:]
            cell_img = T.CenterCrop(20)(torch.moveaxis(cell_img, 2, 0))
            all_cells = torch.cat((all_cells, torch.unsqueeze(cell_img, axis=0)), axis=0)
            
        torch.save(all_cells, cell_file)

        return all_cells
    
    def resnet101_embed(self, cell_dir):
        from skimage import io
        from torchvision.models import resnet101, ResNet101_Weights
        import torchvision.transforms as transforms #import ToTensor, v2

        
        for z in list(range(self.img_z)):
            self.cell_embed_dirs.append(cell_dir+f"_z{z}")
            try:
                os.mkdir(cell_dir+f"_z{z}")
            except Exception as e:
                print(e)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = resnet101(weights=ResNet101_Weights.DEFAULT).to(device)
        model.eval()

        cell_dir_list =  os.listdir(cell_dir)
        cell_dir_list.sort()

        horizontalFlip = transforms.RandomHorizontalFlip(p=1.0)
        verticalFlip = transforms.RandomVerticalFlip(p=1.0)

        with torch.no_grad():
            for file in cell_dir_list:
                cell = file.split("/")[-1].split(".")[0]+".pt"
                print(f"Saving {cell} embedings...")
                img = io.imread(os.path.join(cell_dir, file), plugin='tifffile').astype("float64")
                img = (img/self.max_image_int)*255

                for z in list(range(img.shape[2])):
                    imgi = np.expand_dims(img[:,:,z], axis=2)
                    imgi = np.concatenate((imgi, imgi, imgi), axis=2)
                    imgi = transforms.ToTensor()(imgi)
                    imgi = torch.unsqueeze(imgi, dim=0)
                    rot1 = horizontalFlip(imgi)
                    rot2 = verticalFlip(imgi)
                    imgi = torch.concat((imgi, rot1, rot2), dim=0)
                    imgi = ResNet101_Weights.DEFAULT.transforms(antialias=True)(imgi)
                    datai = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(imgi)))))))))
                    torch.save(datai[:,:,0,0], os.path.join(self.cell_embed_dirs[z], cell))

    def download(self):
        pass

    def process(self):
        for i, file in enumerate(self.raw_paths):
            self._process_one_step(file, i)
        self._mean_std_calc()
        self.save()

    def _process_one_step(self, file, index):
        df = pd.read_csv(file, header=0, sep=",")
        df = df[["Image", "Class", "Centroid X px", "Centroid Y px"]]
        df = df[df["Image"] == "001.tiff"]  #TODO: change when more images :)
        df = df.drop("Image", axis=1)
        # Replace commas with dots and convert to float
        df["Centroid X px"] = df["Centroid X px"].str.replace(',', '.').astype(float)
        df["Centroid Y px"] = df["Centroid Y px"].str.replace(',', '.').astype(float)

        counts = np.random.default_rng(42).integers(0, 15, size=(df.shape[0], 100))

        coordinates = np.column_stack((df["Centroid X px"].to_numpy(), df["Centroid Y px"].to_numpy()))

        adata = AnnData(counts, obsm={"spatial": coordinates})

        label = df["Class"].values
        string_labels = list(df["Class"].unique())
        if len(self.string_labels_map.keys()) == 0:
            int_labels = list(range(len(string_labels)))
            self.string_labels_map = dict(zip(string_labels, int_labels))
            label = torch.tensor(np.vectorize(self.string_labels_map.get)(label))
        else:
            keys = list(self.string_labels_map.keys())
            for l in string_labels:
                if l not in keys:
                    self.string_labels_map[l] = len(keys)
                    keys.append(l)
            label = torch.tensor(np.vectorize(self.string_labels_map.get)(label))
        

        sq.gr.spatial_neighbors(adata, coord_type="generic", radius=30.0)

        edge_index, edge_attr = torch_geometric.utils.convert.from_scipy_sparse_matrix(adata.obsp["spatial_connectivities"])

        node_features = self.cell_images(self.img_file[index], df)
        
        data = Data(x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=label
                    )
        torch.save(data, os.path.join(self.processed_path, f"graph_{index}.pt"))


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        return data



data = GeoMXDataset()#load="data/processed/dataset.pt")
print("")