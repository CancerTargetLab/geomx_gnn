from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import os
import torchvision.transforms.v2 as T
from tqdm import tqdm

class RandomArtefact(T.RandomErasing):
    def __init__(self,
                 p=0.5,
                 scale=(0.02, 0.33),
                 ratio=(0.3, 3.3),
                 min_value=0,
                 max_value=2**16,
                 inplace=False):
        super().__init__(p=p,
                         scale=scale,
                         ratio=ratio,
                         inplace=inplace)
        self.min_value = min_value
        self.max_value = max_value
    
    def forward(self, img):
        super().value = torch.randint(low=self.min_value, high=self.max_value)
        return super().forward(img)

class RandomBackground(torch.nn.Module):
    def __init__(self,
                 std=1,
                 std_frac=0.5,
                 p=0.5,
                 min_value=0,
                 max_value=2**16 - 1,
                 inplace=False):
        self.std = std
        self.std_frac = std_frac
        self.p = p
        self.min_value = min_value
        self.max_value = max_value
        self.inplace = inplace
    
    def forward(self, img):
        if torch.rand(1) < self.p:
            if not self.inplace:
                img = img.clone()
            background = self.std_frac * self.std * torch.randn(img.shape[-3])
            img += background.unsqueeze(1).unsqueeze(1).expand(img.shape[-3], img.shape[-2], img.shape[-1])
            img = torch.clamp(img, self.min_value, self.max_value)
        return img


class EmbedDataset(Dataset):
    """
    Dataset of tiffile zscore normalized, per ROI cut out cells.
    """

    def __init__(self,
                 root_dir="data/raw",
                 split='train',
                 crop_factor=0.5,
                 n_clusters=1):
        """
        Init dataset.

        Parameters:
        root_dir (str): Path to dir containing zscore normalized, per ROI cut out cells
        crop_factor (float): Min crop size
        train_ratio (float): Ratio of cells used for training
        val_ratio (float): Ratio of cells used for validation
        n_clusters (int): Number of KMeans clusters to calculate pseudo labels to balance sampling, ignored when 1
        """
        assert split in ['train', 'test'], f'split must be either train or test, but is {split}'
        self.root_dir = os.path.join(os.getcwd(), root_dir, split)
        self.crop_factor = crop_factor

        self.cells_path = [os.path.join(self.root_dir, p) for p in os.listdir(self.root_dir) if p.endswith('_cells.npy')] #TODO: to pt when saving torch uint16 is supported
        self.cells_path.sort()

        csv_path = [os.path.join(self.root_dir, p) for p in os.listdir(self.root_dir) if p.endswith('.csv')][0]
        self.cell_number = pd.read_csv(csv_path, header=0, sep=',').shape[0]
        #img = torch.load(self.cells_path[0]).to(torch.uint16)
        img = torch.from_numpy(np.load(self.cells_path[0]))
        img_shape = img.shape
        self.data = torch.zeros((self.cell_number, img_shape[1], img_shape[2], img_shape[3]), dtype=img.dtype)
        del img

        last_idx = 0
        for cells in self.cells_path:
            data = torch.from_numpy(np.load(cells)).to(self.data.dtype)
            #data = torch.load(cells).to(torch.uint16)
            self.data[last_idx:data.shape[0]+last_idx] = data
            last_idx += data.shape[0]
        
        self.mean = torch.from_numpy(np.load(os.path.join(self.root_dir, 'mean.npy')))
        self.std = torch.from_numpy(np.load(os.path.join(self.root_dir, 'std.npy')))

        if n_clusters > 1:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            means = self.data[:,:,int(img_shape[-2]/2),int(img_shape[-1]/2)].numpy()
            print('Calculate KMeans...')
            kmeans = KMeans(n_clusters=n_clusters, n_init=5).fit(means) #TODO: save
            print('Calculate SIL score...')
            sil = silhouette_score(means, kmeans.labels_, metric = 'euclidean')
            print(f'KMeans has SIL score of {sil}')
            n_label  = [np.sum(kmeans.labels_ == l) for l in sorted(np.unique(kmeans.labels_).tolist())]
            self.weight = [1/n_label[kmeans.labels_[i]] for i in range(kmeans.labels_.shape[0])]

    def transform(self, data):
        """"
        Create two transformed views of Image.

        Paramters:
        data (torch.Tensor): Cell Image

        Returns:
        torch.Tensor: Cell Image 1 transformed
        torch.Tensor: Cell Image 2 transformed
        """
        gausblur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 3.))
        rnd_gausblur = T.RandomApply([gausblur], p=0.5)
        gausnoise = T.GaussianNoise(clip=False)
        rnd_gausnoise = T.RandomApply([gausnoise], p=0.2)
        

        compose = T.Compose([
            T.RandomResizedCrop(size=(data.shape[-1], data.shape[-2]), scale=(self.crop_factor, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            RandomBackground(std=self.std, std_frac=0.5),
            RandomArtefact(),
            #T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=self.mean, std=self.std),
            rnd_gausnoise,
            rnd_gausblur
        ])
        x1, x2 = compose(data.to(torch.float32)), compose(data.to(torch.float32))
        return x1, x2

    def __len__(self):
        """
        Set mode of dataset.
        """
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Get specific cell cut out.

        Parameters:
        idx (int): index

        Returns:
        torch.Tensor, cell cut out
        """
        return self.transform(self.data[idx])
    
    
    def save_embed_data(self, model, device='cpu', batch_size=256):
        """
        Save model representations of all cells per ROI.

        model (torch.Module): model
        device (str): device to operate on
        batch_size (int): Number of cells to extract representations from at once
        """
        del self.data
        with torch.no_grad():
            with tqdm(self.cells_path, total=len(self.cells_path), desc='Save embedings') as cells_path:
                for path in cells_path:
                    data = T.Normalize(mean=self.mean, std=self.std)(torch.from_numpy(np.load((os.path.join(path)))))#TODO: rm
                    #data = T.Normalize(mean=self.mean, std=self.std)(torch.load(os.path.join(path)))
                    embed = torch.zeros((data.shape[0], model.embed_size), dtype=torch.float32)
                    num_batches = (data.shape[0] // batch_size) + 1
                    for batch_idx in range(num_batches):
                        if batch_idx < num_batches - 1:
                            embed[batch_idx*batch_size:batch_idx*batch_size+batch_size] = model(data[batch_idx*batch_size:batch_idx*batch_size+batch_size].to(device, torch.float32)).to('cpu')
                        else:
                            embed[batch_idx*batch_size:] = model(data[batch_idx*batch_size:].to(device, torch.float32)).to('cpu')
                    torch.save(embed, os.path.join(path, path.split('.')[0]+'_embed.pt'))

