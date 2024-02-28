from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import torchvision.transforms as T
import random
from tqdm import tqdm

class EmbedDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir="data/raw", crop_factor=0.5, train_ratio = 0.6,
                 val_ratio = 0.2):
        """
        Arguments:
        """
        self.root_dir = os.path.join(os.getcwd(), root_dir)
        self.crop_factor = crop_factor

        self.cells_path = [os.path.join(self.root_dir, p) for p in os.listdir(self.root_dir) if p.endswith('_cells.pt')]
        self.cells_path.sort()

        csv_path = [os.path.join(self.root_dir, p) for p in os.listdir(self.root_dir) if p.endswith('.csv')][0]
        self.cell_number = pd.read_csv(csv_path, header=0, sep=',').shape[0]
        img_shape = torch.load(self.cells_path[0]).shape
        self.data = torch.zeros((self.cell_number, img_shape[1], img_shape[2], img_shape[3]), dtype=torch.float16)

        last_idx = 0
        for cells in self.cells_path:
            data = torch.load(cells)
            self.data[last_idx:data.shape[0]+last_idx] = data
            last_idx += data.shape[0]
        
        total_samples = self.data.shape[0]
        train_size = int(train_ratio * total_samples)
        val_size = int(val_ratio * total_samples)
        test_size = total_samples - train_size - val_size

        # Use random_split to split the data tensor
        train_map, val_map, test_map = torch.utils.data.random_split(self.data, [train_size, val_size, test_size])
        self.train_map, self.val_map, self.test_map = train_map.indices, val_map.indices, test_map.indices

        self.train_data = self.data[self.train_map]
        self.val_data = self.data[self.val_map]
        self.test_data = self.data[self.test_map]

        self.mode = 'TRAIN'
        self.train = 'TRAIN'
        self.val = 'VAL'
        self.test = 'TEST'
        self.embed = 'EMBED'
    
    def setMode(self, mode):
        if mode.upper() in [self.train, self.val, self.test, self.embed]:
            self.mode = mode.upper()
        else:
            print(f'Mode {mode} not suported, has to be one of .train, .val .test or .embed')


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
        x1, x2 = compose(data.to(torch.float32)), compose(data.to(torch.float32))
        return x1, x2

    def __len__(self):
        if self.mode == self.train:
            return len(self.train_map)
        elif self.mode == self.val:
            return len(self.val_map)
        elif self.mode == self.test:
            return len(self.test_map)
        else:
            return self.data.shape[0]

    def __getitem__(self, idx):
        if self.mode == self.train:
            return self.transform(self.train_data[idx])
        elif self.mode == self.val:
            return self.transform(self.val_data[idx])
        elif self.mode == self.test:
            return self.transform(self.test_data[idx])
        elif self.mode == self.embed:
            return self.data[idx]
        else:
            return self.transform(self.data[idx])
    
    
    def save_embed_data(self, model, device='cpu'):
        with torch.no_grad():
            with tqdm(self.cells_path, total=len(self.cells_path), desc='Save embedings') as cells_path:
                for path in cells_path:
                    data = torch.load(os.path.join(path))
                    num_batches = (data.shape[0] // 256) + 1
                    for batch_idx in range(num_batches):
                        if batch_idx < num_batches - 1:
                            data[batch_idx*256:batch_idx*256+256] = model(data[batch_idx*256:batch_idx*256+256].to(device, torch.float32))
                        else:
                            data[batch_idx*256:] = model(data[batch_idx*256:].to(device, torch.float32))
                    torch.save(data, os.path.join(path, path.split('.')[0]+'_embed.pt'))

