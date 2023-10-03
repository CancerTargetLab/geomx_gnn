from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import os
import torchvision.transforms as T
import random

class EmbedDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir="data/raw", crop_factor=0.5):
        """
        Arguments:
        """
        self.root_dir = os.path.join(os.getcwd(), root_dir)
        self.crop_factor = crop_factor

        cells_path = [os.path.join(self.root_dir, p) for p in os.listdir(self.root_dir) if p.endswith('_cells.pt')]

        self.data = torch.Tensor()

        for cells in cells_path:
            data = torch.load(cells)
            self.data = torch.cat((self.data, data))

    def transform(self, data):
        x_lower = int(self.crop_factor * data.shape[-1])
        y_lower = int(self.crop_factor * data.shape[-2])
        # Generate random integers for x and y within the specified range
        random_x = random.randint(x_lower, data.shape[-1])
        random_y = random.randint(y_lower, data.shape[-2])
        
        compose = T.Compose([
            T.RandomCrop((random_y, random_x)),
            T.Resize((data.shape[-1], data.shape[-2])),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip()
        ])
        x1, x2 = compose(data), compose(data)
        return x1, x2

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.transform(self.data[idx])
    
