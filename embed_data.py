from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import os
import torchvision.transforms as T

class EmbedDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir="data/raw", embed_size=2048, train=0.6, test=0.2, val=0.2):
        """
        Arguments:
        """
        self.root_dir = os.path.join(os.getcwd(), root_dir)

        cells_path = [os.path.join(self.root_dir, p) for p in os.listdir(self.root_dir) if p.endswith('_cells.pt')]

        self.data = torch.Tensor()

        for cells in cells_path:
            data = torch.load(cells)
            self.data = torch.cat((self.data, data))

    def transform(self, data):
        compose = T.Compose([
            T.RandomCrop(),
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
    
