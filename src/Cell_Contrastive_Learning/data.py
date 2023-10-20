from torch.utils.data import Dataset
import torch
import os
from torchvision.transforms import v2 as T
import random
from tqdm import tqdm

class EmbedDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir="data/raw", crop_factor=0.5, train_ratio = 0.6,
                 val_ratio = 0.2, device='cpu'):
        """
        Arguments:
        """
        self.root_dir = os.path.join(os.getcwd(), root_dir)
        self.crop_factor = crop_factor
        self.device = device

        self.cells_path = [os.path.join(self.root_dir, p) for p in os.listdir(self.root_dir) if p.endswith('_cells.pt')]
        self.cells_path.sort()

        self.data = torch.Tensor()
        self.data_index_list = [0]

        for cells in self.cells_path:
            data = torch.load(cells)
            self.data_index_list.append(data.shape[0])
            self.data = torch.cat((self.data, data))
        
        total_samples = self.data.shape[0]
        train_size = int(train_ratio * total_samples)
        val_size = int(val_ratio * total_samples)
        test_size = total_samples - train_size - val_size

        # Use random_split to split the data tensor
        train_map, val_map, test_map = torch.utils.data.random_split(self.data, [train_size, val_size, test_size])
        self.train_map, self.val_map, self.test_map = train_map.indices, val_map.indices, test_map.indices

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
            T.RandomVerticalFlip()
        ])
        x1, x2 = compose(data), compose(data)
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
            return self.transform(self.data[self.train_map][idx])
        elif self.mode == self.val:
            return self.transform(self.data[self.val_map][idx])
        elif self.mode == self.test:
            return self.transform(self.data[self.test_map][idx])
        elif self.mode == self.embed:
            return self.data[idx]
        else:
            return self.transform(self.data[idx])
    
    def save_embed_data(self, data):
        if data.shape[0] == self.data.shape[0]:
            with tqdm(self.cells_path, total=len(self.cells_path), desc='Save embedings') as cells_path:
                c_sum = 0
                for i, path in enumerate(cells_path):
                    torch.save(data[c_sum:c_sum + self.data_index_list[i+1]].to('cpu'), path.split('.')[0]+'_embed.pt')
                    c_sum += self.data_index_list[i+1]
        else:
            print('Warning: Data to save not equal number of examples as data loaded.')
    
