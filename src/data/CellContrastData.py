from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import torchvision.transforms as T
from tqdm import tqdm

class AddGaussianNoiseToRandomChannels(object):
    """
    Add GaussianNoise with per channel random chance.
    # Obtained and adapted from ptrblck:
    # https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745/2
    """
    def __init__(self, mean=0., std=1., p=0.5):
        """
        Init state.

        Paramters:
        mean (float): mean
        std (float): std
        p (float): per channel chance to add noise
        """
        self.std = std
        self.mean = mean
        self.p = p
        
    def __call__(self, tensor):
        for channel in range(tensor.shape[0]):
            if  (torch.randn(1) < self.p).item():
                tensor[channel] = tensor[channel] + torch.randn(tensor[channel].size()) * self.std + self.mean
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, p={2})'.format(self.mean, self.std, self.p)


class EmbedDataset(Dataset):
    """
    Dataset of tiffile zscore normalized, per ROI cut out cells.
    """

    def __init__(self,
                 root_dir="data/raw",
                 crop_factor=0.5,
                 train_ratio=0.6,
                 val_ratio=0.2):
        """
        Init dataset.

        Parameters:
        root_dir (str): Path to dir containing zscore normalized, per ROI cut out cells
        crop_factor (float): Min crop size
        train_ratio (float): Ratio of cells used for training
        val_ratio (float): Ratio of cells used for validation
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
        """
        Set mode of dataset.

        Parameters:
        mode (str): mode of dataset to set to
        """
        if mode.upper() in [self.train, self.val, self.test, self.embed]:
            self.mode = mode.upper()
        else:
            print(f'Mode {mode} not suported, has to be one of .train, .val .test or .embed')


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

        compose = T.Compose([
            T.RandomResizedCrop(size=(data.shape[-1], data.shape[-2]), scale=(self.crop_factor, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomErasing(value=0),
            AddGaussianNoiseToRandomChannels(),
            rnd_gausblur
        ])
        x1, x2 = compose(data.to(torch.float32)), compose(data.to(torch.float32))
        return x1, x2

    def __len__(self):
        """
        Set mode of dataset.
        """
        if self.mode == self.train:
            return len(self.train_map)
        elif self.mode == self.val:
            return len(self.val_map)
        elif self.mode == self.test:
            return len(self.test_map)
        else:
            return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Get specific cell cut out.

        Parameters:
        idx (int): index

        Returns:
        torch.Tensor, cell cut out
        """
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
                    data = torch.load(os.path.join(path))
                    embed = torch.zeros((data.shape[0], model.embed_size), dtype=torch.float32)
                    num_batches = (data.shape[0] // batch_size) + 1
                    for batch_idx in range(num_batches):
                        if batch_idx < num_batches - 1:
                            embed[batch_idx*batch_size:batch_idx*batch_size+batch_size] = model(data[batch_idx*batch_size:batch_idx*batch_size+batch_size].to(device, torch.float32)).to('cpu')
                        else:
                            embed[batch_idx*batch_size:] = model(data[batch_idx*batch_size:].to(device, torch.float32)).to('cpu')
                    torch.save(embed, os.path.join(path, path.split('.')[0]+'_embed.pt'))

