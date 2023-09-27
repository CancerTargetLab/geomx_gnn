from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import os
import re

class EmbedDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir="data/raw", embed_size=2048, train=0.6, test=0.2, val=0.2):
        """
        Arguments:
        """
        self.root_dir = os.path.join(os.getcwd(), root_dir)
        self.embed_size = embed_size
        self.train_split = train
        self.test_split = test
        self.val_split = val

        dirs = os.listdir(self.root_dir)

        pattern = r".*_z.*"
        self.sample_dirs = [os.path.join(self.root_dir, s) for s in dirs if re.match(pattern, s)]

        self.data = torch.Tensor()

        if os.path.exists(os.path.join(self.root_dir, "data.pt")):
            self.data = torch.cat((self.data, torch.load(os.path.join(self.root_dir, "data.pt"))), axis=0)
        else:
            for dir in self.sample_dirs:
                print(f"Loading samples from {dir}...")
                files = os.listdir(dir)
                for file in files:
                    data = torch.load(os.path.join(dir, file))
                    self.data = torch.cat((self.data, data), axis=0)
            torch.save(self.data, os.path.join(self.root_dir, "data.pt"))
        print("Done loading")

        train_data, temp_data = train_test_split(self.data, test_size=(1 - self.train_split), random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=(self.test_split / (self.test_split + self.val_split)), random_state=42)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.mode = "TRAIN"
        self.train = "TRAIN"
        self.test = "TEST"
        self.val = "VAL"


    def setMode(self, mode):
        if mode.upper() == self.train:
            self.mode = mode.upper()
        elif mode.upper() == self.val:
            self.mode = mode.upper()
        elif mode.upper() == self.test:
            self.mode = mode.upper()
        else:
            raise "Unknown mode, mode has to be one of 'train', 'test' or 'val'."

    def __len__(self):
        if self.mode == self.train:
            length = self.train_data.shape[0]
        elif self.mode == self.val:
            length = self.val_data.shape[0]
        else:
            length = self.test_data.shape[0]
        return length

    def __getitem__(self, idx):
        if self.mode == self.train:
            data = self.train_data[idx]
        elif self.mode == self.val:
            data = self.val_data[idx]
        else:
            data = self.test_data[idx]
        return data
    
