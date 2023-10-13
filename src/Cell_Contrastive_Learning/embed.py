from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.Cell_Contrastive_Learning.data import EmbedDataset
from src.Cell_Contrastive_Learning.CellContrastiveModel import ContrastiveLearning
from src.utils.load import load


batch_size = 256
num_workers = 8

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = EmbedDataset()
dataset.setMode(dataset.embed)
embed_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

model = ContrastiveLearning(channels=3).to(device, dtype=float)
model.load_state_dict(load('ImageContrastModel.pt', save_keys='model', device=device))
model.eval()

with torch.no_grad():
    with tqdm(embed_loader, total=len(embed_loader), desc='Embeding Cell Images') as embed_loader:
        embed = torch.Tensor()
        for idx, batch in enumerate(embed_loader):
            out = model(batch)
            embed = torch.cat((embed, out), dim=0)
        dataset.save_embed_data(embed)
