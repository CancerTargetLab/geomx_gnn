from src.gnn.data import GeoMXDataset
from src.gnn.models import GraphLearning
from src.utils.setSeed import set_seed
import torch_geometric
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm

EPOCH = 100

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = GeoMXDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for epoch in list(range(EPOCH)):
    running_loss = 0.
    for batch in loader:
        print(batch)