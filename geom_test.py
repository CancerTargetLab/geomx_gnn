from src.gnn.models import GraphLearning
import torch
import torch_geometric
from torch_geometric.datasets import FakeDataset
from torch_geometric.loader import DataLoader

dataset = FakeDataset(avg_num_nodes=10, edge_dim=1, avg_degree=5, num_graphs=2, task="node")

model = GraphLearning(num_node_features=64, num_embed_features=10, layers=4, heads=7, embed_dropout=0, conv_dropout=0, skip_dropout=0)

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loader = DataLoader(dataset, batch_size=2, shuffle=True)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for epoch in list(range(400)):
    running_loss = 0.
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.to(device)).softmax(dim=1)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    running_loss = running_loss/len(loader)
    print("Epoch:", epoch, "Loss: ", running_loss)

model.eval()
out = model(dataset).softmax(dim=1).argmax(dim=1)

p_correct = torch.sum(torch.eq(out, dataset.y)) / dataset.y.shape[0]


print("Correct id: ", p_correct)

