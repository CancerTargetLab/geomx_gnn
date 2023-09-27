import torch
from models import AutoEncodeEmbedding
from embed_data import EmbedDataset
from torch.utils.data import DataLoader

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = EmbedDataset()
model = AutoEncodeEmbedding()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss = torch.nn.L1Loss(reduction="mean")

dataset.setMode(mode="train")
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1, drop_last=False)
dataset.setMode(mode="val")
val_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=1, drop_last=False)
dataset.setMode(mode="test")
test_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=1, drop_last=False)

EPOCH = 500
for epoch in list(range(EPOCH)):
    running_loss = 0
    model.train()
    dataset.setMode("train")

    for idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        out = model(batch)
        l = loss(out, batch)
        l.backward()
        optimizer.step()
        running_loss += l.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"{epoch}: Train Loss: {epoch_loss}")

    with torch.no_grad():
        running_loss = 0
        model.eval()
        dataset.setMode("val")

        for idx, batch in enumerate(val_loader):
            out = model(batch)
            l = loss(out, batch)
            running_loss += l.item()

        epoch_loss = running_loss / len(val_loader)
        print(f"{epoch}: Val Loss: {epoch_loss}")


with torch.no_grad():
    running_loss = 0
    model.eval()
    dataset.setMode("test")

    for idx, batch in enumerate(test_loader):
        out = model(batch)
        l = loss(out, batch)
        running_loss += l.item()

    epoch_loss = running_loss / len(test_loader)
    print(f"Test Loss: {epoch_loss}")