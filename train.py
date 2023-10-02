import torch
from models import ContrastiveLearning
from embed_data import EmbedDataset
from torch.utils.data import DataLoader
from loss import add_contrastive_loss

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = EmbedDataset()
model = ContrastiveLearning()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, weight_decay=5e-4)
loss = add_contrastive_loss

train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1, drop_last=True)
val_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1, drop_last=True)

EPOCH = 500
for epoch in list(range(EPOCH)):
    running_loss = 0
    model.train()

    for idx, batch in enumerate(train_loader):
        batch = torch.cat(batch[0], batch[1])
        optimizer.zero_grad()
        out = model(batch)
        l, _, _ = loss(out)
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
            batch = torch.cat(batch[0], batch[1])
            out = model(batch)
            l, _, _ = loss(out)
            running_loss += l.item()

        epoch_loss = running_loss / len(val_loader)
        print(f"{epoch}: Val Loss: {epoch_loss}")


with torch.no_grad():
    running_loss = 0
    model.eval()
    dataset.setMode("test")

    for idx, batch in enumerate(val_loader):
        batch = torch.cat(batch[0], batch[1])
        out = model(batch)
        l, _, _ = loss(out)
        running_loss += l.item()

    epoch_loss = running_loss / len(val_loader)
    print(f"Test Loss: {epoch_loss}")