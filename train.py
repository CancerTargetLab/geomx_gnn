import torch
from models import ContrastiveLearning
from embed_data import EmbedDataset
from torch.utils.data import DataLoader
from loss import add_contrastive_loss

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = EmbedDataset()
model = ContrastiveLearning(channels=3).to(device, dtype=float)


optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, weight_decay=5e-4)
loss = add_contrastive_loss

train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1, drop_last=True)
val_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1, drop_last=True)

train_acc_list = []
train_loss_list = []
best_acc = 0.0
best_run = 0

EPOCH = 200
for epoch in list(range(EPOCH)):
    best_run += 1
    running_loss = 0
    running_acc = 0
    model.train()

    if best_run < 5:
        for idx, batch in enumerate(train_loader):
            batch = torch.cat((batch[0], batch[1])).to(device)
            optimizer.zero_grad()
            out = model(batch)
            l, logits, labels = loss(out)
            l.backward()
            optimizer.step()
            running_loss += l.item()

            # Compute element-wise equality between the predicted labels and true labels
            contrast_acc = torch.eq(torch.argmax(labels, dim=1), torch.argmax(logits, dim=1))
            # Convert the boolean tensor to float32 and compute the mean
            running_acc += torch.mean(contrast_acc.float()).item()
            print(idx)

        train_acc = 100*running_acc / len(train_loader)
        train_acc_list.append(train_acc)
        epoch_loss = running_loss / len(train_loader)
        train_loss_list.append(epoch_loss)
        if train_acc > best_acc:
            best_acc = train_acc
            best_run = 0
            torch.save({
                "model": model,
                "train_acc": train_acc_list,
                "train_list": train_loss_list,
                "epoch": epoch
            }, 'ImageContrastModel.pt')
        print(f"{epoch}: Train Loss: {epoch_loss}, Train Accuracy: {train_acc}")

    # with torch.no_grad():
    #     running_loss = 0
    #     model.eval()
    #     dataset.setMode("val")

    #     for idx, batch in enumerate(val_loader):
    #         batch = torch.cat(batch[0], batch[1])
    #         out = model(batch)
    #         l, _, _ = loss(out)
    #         running_loss += l.item()

    #     epoch_loss = running_loss / len(val_loader)
    #     print(f"{epoch}: Val Loss: {epoch_loss}")


with torch.no_grad():
    running_loss = 0
    running_acc = 0
    model.eval()

    for idx, batch in enumerate(val_loader):
        batch = torch.cat(batch[0], batch[1]).to(device)
        out = model(batch)
        l, _, _ = loss(out)
        running_loss += l.item()

        # Compute element-wise equality between the predicted labels and true labels
        contrast_acc = torch.eq(torch.argmax(labels, dim=1), torch.argmax(logits, dim=1))
        # Convert the boolean tensor to float32 and compute the mean
        running_acc += torch.mean(contrast_acc.float()).item()

    test_acc = running_acc / len(val_loader)
    epoch_loss = running_loss / len(val_loader)
    print(f"Test Loss: {epoch_loss}, Train Accuracy: {test_acc}")