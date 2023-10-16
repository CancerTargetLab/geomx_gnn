from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.Cell_Contrastive_Learning.CellContrastiveModel import ContrastiveLearning
from src.Cell_Contrastive_Learning.data import EmbedDataset
from src.Cell_Contrastive_Learning.loss import add_contrastive_loss
from src.Cell_Contrastive_Learning.larc import LARC
from src.utils.setSeed import set_seed

batch_size = 256
max_lr = 0.1
lr = 0.004
warmup_epochs = 10
EPOCH = 100
num_workers = 8
early_stopping = 10
seed = 42

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(42)

dataset = EmbedDataset(root_dir='data/raw/TMA1_preprocessed', device=device)
model = ContrastiveLearning(channels=3, resnet='18').to(device, dtype=float)

#TODO: https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
# -> more mem eff
dataset.setMode(dataset.train)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
iters = len(train_loader)
dataset.setMode(dataset.val)
val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
dataset.setMode(dataset.test)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

#TODO: lr scheduling
#sc_lr = lr * batch_size / 256
dataset.setMode(dataset.train)
#warmup_steps = int(round(warmup_epochs*len(dataset)/batch_size))
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                max_lr=max_lr, 
                                                epochs=EPOCH, 
                                                steps_per_epoch=len(train_loader), 
                                                pct_start=0.1,
                                                div_factor=25,
                                                final_div_factor=1e5)
optimizer = LARC(optimizer, clip=False)
loss = add_contrastive_loss


train_acc_list = []
train_loss_list = []
val_acc_list = []
val_loss_list = []
best_acc = 0.0
best_run = 0

for epoch in list(range(EPOCH)):
    best_run += 1
    running_loss = 0
    running_acc = 0
    model.train()
    dataset.setMode(dataset.train)

    if best_run < early_stopping:
        with tqdm(train_loader, total=len(train_loader), desc=f"Training epoch {epoch}") as train_loader:
            for idx, batch in enumerate(train_loader):
                batch = torch.cat((batch[0], batch[1])).to(device)
                optimizer.zero_grad()
                out = model(batch)
                l, logits, labels = loss(out)
                l.backward()
                optimizer.step()
                scheduler.step()
                running_loss += l.item()

                # Compute element-wise equality between the predicted labels and true labels
                contrast_acc = torch.eq(torch.argmax(labels, dim=1), torch.argmax(logits, dim=1))
                # Convert the boolean tensor to float32 and compute the mean
                running_acc += torch.mean(contrast_acc.float()).item()

            train_acc = running_acc / len(train_loader)
            train_acc_list.append(train_acc)
            epoch_loss = running_loss / len(train_loader)
            train_loss_list.append(epoch_loss)
            print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        with torch.no_grad():
            running_loss = 0
            running_acc = 0
            model.eval()
            dataset.setMode("val")

            with tqdm(val_loader, total=len(val_loader), desc=f"Validation epoch {epoch}") as val_loader:
                for idx, batch in enumerate(val_loader):
                    batch = torch.cat((batch[0], batch[1])).to(device)
                    out = model(batch)
                    l, logits, labels = loss(out)
                    running_loss += l.item()

                    # Compute element-wise equality between the predicted labels and true labels
                    contrast_acc = torch.eq(torch.argmax(labels, dim=1), torch.argmax(logits, dim=1))
                    # Convert the boolean tensor to float32 and compute the mean
                    running_acc += torch.mean(contrast_acc.float()).item()

                val_acc = running_acc / len(val_loader)
                val_acc_list.append(val_acc)
                epoch_loss = running_loss / len(val_loader)
                val_loss_list.append(epoch_loss)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_run = 0
                    torch.save({
                        "model": model.state_dict(),
                        "opt": optimizer.state_dict(),
                        "train_acc": train_acc_list,
                        "train_list": train_loss_list,
                        "val_acc": val_acc_list,
                        "val_list": val_loss_list,
                        "epoch": epoch
                    }, 'ImageContrastModel.pt')
                print(f"Val Loss: {epoch_loss:.4f}, Val Accuracy: {val_acc:.4f}")


with torch.no_grad():
    running_loss = 0
    running_acc = 0
    model.load_state_dict(torch.load('ImageContrastModel.pt')['model'])
    model.eval()
    dataset.setMode(dataset.test)

    with tqdm(test_loader, total=len(test_loader), desc="Test") as test_loader:
        for idx, batch in enumerate(test_loader):
            batch = torch.cat((batch[0], batch[1])).to(device)
            out = model(batch)
            l, logits, labels = loss(out)
            running_loss += l.item()

            # Compute element-wise equality between the predicted labels and true labels
            contrast_acc = torch.eq(torch.argmax(labels, dim=1), torch.argmax(logits, dim=1))
            # Convert the boolean tensor to float32 and compute the mean
            running_acc += torch.mean(contrast_acc.float()).item()

        test_acc = running_acc / len(test_loader)
        epoch_loss = running_loss / len(test_loader)
        print(f"Test Loss: {epoch_loss:.4f}, Test Accuracy: {test_acc:.4f}")