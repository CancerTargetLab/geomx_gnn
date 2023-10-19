from src.gnn.data import GeoMXDataset
from src.gnn.models import ROIExpression
from src.utils.setSeed import set_seed
import torch_geometric
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm

EPOCH = 100
SEED = 42
batch_size = 2
lr = 0.005
num_workers = 0
early_stopping = 10

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(SEED)

dataset = GeoMXDataset(node_dropout=0.3, edge_dropout=0.4)
dataset.setMode(dataset.train)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
dataset.setMode(dataset.val)
val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
dataset.setMode(dataset.test)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

model = ROIExpression(num_out_features=dataset.get(0).y.shape[0], layers=3, heads=1).to(device, dtype=float)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
dataset.setMode(dataset.train)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                max_lr=lr, 
                                                epochs=EPOCH, 
                                                steps_per_epoch=len(train_loader), 
                                                pct_start=0.1,
                                                div_factor=25,
                                                final_div_factor=1e6)

loss = torch.nn.MSELoss()
similarity = torch.nn.CosineSimilarity()

train_acc_list = []
train_loss_list = []
val_acc_list = []
val_loss_list = []
best_acc = -1.0
best_run = 0

for epoch in list(range(EPOCH)):
    best_run += 1
    running_loss = 0
    running_acc = 0
    num_graphs = 0
    model.train()
    dataset.setMode(dataset.train)

    if best_run < early_stopping:
        with tqdm(train_loader, total=len(train_loader), desc=f"Training epoch {epoch}") as train_loader:
            for idx, batch in enumerate(train_loader):
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                l = loss(torch.log10(out), torch.log10(batch.y.view(out.shape[0], out.shape[1])))
                l.backward()
                optimizer.step()
                scheduler.step()
                running_loss += l.item() * out.shape[0]
                running_acc += torch.mean(similarity(out, batch.y.view(out.shape[0], out.shape[1]))).item() * out.shape[0]
                num_graphs += out.shape[0]

            train_acc = running_acc / num_graphs
            train_acc_list.append(train_acc)
            epoch_loss = running_loss / num_graphs
            train_loss_list.append(epoch_loss)
            print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        with torch.no_grad():
            running_loss = 0
            running_acc = 0
            num_graphs = 0
            model.eval()
            dataset.setMode("val")

            with tqdm(val_loader, total=len(val_loader), desc=f"Validation epoch {epoch}") as val_loader:
                for idx, batch in enumerate(val_loader):
                    batch = batch.to(device)
                    out = model(batch)
                    l = loss(torch.log10(out), torch.log10(batch.y.view(out.shape[0], out.shape[1])))
                    running_loss += l.item()
                    running_acc += torch.mean(similarity(out, batch.y.view(out.shape[0], out.shape[1]))).item() * out.shape[0]
                    num_graphs += out.shape[0]

                val_acc = running_acc / num_graphs
                val_acc_list.append(val_acc)
                epoch_loss = running_loss / num_graphs
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
                    }, 'ROIModel.pt')
                print(f"Val Loss: {epoch_loss:.4f}, Val Accuracy: {val_acc:.4f}")


with torch.no_grad():
    running_loss = 0
    running_acc = 0
    num_graphs = 0
    model.load_state_dict(torch.load('ROIModel.pt')['model'])
    model.eval()
    dataset.setMode(dataset.test)

    with tqdm(test_loader, total=len(test_loader), desc="Test") as test_loader:
        for idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            out = model(batch)
            l = loss(torch.log10(out), torch.log10(batch.y.view(out.shape[0], out.shape[1])))
            running_loss += l.item()
            running_acc += torch.mean(similarity(out, batch.y.view(out.shape[0], out.shape[1]))).item() * out.shape[0]
            num_graphs += out.shape[0]

        test_acc = running_acc / num_graphs
        epoch_loss = running_loss / num_graphs
        print(f"Test Loss: {epoch_loss:.4f}, Test Accuracy: {test_acc:.4f}")
