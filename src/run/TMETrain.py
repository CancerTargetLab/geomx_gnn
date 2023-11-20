from src.data.TMEData import TMEDataset
from src.models.GraphModel import kTME
from src.loss.ContrastiveLoss import add_contrastive_loss
from src.utils.setSeed import set_seed
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm

def train(args):

    EPOCH = args['epochs_tme']
    SEED = args['seed']
    batch_size = args['batch_size_tme']
    lr = args['lr_tme']
    num_workers = args['num_workers_tme']
    early_stopping = args['early_stopping_tme']

    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(SEED)

    dataset = TMEDataset(root_dir=args['tme_dir'],
                           raw_subset_dir=args['tme_raw_subset_dir'],
                           train_ratio=args['tme_ratio_graph'],
                           val_ratio=args['tme_ratio_graph'],
                           label_data=args['tme_label_data'],
                           walk_length=args['walk_length_tme'],
                           repeat=args['repeat_tme'])
    dataset.setMode(dataset.train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, follow_batch=['x_s', 'x_t'])
    dataset.setMode(dataset.val)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, follow_batch=['x_s', 'x_t'])
    dataset.setMode(dataset.test)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, follow_batch=['x_s', 'x_t'])

    model = kTME(k=args['layers_tme'],
                          num_node_features=args['num_node_features_tme'],
                          num_edge_features=args['num_edge_features_tme'],
                          num_embed_features=args['num_embed_features_tme'],
                          embed_dropout=args['embed_dropout_tme'],
                          conv_dropout=args['conv_dropout_tme'],
                          num_out_features=args['num_out_features_tme'],
                          heads=args['heads_tme']).to(device, dtype=float)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    dataset.setMode(dataset.train)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
    #                                                 max_lr=lr, 
    #                                                 epochs=EPOCH, 
    #                                                 steps_per_epoch=len(train_loader), 
    #                                                 pct_start=0.1,
    #                                                 div_factor=25,
    #                                                 final_div_factor=1e6)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    loss = add_contrastive_loss

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
                    batch = dataset.subgraph_batching(batch)
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    out = model(batch)
                    l, logits, labels = loss(out)
                    l.backward()
                    optimizer.step()
                    # scheduler.step()
                    running_loss += l.item() * out.shape[0]
                    # Compute element-wise equality between the predicted labels and true labels
                    contrast_acc = torch.eq(torch.argmax(labels, dim=1), torch.argmax(logits, dim=1))
                    # Convert the boolean tensor to float32 and compute the mean
                    running_acc += torch.mean(contrast_acc.float()).item() * out.shape[0]
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
                        batch = dataset.subgraph_batching(batch)
                        batch = batch.to(device)
                        out = model(batch)
                        l, logits, labels = loss(out)
                        running_loss += l.item() * out.shape[0]
                        # Compute element-wise equality between the predicted labels and true labels
                        contrast_acc = torch.eq(torch.argmax(labels, dim=1), torch.argmax(logits, dim=1))
                        # Convert the boolean tensor to float32 and compute the mean
                        running_acc += torch.mean(contrast_acc.float()).item() * out.shape[0]
                        num_graphs += out.shape[0]

                    val_acc = running_acc / num_graphs
                    val_acc_list.append(val_acc)
                    epoch_loss = running_loss / num_graphs
                    # scheduler.step(epoch_loss)
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
                        }, args['output_name_tme'])
                    print(f"Val Loss: {epoch_loss:.4f}, Val Accuracy: {val_acc:.4f}")


    with torch.no_grad():
        running_loss = 0
        running_acc = 0
        num_graphs = 0
        model.load_state_dict(torch.load(args['output_name_tme'])['model'])
        model.eval()
        dataset.setMode(dataset.test)

        with tqdm(test_loader, total=len(test_loader), desc="Test") as test_loader:
            for idx, batch in enumerate(test_loader):
                batch = dataset.subgraph_batching(batch)
                batch = batch.to(device)
                out = model(batch)
                l, logits, labels = loss(out)
                running_loss += l.item() * out.shape[0]
                # Compute element-wise equality between the predicted labels and true labels
                contrast_acc = torch.eq(torch.argmax(labels, dim=1), torch.argmax(logits, dim=1))
                # Convert the boolean tensor to float32 and compute the mean
                running_acc += torch.mean(contrast_acc.float()).item() * out.shape[0]
                num_graphs += out.shape[0]

            test_acc = running_acc / num_graphs
            epoch_loss = running_loss / num_graphs
            print(f"Test Loss: {epoch_loss:.4f}, Test Accuracy: {test_acc:.4f}")