from src.data.GeoMXData import GeoMXDataset
from src.models.GraphModel import ROIExpression, ROIExpression_ph, ROIExpression_lin, ROIExpression_lin_ph
from src.loss.CellEntropyLoss import phenotype_entropy_loss
from src.utils.setSeed import set_seed
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm

def train(raw_subset_dir, label_data, output_name, args):

    EPOCH = args['epochs_graph']
    SEED = args['seed']
    model_type = args['graph_model_type']
    batch_size = args['batch_size_graph']
    lr = args['lr_graph']
    num_workers = args['num_workers_graph']
    early_stopping = args['early_stopping_graph']
    is_log = args['data_is_log_tme']

    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args['deterministic']:
        set_seed(SEED)

    dataset = GeoMXDataset(root_dir=args['graph_dir'],
                           raw_subset_dir=raw_subset_dir,
                           train_ratio=args['train_ratio_graph'],
                           val_ratio=args['val_ratio_graph'],
                           node_dropout=args['node_dropout'],
                           edge_dropout=args['edge_dropout'],
                           label_data=label_data)
    dataset.setMode(dataset.train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataset.setMode(dataset.val)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataset.setMode(dataset.test)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if model_type == 'GAT':
        model = ROIExpression(layers=args['layers_graph'],
                            num_node_features=args['num_node_features'],
                            num_edge_features=args['num_edge_features'],
                            num_embed_features=args['num_embed_features'],
                            embed_dropout=args['embed_dropout_graph'],
                            conv_dropout=args['conv_dropout_graph'],
                            num_out_features=dataset.get(0).y.shape[0],
                            heads=args['heads_graph']).to(device, dtype=torch.float32)
    elif model_type == 'GAT_ph':
        model = ROIExpression_ph(layers=args['layers_graph'],
                            num_node_features=args['num_node_features'],
                            num_edge_features=args['num_edge_features'],
                            num_embed_features=args['num_embed_features'],
                            embed_dropout=args['embed_dropout_graph'],
                            conv_dropout=args['conv_dropout_graph'],
                            num_out_features=dataset.get(0).y.shape[0],
                            heads=args['heads_graph'],
                            num_phenotypes=args['num_phenotypes_graph'],
                            num_phenotype_layers=args['num_phenotypes_layers_graph']).to(device, dtype=torch.float32)
    elif model_type == 'LIN':
        model = ROIExpression_lin(layers=args['layers_graph'],
                            num_node_features=args['num_node_features'],
                            num_embed_features=args['num_embed_features'],
                            embed_dropout=args['embed_dropout_graph'],
                            conv_dropout=args['conv_dropout_graph'],
                            num_out_features=dataset.get(0).y.shape[0]).to(device, dtype=torch.float32)
    elif model_type == 'LIN_ph':
        model = ROIExpression_lin_ph(layers=args['layers_graph'],
                            num_node_features=args['num_node_features'],
                            num_embed_features=args['num_embed_features'],
                            embed_dropout=args['embed_dropout_graph'],
                            conv_dropout=args['conv_dropout_graph'],
                            num_out_features=dataset.get(0).y.shape[0],
                            num_phenotypes=args['num_phenotypes_graph'],
                            num_phenotype_layers=args['num_phenotypes_layers_graph']).to(device, dtype=torch.float32)
    else:
        raise Exception(f'{model_type} not a valid model type, must be one of GAT, GAT_ph, LIN, LIN_ph')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    dataset.setMode(dataset.train)

    loss = torch.nn.MSELoss()
    similarity = torch.nn.CosineSimilarity()


    train_acc_list = []
    train_loss_list = []
    train_ph_entropy_list = []
    train_total_loss_list = []
    val_acc_list = []
    val_loss_list = []
    val_ph_entropy_list = []
    val_total_loss_list = []
    best_acc = -1.0
    best_run = 0

    for epoch in list(range(EPOCH)):
        best_run += 1
        running_loss = 0
        running_total_loss = 0
        running_acc = 0
        running_ph_entropy = 0
        num_graphs = 0
        model.train()
        dataset.setMode(dataset.train)

        if best_run < early_stopping:
            with tqdm(train_loader, total=len(train_loader), desc=f"Training epoch {epoch}") as train_loader:
                for idx, batch in enumerate(train_loader):
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    if model_type.endswith('_ph'):
                        out, ph_logits = model(batch)
                        ph = phenotype_entropy_loss(torch.softmax(ph_logits, 1))
                    else:
                        out = model(batch)
                    if is_log:
                        l = loss(torch.log(out), batch.y.view(out.shape[0], out.shape[1]))
                    else:
                        l = loss(torch.log(out), torch.log(batch.y.view(out.shape[0], out.shape[1])))
                    sim = torch.mean(similarity(out, batch.y.view(out.shape[0], out.shape[1])))
                    running_loss += l.item() * out.shape[0]
                    running_acc += sim.item() * out.shape[0]
                    num_graphs += out.shape[0]
                    if model_type.endswith('_ph'):
                        running_ph_entropy += ph.item() * out.shape[0]
                        l += 1 - sim + ph
                    else:
                        l += 1 - sim
                    running_total_loss += l.item() * out.shape[0]
                    l.backward()
                    optimizer.step()

                train_acc = running_acc / num_graphs
                train_acc_list.append(train_acc)
                geo_loss = running_loss / num_graphs
                train_loss_list.append(geo_loss)
                epoch_loss = running_total_loss / num_graphs
                train_total_loss_list.append(epoch_loss)
                if model_type.endswith('_ph'):
                    ph_entropy = running_ph_entropy / num_graphs
                    train_ph_entropy_list.append(ph_entropy)
                    print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train Phenotype Entropy: {ph_entropy:.4f}")
                else: 
                    print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_acc:.4f}")

            with torch.no_grad():
                running_loss = 0
                running_total_loss = 0
                running_acc = 0
                running_ph_entropy = 0
                num_graphs = 0
                model.eval()
                dataset.setMode("val")

                with tqdm(val_loader, total=len(val_loader), desc=f"Validation epoch {epoch}") as val_loader:
                    for idx, batch in enumerate(val_loader):
                        batch = batch.to(device)
                        if model_type.endswith('_ph'):
                            out, ph_logits = model(batch)
                            ph = phenotype_entropy_loss(torch.softmax(ph_logits, 1))
                        else: 
                            out = model(batch)
                        if is_log:
                            l = loss(torch.log(out), batch.y.view(out.shape[0], out.shape[1]))
                        else:
                            l = loss(torch.log(out), torch.log(batch.y.view(out.shape[0], out.shape[1])))
                        sim = torch.mean(similarity(out, batch.y.view(out.shape[0], out.shape[1])))
                        running_loss += l.item() * out.shape[0]
                        running_acc += sim.item() * out.shape[0]
                        num_graphs += out.shape[0]
                        if model_type.endswith('_ph'):
                            running_ph_entropy += ph.item() * out.shape[0]
                            l += 1 - sim + ph
                        else:
                            l += 1 - sim
                        running_total_loss += l.item() * out.shape[0]

                    val_acc = running_acc / num_graphs
                    val_acc_list.append(val_acc)
                    geo_loss = running_loss / num_graphs
                    val_loss_list.append(geo_loss)
                    epoch_loss = running_total_loss / num_graphs
                    val_total_loss_list.append(epoch_loss)
                    if model_type.endswith('_ph'):
                        ph_entropy = running_ph_entropy / num_graphs
                        val_ph_entropy_list.append(ph_entropy)
                        if val_acc > best_acc:
                            best_acc = val_acc
                            best_run = 0
                            torch.save({
                                "model": model.state_dict(),
                                "opt": optimizer.state_dict(),
                                "train_acc": train_acc_list,
                                "train_list": train_loss_list,
                                "train_total_list": train_total_loss_list,
                                "val_acc": val_acc_list,
                                "val_list": val_loss_list,
                                "val_total_list": val_total_loss_list,
                                "epoch": epoch
                            }, output_name)
                        print(f"Val Loss: {epoch_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val Phenotype Entropy: {ph_entropy:.4f}")
                    else:
                        if val_acc > best_acc:
                            best_acc = val_acc
                            best_run = 0
                            torch.save({
                                "model": model.state_dict(),
                                "opt": optimizer.state_dict(),
                                "train_acc": train_acc_list,
                                "train_list": train_loss_list,
                                "train_total_list": train_total_loss_list,
                                "val_acc": val_acc_list,
                                "val_list": val_loss_list,
                                "val_total_list": val_total_loss_list,
                                "epoch": epoch
                            }, output_name)
                        print(f"Val Loss: {epoch_loss:.4f}, Val Accuracy: {val_acc:.4f}")


    with torch.no_grad():
        running_loss = 0
        running_total_loss = 0
        running_acc = 0
        running_ph_entropy = 0
        num_graphs = 0
        model.load_state_dict(torch.load(output_name)['model'])
        model.eval()
        dataset.setMode(dataset.test)

        with tqdm(test_loader, total=len(test_loader), desc="Test") as test_loader:
            for idx, batch in enumerate(test_loader):
                batch = batch.to(device)
                if model_type.endswith('_ph'):
                    out, ph_logits = model(batch)
                    ph = phenotype_entropy_loss(torch.softmax(ph_logits, 1))
                else:
                    out = model(batch)
                if is_log:
                    l = loss(torch.log(out), batch.y.view(out.shape[0], out.shape[1]))
                else:
                    l = loss(torch.log(out), torch.log(batch.y.view(out.shape[0], out.shape[1])))
                sim = torch.mean(similarity(out, batch.y.view(out.shape[0], out.shape[1])))
                running_loss += l.item() * out.shape[0]
                running_acc += sim.item() * out.shape[0]
                num_graphs += out.shape[0]
                if model_type.endswith('_ph'):
                    running_ph_entropy += ph.item() * out.shape[0]
                    l += 1 - sim + ph
                else:
                    l += 1 - sim
                running_total_loss += l.item() * out.shape[0]

            test_acc = running_acc / num_graphs
            geo_loss = running_loss / num_graphs
            epoch_loss = running_total_loss / num_graphs
            if model_type.endswith('_ph'):
                ph_entropy = running_ph_entropy / num_graphs
                print(f"Test Loss: {epoch_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test Phenotype Entropy: {ph_entropy:.4f}")
            else:
                print(f"Test Loss: {epoch_loss:.4f}, Test Accuracy: {test_acc:.4f}")
