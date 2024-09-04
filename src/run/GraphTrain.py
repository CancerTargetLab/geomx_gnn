from src.data.GeoMXData import GeoMXDataset
from src.data.ImageGraphData import ImageGraphDataset
from src.models.GraphModel import ROIExpression, ROIExpression_lin, ROIExpression_Image_gat, ROIExpression_Image_lin
from src.loss.CellEntropyLoss import phenotype_entropy_loss
from src.loss.zinb import ZINBLoss, NBLoss
from src.utils.setSeed import set_seed
from src.utils.stats import per_gene_pcc
from torch_geometric.loader import DataLoader
import torch
import os
from tqdm import tqdm

def train(raw_subset_dir, label_data, output_name, args):
    """
    Train model to predict sc expression of cells.

    Parameters:
    raw_subset_dir (str): name of dir in which torch.tensors of visual cell embeddings are
    label_data (str): Name of .csv in raw/ containing label information of ROIs
    output_name (str): Path and name of model torch save dict
    args (dict): Arguments
    """

    EPOCH = args['epochs_graph']
    SEED = args['seed']
    model_type = args['graph_model_type']
    batch_size = args['batch_size_graph']
    lr = args['lr_graph']
    num_workers = args['num_workers_graph']
    early_stopping = args['early_stopping_graph']
    is_log = args['data_use_log_graph']
    alpha = args['graph_mse_mult']
    beta = args['graph_cos_sim_mult']
    theta = args['graph_entropy_mult']

    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args['deterministic']:
        set_seed(SEED)

    #Wether to train together with image model
    if 'IMAGE' in model_type:
        dataset = ImageGraphDataset(root_dir=args['graph_dir'],
                                    raw_subset_dir=raw_subset_dir,
                                    train_ratio=args['train_ratio_graph'],
                                    val_ratio=args['val_ratio_graph'],
                                    num_folds=args['num_folds'],
                                    node_dropout=args['node_dropout'],
                                    edge_dropout=args['edge_dropout'],
                                    pixel_pos_jitter=args['cell_pos_jitter'],
                                    n_knn=args['cell_n_knn'],
                                    subgraphs_per_graph=args['subgraphs_per_graph'],
                                    num_hops=args['num_hops_subgraph'],
                                    label_data=label_data,
                                    crop_factor=args['crop_factor'],
                                    output_name=output_name)
    else:
        dataset = GeoMXDataset(root_dir=args['graph_dir'],
                            raw_subset_dir=raw_subset_dir,
                            train_ratio=args['train_ratio_graph'],
                            val_ratio=args['val_ratio_graph'],
                            num_folds=args['num_folds'],
                            node_dropout=args['node_dropout'],
                            edge_dropout=args['edge_dropout'],
                            pixel_pos_jitter=args['cell_pos_jitter'],
                            n_knn=args['cell_n_knn'],
                            subgraphs_per_graph=args['subgraphs_per_graph'],
                            num_hops=args['num_hops_subgraph'],
                            label_data=label_data,
                            output_name=output_name)

    for k in range(args['num_folds']):
        output_name_model = os.path.join(output_name.split('.')[0], f'{k}'+'.'+output_name.split('.')[-1])
        dataset.set_fold_k()
        dataset.setMode(dataset.train)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        dataset.setMode(dataset.val)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        dataset.setMode(dataset.test)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        if 'IMAGEGAT' in model_type:
            model = ROIExpression_Image_gat(channels=dataset.get(0).x.shape[1],
                                            embed=args['embedding_size_image'],
                                            contrast=args['contrast_size_image'], 
                                            resnet=args['resnet_model'],
                                            layers=args['layers_graph'],
                                            num_edge_features=args['num_edge_features'],
                                            num_embed_features=args['num_embed_features'],
                                            num_out_features=dataset.get(0).y.shape[0],
                                            heads=args['heads_graph'],
                                            embed_dropout=args['embed_dropout_graph'],
                                            conv_dropout=args['conv_dropout_graph'],
                                            mtype=model_type,
                                            path_image_model=args['init_image_model'],
                                            path_graph_model=args['init_graph_model']).to(device, dtype=torch.float32)
        elif 'GAT' in model_type:
            model = ROIExpression(layers=args['layers_graph'],
                                num_node_features=args['num_node_features'],
                                num_edge_features=args['num_edge_features'],
                                num_embed_features=args['num_embed_features'],
                                embed_dropout=args['embed_dropout_graph'],
                                conv_dropout=args['conv_dropout_graph'],
                                num_out_features=dataset.get(0).y.shape[0],
                                heads=args['heads_graph'],
                                mtype=model_type).to(device, dtype=torch.float32)
        elif 'IMAGELIN' in model_type:
            model = ROIExpression_Image_lin(channels=dataset.get(0).x.shape[1],
                                            embed=args['embedding_size_image'],
                                            contrast=args['contrast_size_image'], 
                                            resnet=args['resnet_model'],
                                            layers=args['layers_graph'],
                                            num_embed_features=args['num_embed_features'],
                                            num_out_features=dataset.get(0).y.shape[0],
                                            embed_dropout=args['embed_dropout_graph'],
                                            conv_dropout=args['conv_dropout_graph'],
                                            mtype=model_type,
                                            path_image_model=args['init_image_model'],
                                            path_graph_model=args['init_graph_model']).to(device, dtype=torch.float32)
        elif 'LIN' in model_type:
            model = ROIExpression_lin(layers=args['layers_graph'],
                                num_node_features=args['num_node_features'],
                                num_embed_features=args['num_embed_features'],
                                embed_dropout=args['embed_dropout_graph'],
                                conv_dropout=args['conv_dropout_graph'],
                                num_out_features=dataset.get(0).y.shape[0],
                                mtype=model_type).to(device, dtype=torch.float32)
        else:
            raise Exception(f'{model_type} not a valid model type, must be one of GAT, GAT_ph, LIN, LIN_ph')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        dataset.setMode(dataset.train)

        loss = torch.nn.L1Loss()
        similarity = torch.nn.CosineSimilarity()
        zinb = ZINBLoss(ridge_lambda=0.0, device=device)
        nb = NBLoss(device=device)

        #scale factor, TODO:rm
        sf = dataset.sf.to(device)

        train_acc_list = []
        train_loss_list = []
        train_ph_entropy_list = []
        train_zinb_list = []
        train_total_loss_list = []
        val_acc_list = []
        val_loss_list = []
        val_ph_entropy_list = []
        val_zinb_list = []
        val_total_loss_list = []
        val_pcc_statistic_list = []
        val_pcc_pval_list = []
        best_acc = float('inf')
        best_run = 0

        for epoch in list(range(EPOCH)):
            best_run += 1
            running_loss = 0
            running_total_loss = 0
            running_acc = 0
            running_ph_entropy = 0
            running_zinb = 0
            num_graphs = 0
            model.train()
            dataset.setMode(dataset.train)

            if best_run < early_stopping:
                with tqdm(train_loader, total=len(train_loader), desc=f"Training epoch {epoch} of Fold {k}") as train_loader:
                    for idx, batch in enumerate(train_loader):
                        batch = batch.to(device)
                        optimizer.zero_grad()
                        if model_type.endswith('_ph'):
                            out = model(batch)
                            ph = phenotype_entropy_loss(torch.softmax(out.permute(1, 0), 1)) * theta
                        elif model_type.endswith('_zinb'):
                            out, pred, mean, disp, drop = model(batch)
                            loss_zinb = zinb(pred, mean, disp, drop)  * theta
                        elif model_type.endswith('_nb'):
                            out, pred, mean, disp = model(batch)
                            loss_zinb = nb(pred, mean, disp)  * theta
                        else:
                            out = model(batch)
                        if is_log:
                            l = loss(torch.log(out+1), torch.log(batch.y.view(out.shape[0], out.shape[1])+1))
                        else:
                            l = loss(out, batch.y.view(out.shape[0], out.shape[1]))
                        l = alpha * l
                        sim = torch.mean(similarity(out, batch.y.view(out.shape[0], out.shape[1]))) * beta
                        running_loss += l.item() * out.shape[0]
                        running_acc += sim.item() * out.shape[0]
                        num_graphs += out.shape[0]
                        if model_type.endswith('_ph'):
                            running_ph_entropy += ph.item() * out.shape[0]
                            l += 1 - sim + ph
                        elif model_type.endswith('_zinb'):
                            running_zinb += loss_zinb.item() * out.shape[0]
                            l += 1 - sim + loss_zinb
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
                        print(f"Train Loss: {epoch_loss:.4f}, MSE Loss: {geo_loss:.4f}, Train Cosine Sim: {train_acc:.4f}, Train Phenotype Entropy: {ph_entropy:.4f}")
                    elif model_type.endswith('_zinb') or model_type.endswith('_nb'):
                        zinb_loss = running_zinb / num_graphs
                        train_zinb_list.append(zinb_loss)
                        print(f"Train Loss: {epoch_loss:.4f}, MSE Loss: {geo_loss:.4f}, Train Cosine Sim: {train_acc:.4f}, Train ZINB Loss: {zinb_loss:.4f}")
                    else: 
                        print(f"Train Loss: {epoch_loss:.4f}, MSE Loss: {geo_loss:.4f}, Train Cosine Sim: {train_acc:.4f}")

                with torch.no_grad():
                    running_loss = 0
                    running_total_loss = 0
                    running_acc = 0
                    running_ph_entropy = 0
                    running_zinb = 0
                    num_graphs = 0
                    model.eval()
                    dataset.setMode("val")

                    with tqdm(val_loader, total=len(val_loader), desc=f"Validation epoch {epoch} of Fold {k}") as val_loader:
                        running_y = torch.Tensor().to(device)
                        running_out = torch.Tensor().to(device)
                        for idx, batch in enumerate(val_loader):
                            batch = batch.to(device)
                            if model_type.endswith('_ph'):
                                out = model(batch)
                                ph = phenotype_entropy_loss(torch.softmax(out.permute(1, 0), 1)) * theta
                            elif model_type.endswith('_zinb'):
                                out, pred, mean, disp, drop = model(batch)
                                loss_zinb = zinb(pred, mean, disp, drop)  * theta
                            elif model_type.endswith('_nb'):
                                out, pred, mean, disp = model(batch)
                                loss_zinb = nb(pred, mean, disp)  * theta
                            else: 
                                out = model(batch)
                            running_y = torch.concatenate((running_y, batch.y.view(out.shape[0], out.shape[1])))
                            running_out = torch.concatenate((running_out, out))
                            if is_log:
                                l = loss(torch.log(out+1), torch.log(batch.y.view(out.shape[0], out.shape[1])+1))
                            else:
                                l = loss(out, batch.y.view(out.shape[0], out.shape[1]))
                            l = alpha * l
                            sim = torch.mean(similarity(out, batch.y.view(out.shape[0], out.shape[1]))) * beta
                            running_loss += l.item() * out.shape[0]
                            running_acc += sim.item() * out.shape[0]
                            num_graphs += out.shape[0]
                            if model_type.endswith('_ph'):
                                running_ph_entropy += ph.item() * out.shape[0]
                                l += 1 - sim + ph
                            elif model_type.endswith('_zinb'):
                                running_zinb += loss_zinb.item() * out.shape[0]
                                l += 1 - sim + loss_zinb
                            else:
                                l += 1 - sim
                            running_total_loss += l.item() * out.shape[0]

                        val_acc = running_acc / num_graphs
                        val_acc_list.append(val_acc)
                        geo_loss = running_loss / num_graphs
                        val_loss_list.append(geo_loss)
                        epoch_loss = running_total_loss / num_graphs
                        val_total_loss_list.append(epoch_loss)
                        statistic, pval = per_gene_pcc(running_out.to('cpu').numpy(), running_y.to('cpu').numpy(), mean=True)
                        val_pcc_statistic_list.append(statistic)
                        val_pcc_pval_list.append(pval)

                        if model_type.endswith('_ph'):
                            ph_entropy = running_ph_entropy / num_graphs
                            val_ph_entropy_list.append(ph_entropy)                           
                            print(f"Val Loss: {epoch_loss:.4f}, MSE Loss: {geo_loss:.4f}, Val Cosine Sim: {val_acc:.4f}, Val Phenotype Entropy: {ph_entropy:.4f}, PCC: {statistic:.4f}, PVAL: {pval:.4f}")
                        elif model_type.endswith('_zinb') or model_type.endswith('_nb'):
                            zinb_loss = running_zinb / num_graphs
                            val_zinb_list.append(zinb_loss)
                            print(f"Val Loss: {epoch_loss:.4f}, MSE Loss: {geo_loss:.4f}, Val Cosine Sim: {val_acc:.4f}, Val ZINB Loss: {zinb_loss:.4f}, PCC: {statistic:.4f}, PVAL: {pval:.4f}")
                        else:
                            print(f"Val Loss: {epoch_loss:.4f}, MSE Loss: {geo_loss:.4f}, Val Cosine Sim: {val_acc:.4f}, PCC: {statistic:.4f}, PVAL: {pval:.4f}")

                        if epoch_loss < best_acc:
                            best_acc = epoch_loss
                            best_run = 0
                            torch.save({
                                "model": model.state_dict(),
                                "opt": optimizer.state_dict(),
                                "mtype": model_type,
                                "train_acc": train_acc_list,
                                "train_list": train_loss_list,
                                "train_ph_entropy": train_ph_entropy_list,
                                "train_total_list": train_total_loss_list,
                                "train_zinb_list": train_zinb_list,
                                "val_acc": val_acc_list,
                                "val_list": val_loss_list,
                                "val_ph_entropy": val_ph_entropy_list,
                                "val_total_list": val_total_loss_list,
                                "val_zinb_list": val_zinb_list,
                                "val_pcc_statistic_list": val_pcc_statistic_list,
                                "val_pcc_pval_list": val_pcc_pval_list,
                                "epoch": epoch
                            }, output_name_model)


        with torch.no_grad():
            running_loss = 0
            running_total_loss = 0
            running_acc = 0
            running_ph_entropy = 0
            running_zinb = 0
            num_graphs = 0
            save_data = torch.load(output_name_model)
            model.load_state_dict(save_data['model'])
            model.eval()
            dataset.setMode(dataset.test)

            with tqdm(test_loader, total=len(test_loader), desc=f"Test of fold {k}") as test_loader:
                running_y = torch.Tensor().to(device)
                running_out = torch.Tensor().to(device)
                for idx, batch in enumerate(test_loader):
                    batch = batch.to(device)
                    if model_type.endswith('_ph'):
                        out = model(batch)
                        ph = phenotype_entropy_loss(torch.softmax(out.permute(1, 0), 1)) * theta
                    elif model_type.endswith('_zinb'):
                        out, pred, mean, disp, drop = model(batch)
                        loss_zinb = zinb(pred, mean, disp, drop)  * theta
                    elif model_type.endswith('_nb'):
                        out, pred, mean, disp = model(batch)
                        loss_zinb = nb(pred, mean, disp)  * theta
                    else:
                        out = model(batch)
                    running_y = torch.concatenate((running_y, batch.y.view(out.shape[0], out.shape[1])))
                    running_out = torch.concatenate((running_out, out))
                    if is_log:
                        l = loss(torch.log(out+1), torch.log(batch.y.view(out.shape[0], out.shape[1])+1))
                    else:
                        l = loss(out, batch.y.view(out.shape[0], out.shape[1]))
                    l = alpha * l
                    sim = torch.mean(similarity(out, batch.y.view(out.shape[0], out.shape[1]))) * beta
                    running_loss += l.item() * out.shape[0]
                    running_acc += sim.item() * out.shape[0]
                    num_graphs += out.shape[0]
                    if model_type.endswith('_ph'):
                        running_ph_entropy += ph.item() * out.shape[0]
                        l += 1 - sim + ph
                    elif model_type.endswith('_zinb'):
                        running_zinb += loss_zinb.item() * out.shape[0]
                        l += 1 - sim + loss_zinb
                    else:
                        l += 1 - sim
                    running_total_loss += l.item() * out.shape[0]

                test_acc = running_acc / num_graphs
                geo_loss = running_loss / num_graphs
                epoch_loss = running_total_loss / num_graphs
                statistic, pval = per_gene_pcc(running_out.to('cpu').numpy(), running_y.to('cpu').numpy(), mean=True)
                if model_type.endswith('_ph'):
                    ph_entropy = running_ph_entropy / num_graphs
                    print(f"Test Loss: {epoch_loss:.4f}, MSE Loss: {geo_loss:.4f}, Test Cosine Sim: {test_acc:.4f}, Test Phenotype Entropy: {ph_entropy:.4f}, PCC: {statistic:.4f}, PVAL: {pval:.4f}")
                elif model_type.endswith('_zinb') or model_type.endswith('_nb'):
                    zinb_loss = running_zinb / num_graphs
                    print(f"Test Loss: {epoch_loss:.4f}, MSE Loss: {geo_loss:.4f}, Test Cosine Sim: {val_acc:.4f}, Test ZINB Loss: {zinb_loss:.4f}, PCC: {statistic:.4f}, PVAL: {pval:.4f}")
                else:
                    print(f"Test Loss: {epoch_loss:.4f}, MSE Loss: {geo_loss:.4f}, Test Cosine Sim: {test_acc:.4f}, PCC: {statistic:.4f}, PVAL: {pval:.4f}")
                
                save_data['test_total_loss'] = epoch_loss
                save_data['test_list'] = geo_loss
                save_data['test_acc'] = test_acc
                save_data['test_pcc_statistic_list'] = statistic
                save_data['test_pcc_pval_list'] = pval
                torch.save(save_data, output_name_model)
