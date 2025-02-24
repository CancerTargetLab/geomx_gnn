from src.data.GeoMXData import GeoMXDataset
from src.data.ImageGraphData import ImageGraphDataset
from src.models.GraphModel import ROIExpression, ROIExpression_Image, Lin
from src.optimizer.grokfast import gradfilter_ema
from src.loss.CellEntropyLoss import phenotype_entropy_loss
from src.utils.utils import set_seed
from src.utils.utils import per_gene_pcc
from torch_geometric.loader import DataLoader
import torch
import os
from tqdm import tqdm

def train(**args):
    """
    Train model to predict sc expression of cells.

    Parameters:
    raw_subset_dir (str): name of dir in which torch.tensors of visual cell embeddings are
    label_data (str): Name of .csv in raw/ containing label information of ROIs
    output_name (str): Path and name of model torch save dict
    args (dict): Arguments
    """

    output_name = args['output_name']

    EPOCH = args['epochs']
    SEED = args['seed']
    model_type = args['model_type']
    batch_size = args['batch_size']
    lr = args['lr']
    weight_decay = args['weight_decay']
    num_workers = args['num_workers']
    early_stopping = args['early_stopping']
    is_log = args['data_use_log_graph']
    alpha = args['graph_mse_mult']
    beta = args['graph_cos_sim_mult']

    if EPOCH > 900: # Weird bug, when running for more then 982 epochs we get an recursion error
        import sys
        sys.setrecursionlimit(100000)
    
    if args['num_cfolds'] > 1 and not os.path.isdir(os.path.join(output_name.split('.')[0])):
        os.makedirs(os.path.join(output_name.split('.')[0]))

    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args['deterministic']:
        set_seed(SEED)

    #Wether to train together with image model
    if 'IMAGE' in model_type:
        train_dataset = ImageGraphDataset(split='train',
                                    num_folds=args['num_cfolds'],
                                    **args)
        test_dataset = ImageGraphDataset(split='test',
                                    num_folds=1,
                                    **args)
    else:
        train_dataset = GeoMXDataset(split='train',
                            num_folds=args['num_cfolds'],
                            **args)
        test_dataset = GeoMXDataset(split='test',
                            num_folds=1,
                            **args)

    num_folds = args['num_cfolds'] if args['num_cfolds'] > 1 else 1
    for k in range(num_folds):
        if args['num_cfolds'] > 1:
            output_name_model = os.path.join(output_name.split('.')[0], f'{k}'+'.'+output_name.split('.')[-1])
        else:
            output_name_model = output_name
        train_dataset.set_fold_k()
        train_dataset.setMode(train_dataset.train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        train_dataset.setMode(train_dataset.val)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_dataset.setMode(test_dataset.test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        grads = None

        if 'IMAGE' in model_type:
            model = ROIExpression_Image(channels=train_dataset.get(0).x.shape[1],
                                        num_out_features=train_dataset.get(0).y.shape[0],
                                        **args).to(device, dtype=torch.float32)
        elif 'Image2Count' in model_type:
            model = ROIExpression(num_out_features=train_dataset.get(0).y.shape[0],
                                **args).to(device, dtype=torch.float32)
        elif 'LIN' in model_type:
            model = Lin(num_out_features=train_dataset.get(0).y.shape[0],
                        **args).to(device, dtype=torch.float32)
        else:
            raise Exception(f'{model_type} Image2Count, IMAGEImage2Count, LIN')
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr,
                                     weight_decay=weight_decay)
        train_dataset.setMode(train_dataset.train)

        loss = torch.nn.L1Loss()
        similarity = torch.nn.CosineSimilarity()

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
            num_graphs = 0
            model.train()
            train_dataset.setMode(train_dataset.train)

            if best_run < early_stopping:
                with tqdm(train_loader, total=len(train_loader), desc=f"Training epoch {epoch} of Fold {k}") as train_loader:
                    for idx, batch in enumerate(train_loader):
                        batch = batch.to(device)
                        optimizer.zero_grad()
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
                        l += 1 * beta - sim
                        running_total_loss += l.item() * out.shape[0]
                        l.backward()
                        #grads = gradfilter_ema(model, grads=grads, alpha=0.75, lamb=2)
                        optimizer.step()

                    train_acc = running_acc / num_graphs
                    train_acc_list.append(train_acc)
                    geo_loss = running_loss / num_graphs
                    train_loss_list.append(geo_loss)
                    epoch_loss = running_total_loss / num_graphs
                    train_total_loss_list.append(epoch_loss)
                    print(f"Train Loss: {epoch_loss:.4f}, MSE Loss: {geo_loss:.4f}, Train Cosine Sim: {train_acc:.4f}")

                with torch.no_grad():
                    running_loss = 0
                    running_total_loss = 0
                    running_acc = 0
                    num_graphs = 0
                    model.eval()
                    train_dataset.setMode(train_dataset.val)

                    with tqdm(val_loader, total=len(val_loader), desc=f"Validation epoch {epoch} of Fold {k}") as val_loader:
                        running_y = torch.Tensor().to(device)
                        running_out = torch.Tensor().to(device)
                        for idx, batch in enumerate(val_loader):
                            batch = batch.to(device)
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
                            l += 1 * beta - sim
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
            num_graphs = 0
            save_data = torch.load(output_name_model, weights_only=False)
            model.load_state_dict(save_data['model'])
            model.eval()
            test_dataset.setMode(test_dataset.test)

            with tqdm(test_loader, total=len(test_loader), desc=f"Test of fold {k}") as test_loader:
                running_y = torch.Tensor().to(device)
                running_out = torch.Tensor().to(device)
                for idx, batch in enumerate(test_loader):
                    batch = batch.to(device)
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
                    l += 1 * beta - sim
                    running_total_loss += l.item() * out.shape[0]

                test_acc = running_acc / num_graphs
                geo_loss = running_loss / num_graphs
                epoch_loss = running_total_loss / num_graphs
                statistic, pval = per_gene_pcc(running_out.to('cpu').numpy(), running_y.to('cpu').numpy(), mean=True)
                print(f"Test Loss: {epoch_loss:.4f}, MSE Loss: {geo_loss:.4f}, Test Cosine Sim: {test_acc:.4f}, PCC: {statistic:.4f}, PVAL: {pval:.4f}")
                
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
                            "epoch": save_data['epoch'],
                            "test_total_list": epoch_loss,
                            "test_list": geo_loss,
                            "test_acc": test_acc,
                            "test_pcc_statistic_list": statistic,
                            "test_pcc_pval_list": pval,
                            "args": args
                        }, output_name_model)
