from src.data.GeoMXData import GeoMXDataset
from src.models.GraphModel import ROIExpression, ROIExpression_lin
from src.utils.setSeed import set_seed
import torch
import os

def embed(raw_subset_dir, label_data, model_name, output_dir, args):

    SEED = args['seed']

    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = args['graph_model_type']
    set_seed(SEED)

    dataset = GeoMXDataset(root_dir=args['graph_dir'],
                           raw_subset_dir=raw_subset_dir,
                           train_ratio=args['train_ratio_graph'],
                           val_ratio=args['val_ratio_graph'],
                           node_dropout=args['node_dropout'],
                           edge_dropout=args['edge_dropout'],
                           label_data=label_data)

    if 'GAT' in model_type:
        model = ROIExpression(layers=args['layers_graph'],
                            num_node_features=args['num_node_features'],
                            num_edge_features=args['num_edge_features'],
                            num_embed_features=args['num_embed_features'],
                            embed_dropout=args['embed_dropout_graph'],
                            conv_dropout=args['conv_dropout_graph'],
                            num_out_features=dataset.get(0).y.shape[0],
                            heads=args['heads_graph']).to(device, dtype=torch.float32)
    elif 'LIN' in model_type:
        model = ROIExpression_lin(layers=args['layers_graph'],
                            num_node_features=args['num_node_features'],
                            num_embed_features=args['num_embed_features'],
                            embed_dropout=args['embed_dropout_graph'],
                            conv_dropout=args['conv_dropout_graph'],
                            num_out_features=dataset.get(0).y.shape[0]).to(device, dtype=torch.float32)
    else:
        raise Exception(f'{model_type} not a valid model type, must be one of GAT, GAT_ph, LIN, LIN_ph')
    model.eval()
    model.load_state_dict(torch.load(model_name)['model'])
    if not os.path.exists(output_dir) and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    dataset.embed(model, output_dir, device=device)