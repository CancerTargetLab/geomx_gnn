from src.data.GeoMXData import GeoMXDataset
from src.models.GraphModel import ROIExpression
from src.utils.setSeed import set_seed
import torch
import os

def embed(args):

    SEED = args['seed']

    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(SEED)

    dataset = GeoMXDataset(root_dir=args['graph_dir'],
                           raw_subset_dir=args['graph_raw_subset_dir'],
                           train_ratio=args['train_ratio_graph'],
                           val_ratio=args['val_ratio_graph'],
                           node_dropout=args['node_dropout'],
                           edge_dropout=args['edge_dropout'],
                           label_data=args['graph_label_data'])

    model = ROIExpression(layers=args['layers_graph'],
                          num_node_features=args['num_node_features'],
                          num_edge_features=args['num_edge_features'],
                          num_embed_features=args['num_embed_features'],
                          embed_dropout=args['embed_dropout_graph'],
                          conv_dropout=args['conv_dropout_graph'],
                          num_out_features=dataset.get(0).y.shape[0],
                          heads=args['heads_graph']).to(device, dtype=float)
    model.eval()
    model.load_state_dict(torch.load(args['output_name_graph'])['model'])
    if not os.path.exists(args['output_graph_embed']) and not os.path.isdir(args['output_graph_embed']):
        os.makedirs(args['output_graph_embed'])
    dataset.embed(model, args['output_graph_embed'], device=device)