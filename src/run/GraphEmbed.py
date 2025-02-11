from src.data.GeoMXData import GeoMXDataset
from src.data.ImageGraphData import ImageGraphDataset
from src.models.GraphModel import ROIExpression,ROIExpression_Image, Lin
from src.utils.setSeed import set_seed
import torch
import os

def embed(raw_subset_dir, label_data, model_name, output_dir, args):
    """
    Embed predicted sc expression of cells.

    Parameters:
    raw_subset_dir (str): name of dir in which torch.tensors of visual cell embeddings are
    label_data (str): Name of .csv in raw/ containing label information of ROIs
    model_name (str): Path and name of model torch save dict
    output_dir (str): Path to dir to save sc expression embeddings
    args (dict): Arguments
    """

    SEED = args['seed']

    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = args['graph_model_type']
    set_seed(SEED)

    if args['embed_graph_train_data']:
        split = 'train'
    else:
        split = 'test'

    if 'IMAGE' in model_type:
        dataset = ImageGraphDataset(root_dir=args['graph_dir'],
                                    split=split,
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
                                    output_name=None,
                                    embed=True)
    else:
        dataset = GeoMXDataset(root_dir=args['graph_dir'],
                            split=split,
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
                            output_name=None)

    if 'IMAGE' in model_type:
        model = ROIExpression_Image(channels=dataset.get(0).x.shape[1],
                                        embed=args['embedding_size_image'],
                                        contrast=args['contrast_size_image'], 
                                        resnet=args['resnet_model'],
                                        lin_layers=args['lin_layers_graph'],
                                        gat_layers=args['gat_layers_graph'],
                                        num_edge_features=args['num_edge_features'],
                                        num_embed_features=args['num_embed_features'],
                                        num_gat_features=args['num_gat_features'],
                                        num_out_features=dataset.get(0).y.shape[0],
                                        heads=args['heads_graph'],
                                        embed_dropout=args['embed_dropout_graph'],
                                        conv_dropout=args['conv_dropout_graph'],
                                        path_image_model=args['init_image_model'],
                                        path_graph_model=args['init_graph_model']).to(device, dtype=torch.float32)
    elif 'Image2Count' in model_type:
        model = ROIExpression(lin_layers=args['lin_layers_graph'],
                            gat_layers=args['gat_layers_graph'],
                            num_node_features=args['num_node_features'],
                            num_edge_features=args['num_edge_features'],
                            num_embed_features=args['num_embed_features'],
                            num_gat_features=args['num_gat_features'],
                            embed_dropout=args['embed_dropout_graph'],
                            conv_dropout=args['conv_dropout_graph'],
                            num_out_features=dataset.get(0).y.shape[0],
                            heads=args['heads_graph']).to(device, dtype=torch.float32)
    elif 'LIN' in model_type:
            model = Lin(num_node_features=args['num_node_features'],
                        num_out_features=dataset.get(0).y.shape[0]).to(device, dtype=torch.float32)
    else:
        raise Exception(f'{model_type} not a valid model type, must be one of Image2Count, IMAGEImage2Count, LIN')
    model.eval()
    model.load_state_dict(torch.load(model_name, weights_only=False)['model'])
    if not os.path.exists(output_dir) and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if 'IMAGE' in model_type:
        dataset.embed(model, output_dir, device=device, batch_size=args['batch_size_image'], return_mean='mean' in model_type)
    else:
        dataset.embed(model, output_dir, device='cpu', return_mean='mean' in model_type)
