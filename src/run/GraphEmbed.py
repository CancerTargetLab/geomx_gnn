from src.data.GeoMXData import GeoMXDataset
from src.data.ImageGraphData import ImageGraphDataset
from src.models.GraphModel import ROIExpression,ROIExpression_Image, Lin
from src.utils.utils import set_seed
import torch
import os

def embed(**args):
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
    model_name = args['output_name']
    output_dir = args['output_graph_embed']

    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = args['model_type']
    set_seed(SEED)

    if args['embed_graph_train_data']:
        split = 'train'
    else:
        split = 'test'

    if 'IMAGE' in model_type:
        dataset = ImageGraphDataset(split=split,
                                    num_folds=args['num_cfolds'],
                                    embed=True,
                                    **args)
    else:
        dataset = GeoMXDataset(split=split,
                            num_folds=args['num_cfolds'],
                            **args)

    if 'IMAGE' in model_type:
        model = ROIExpression_Image(channels=dataset.get(0).x.shape[1],
                                    num_out_features=dataset.get(0).y.shape[0],
                                    **args).to(device, dtype=torch.float32)
    elif 'Image2Count' in model_type:
        model = ROIExpression(num_out_features=dataset.get(0).y.shape[0],
                            **args).to(device, dtype=torch.float32)
    elif 'LIN' in model_type:
            model = Lin(num_out_features=dataset.get(0).y.shape[0],
                        **args).to(device, dtype=torch.float32)
    else:
        raise Exception(f'{model_type} not a valid model type, must be one of Image2Count, IMAGEImage2Count, LIN')
    model.eval()
    model.load_state_dict(torch.load(model_name, weights_only=False, map_location=device)['model'])
    if not os.path.exists(output_dir) and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if 'IMAGE' in model_type:
        dataset.embed(model, output_dir, device=device, batch_size=args['batch_size'])
    else:
        dataset.embed(model, output_dir, device='cpu')
