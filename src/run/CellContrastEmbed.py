import torch
from src.data.CellContrastData import EmbedDataset
from src.models.CellContrastModel import ContrastiveLearning
from src.utils.load import load
from src.utils.setSeed import set_seed

def embed(image_dir, model_name, args):
    """
    Embed visual representations of cells.

    Parameters:
    image_dir (str): Path to dir in which torch.tensors of cell cut outs are
    model_name (str): Path and name of model torch save dict
    args (dict): Arguments
    """

    batch_size = args['batch_size_image']
    seed = args['seed']

    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    train_dataset = EmbedDataset(root_dir=image_dir,
                                 split='train',
                                 crop_factor=args['crop_factor'])
    test_dataset = EmbedDataset(root_dir=image_dir,
                                 split='test',
                                 crop_factor=args['crop_factor'])

    model = ContrastiveLearning(channels=train_dataset.__getitem__(0)[0].shape[0],
                                embed=args['embedding_size_image'],
                                contrast=args['contrast_size_image'], 
                                resnet=args['resnet_model']).to(device, torch.float32)
    model.load_state_dict(load(model_name, save_keys='model', device=device))
    model.eval()
    model.mode = 'embed'

    train_dataset.save_embed_data(model, device=device, batch_size=batch_size)
    test_dataset.save_embed_data(model, device=device, batch_size=batch_size)
