from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.data.CellContrastData import EmbedDataset
from src.models.CellContrastModel import ContrastiveLearning
from src.utils.load import load
from src.utils.setSeed import set_seed

def embed(image_dir, model_name, args):

    batch_size = args['batch_size_image']
    seed = args['seed']

    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    dataset = EmbedDataset(root_dir=image_dir, 
                           crop_factor=args['crop_factor'],
                           train_ratio=args['train_ratio_image'],
                           val_ratio=args['val_ratio_image'])
    dataset.setMode(dataset.embed)

    model = ContrastiveLearning(channels=dataset.__getitem__(0).shape[0],
                                embed=args['embedding_size_image'],
                                contrast=args['contrast_size_image'], 
                                resnet=args['resnet_model']).to(device, torch.float32)
    model.load_state_dict(load(model_name, save_keys='model', device=device))
    model.eval()
    model.mode = 'embed'

    dataset.save_embed_data(model, device=device, batch_size=batch_size)
