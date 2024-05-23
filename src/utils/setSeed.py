import random
import numpy as np
import torch

def set_seed(seed, cuda_reproduce=True):
    """
    Set seed for random,  numpy, torch and sklearn.
    
    Parameters:
    seed (int): Seed
    cuda_reproduce (bool): Wether or not to use cuda reproducibility
    """
    
    # Set seed for Python's random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For CUDA devices, if available
    
    # Additional settings for reproducibility
    if cuda_reproduce:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
