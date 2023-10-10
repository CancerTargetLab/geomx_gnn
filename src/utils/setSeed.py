import random
import numpy as np
import torch
from sklearn.utils import check_random_state

def set_seed(seed, cuda_reproduce=True):
    """
    Set seed for random,  numpy, torch and sklearn.
    Usage:
    set_seed(42)
    """
    # Set seed for Python's random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for scikit-learn (sklearn)
    rng = check_random_state(seed)
    
    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For CUDA devices, if available
    
    # Additional settings for reproducibility
    if cuda_reproduce:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
