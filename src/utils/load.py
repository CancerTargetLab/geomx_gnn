import torch

def load(path, save_keys, device='cpu'):
    """
    Load metrics of torch save dict.

    Parameters:
    path (str): Path to torch save dict
    save_keys (list): list containing keys for metrics to extract
    device (str): Location to load torch save dict to
    """
    
    save = torch.load(path, map_location=device)
    if type(save_keys) == list and  type(save) == dict:
        out = {}
        for key in save_keys:
            if key in save.keys():
                out[key] = save[key]
            else:
                print(f'{key} not found in in save {path}')
    elif type(save_keys) == str:
        out = save[save_keys]
    return out
    