import torch

def phenotype_entropy_loss(x):
    return ((-x/x.sum(-1, keepdim=True)*torch.log(x/x.sum(-1, keepdim=True)+1e-15)).sum(dim=-1)/torch.log(torch.tensor(x.shape[1], dtype=torch.float32, device=x.device))).mean()