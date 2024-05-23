import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

def per_gene_pcc(x, y, mean=True):
    #print(f'Use src.utils.stats.per_gene_corr instead of this')
    return per_gene_corr(x, y, mean=mean, method='pearsonr')
    
def per_gene_corr(x, y, mean=True, method='pearsonr'):
    if method.upper() == 'PEARSONR':
        corr = pearsonr
    elif method.upper() == 'SPEARMANR':
        corr = spearmanr
    elif  method.upper() == 'KENDALLTAU':
        corr = kendalltau
    else:
        raise Exception(f'Method {method} not one of pearsonr, spearmanr or kendalltau')
    x, y = x.astype(np.float64), y.astype(np.float64)
    statistic = np.ndarray(x.shape[-1])
    pval = np.ndarray(x.shape[-1])
    for gene in range(x.shape[-1]):
        pcc = corr(x[:,gene], y[:,gene])
        statistic[gene], pval[gene] = pcc.statistic, pcc.pvalue
    if mean:
        return np.nanmean(statistic), np.nanmean(pval)
    else:
        return statistic, pval

def avg_cell_n(path):
      import os
      import torch
      files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.pt')]
      num_cells = 0
      for file in files:
            num_cells += torch.load(file).x.shape[0]
      return num_cells/len(files)