import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

def per_gene_pcc(x, y, mean=True):
    """
    Calculate pearsonr correlation between x and y on a gene/protein wise level.

    Parameters:
    x (np.array): 2D number array
    y (np.array): 2D number array
    mean (bool): Wether to calculate mean gene/protein correlation

    Returns:
    (Correlation value,  p-value)
    """

    #print(f'Use src.utils.stats.per_gene_corr instead of this')
    return per_gene_corr(x, y, mean=mean, method='pearsonr')
    
def per_gene_corr(x, y, mean=True, method='pearsonr'):
    """
    Calculate correlation of method between x and y on a gene/protein wise level.

    Parameters:
    x (np.array): 2D number array
    y (np.array): 2D number array
    mean (bool): Wether to calculate mean gene/protein correlation
    method (str): String indicating method to be used ('PEARSONR', 'SPEARMANR', 'KENDALLTAU')

    Returns:
    (Correlation value,  p-value)
    """

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
      """
      Calculate average number of cells per graph for graphs in directory.

      Parameters:
      path (str): String path to directory containing torch graphs

      Returns:
      float: Average number of cells per graph
      """
      
      import os
      import torch
      files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.pt')]
      num_cells = 0
      for file in files:
            num_cells += torch.load(file).x.shape[0]
      return num_cells/len(files)