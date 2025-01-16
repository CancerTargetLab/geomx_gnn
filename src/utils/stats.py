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

def _get_method(method):
    if method.upper() == 'PEARSONR':
        corr = pearsonr
    elif method.upper() == 'SPEARMANR':
        corr = spearmanr
    elif  method.upper() == 'KENDALLTAU':
        corr = kendalltau
    else:
        raise Exception(f'Method {method} not one of pearsonr, spearmanr or kendalltau')
    return corr
    
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

    corr =  _get_method(method)
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

def total_corr(x, y, method='pearsonr'):
    """
    Calculate correlation of method between x and y on a gene/protein wise level.

    Parameters:
    x (np.array): 2D number array
    y (np.array): 2D number array
    method (str): String indicating method to be used ('PEARSONR', 'SPEARMANR', 'KENDALLTAU')

    Returns:
    (Correlation value,  p-value)
    """

    corr =  _get_method(method)
    x, y = x.astype(np.float64).squeeze(), y.astype(np.float64).squeeze()
    statistic, pval = corr(x, y)
    return np.mean(statistic), np.mean(pval)

def corr_all2all(adata, method='pearsonr'):
    """
    Calculate correlation of method between x and y on a gene/protein wise level.

    Parameters:
    adata (sc.AnnData): AnnData obj
    method (str): String indicating method to be used ('PEARSONR', 'SPEARMANR', 'KENDALLTAU')

    Returns:
    (Correlation value,  p-value)
    """

    corr =  _get_method(method)
    statistic = np.zeros((adata.var_names.values.shape[0],adata.var_names.values.shape[0]))
    pval = np.zeros((adata.var_names.values.shape[0],adata.var_names.values.shape[0]))
    for i in range(statistic.shape[0]):
        for j in range(statistic.shape[0]):
            if j >= i:
                statistic[i,j], pval[i,j] = corr(adata.X[i], adata.X[j])
            else:
                statistic[i,j], pval[i,j] = statistic[j,i], pval[j,i]
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
            num_cells += torch.load(file, weights_only=False).x.shape[0]
      return num_cells/len(files)

def avg_edge_len(path):
      """
      Calculate average number of edge lengths per graph for graphs in directory.

      Parameters:
      path (str): String path to directory containing torch graphs

      Returns:
      float: Average number of cells per graph
      """
      
      import os
      import torch
      files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.pt')]
      num_edges = 0
      edge_sum = 0
      for file in files:
            graph = torch.load(file, weights_only=False)
            num_edges += graph.edge_attr.shape[0]
            edge_sum += torch.sum(graph.edge_attr)
      return edge_sum/num_edges

def avg_roi_area(path):
      """
      Calculate average number of area per graph for graphs in directory. Needs work to be correct.

      Parameters:
      path (str): String path to directory containing torch graphs

      Returns:
      float: Average number of area per graph
      """
      
      import os
      import torch
      files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.pt')]
      area = 0
      area_list = []
      for file in files:
            graph = torch.load(file, weights_only=False)
            tmp = (torch.max(graph.pos[:,0])-torch.min(graph.pos[:,0]))*(torch.max(graph.pos[:,1])-torch.min(graph.pos[:,1]))
            area += tmp
            area_list.append(tmp.item())
      return area/len(files), torch.median(torch.tensor(area_list))