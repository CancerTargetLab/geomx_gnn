import numpy as np
from scipy.stats import pearsonr

def per_gene_pcc(x, y, mean=True):
    x, y = x.astype(np.float64), y.astype(np.float64)
    statistic = np.ndarray(x.shape[-1])
    pval = np.ndarray(x.shape[-1])
    for gene in range(x.shape[-1]):
        pcc = pearsonr(x[:,gene], y[:,gene])
        statistic[gene], pval[gene] = pcc.statistic, pcc.pvalue
    if mean:
        return np.nanmean(statistic), np.nanmean(pval)
    else:
        return statistic, pval
