import numpy as np
from scipy.stats import pearsonr

def per_gene_pcc(x, y, mean=True):
    statistic = np.ndarray(x.shape[-1])
    pval = np.ndarray(x.shape[-1])
    for gene in x.shape[-1]:
        pcc = pearsonr(x[:,gene], y[:,gene])
        statistic[gene], pval[gene] = pcc.statistic, pcc.pvalue
    if mean:
        return np.mean(statistic), np.mean(pval)
    else:
        return statistic, pval
