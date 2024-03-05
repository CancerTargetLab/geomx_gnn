import os
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.stats import per_gene_corr

adata_name = '_crc_lin_sub'
og_df_name = 'CRC_measurements.csv'
out = 'figrues/crc_lin_sub'

df = pd.read_csv(os.path.join('data/raw/CRC', adata_name, header=0, sep=','))

adata = sc.read_h5ad(os.path.join('out', adata_name+'_all.h5ad'))
var_names = adata.var_names.values

pred = adata.X
y = df[df.columns[4:].values].values

p_statistic, p_pval = per_gene_corr(pred, y, mean=False, method='pearsonr')
s_statistic, s_pval = per_gene_corr(pred, y, mean=False, method='pearsonr')
k_statistic, k_pval = per_gene_corr(pred, y, mean=False, method='pearsonr')
    
correlation_data = {
    'Variable': var_names,
    'Pearson Correlation Coef.': [corr for corr in p_statistic],
    'Pearson p-value': [corr for corr in p_pval],
    'Spearman Correlation Coef.': [corr for corr in s_statistic],
    'Spearman p-value': [corr for corr in s_pval],
    'Kendall Correlation Coef.': [corr for corr in k_statistic],
    'Kendall p-value': [corr for corr in k_pval]
}

corr_df = pd.DataFrame(correlation_data)

plt.figure(figsize=(10, 5))
plt.table(cellText=corr_df.values, colLabels=corr_df.columns, loc='center')
plt.axis('off')
plt.savefig(os.path.join(out, 'corr'+adata_name+'.png'))
