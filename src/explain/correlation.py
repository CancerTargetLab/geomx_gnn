import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils.stats import per_gene_corr, total_corr

adata_name = 'crc_gat_test'
og_df_name = 'CRC_measurements.csv'
out = 'figures/crc_gat_test'

df = pd.read_csv(os.path.join('data/raw/CRC', og_df_name), header=0, sep=',')
df['Image'] = df['Image'].apply(lambda x: x.split('.')[0])

adata = sc.read_h5ad(os.path.join('out', adata_name+'_all.h5ad'))
var_names = adata.var_names.values
tmp = pd.DataFrame()
tmp['files'] = adata.obs['files'].values
tmp[var_names] = adata.X
adata = tmp
adata['files'] = adata['files'].apply(lambda x: x.split('_')[-1].split('.')[0])
adata = adata[adata['files'].isin(df['Image'])]     #Selects files that exist in pred, in case of only investiating test data
df = df[df['Image'].isin(adata['files'])]
adata = adata.sort_values(by='files').reset_index(drop=True)
df = df.sort_values(by='Image').reset_index(drop=True)

pred = adata[adata.columns[1:].values].values
y = df[adata.columns[1:].values].values

p_statistic, p_pval = per_gene_corr(pred, y, mean=False, method='PEARSONR')
s_statistic, s_pval = per_gene_corr(pred, y, mean=False, method='SPEARMANR')
k_statistic, k_pval = per_gene_corr(pred, y, mean=False, method='KENDALLTAU')
    
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
mean_values = corr_df[corr_df.columns[1:]].mean()
mean_row = pd.DataFrame({'Variable': 'mean', **mean_values}, index=[0])
std_values = corr_df[corr_df.columns[1:]].std()
std_row = pd.DataFrame({'Variable': 'std', **std_values}, index=[0])
corr_df = pd.concat([mean_row, std_row, corr_df], ignore_index=True)

plt.figure(figsize=(10, 5))
plt.table(cellText=corr_df.values, colLabels=corr_df.columns, loc='center')
plt.axis('off')
plt.savefig(os.path.join(out, 'corr_'+adata_name+'.pdf'), bbox_inches='tight')
plt.close()

corr_p = np.ndarray(adata['files'].unique().shape[0])
corr_s = np.ndarray(adata['files'].unique().shape[0])
corr_k = np.ndarray(adata['files'].unique().shape[0])
pval_p = np.ndarray(adata['files'].unique().shape[0])
pval_s = np.ndarray(adata['files'].unique().shape[0])
pval_k = np.ndarray(adata['files'].unique().shape[0])
sorted_ids = sorted(adata['files'].unique().tolist())
for i, id in enumerate(sorted_ids):
    corr_p[i], pval_p[i] = total_corr(adata[adata['files']==id][adata.columns[1:].values].values,
                                        df[df['Image']==id][adata.columns[1:].values].values,
                                        method='PEARSONR')
    corr_s[i], pval_s[i] = total_corr(adata[adata['files']==id][adata.columns[1:].values].values,
                                        df[df['Image']==id][adata.columns[1:].values].values,
                                        method='SPEARMANR')
    corr_k[i], pval_k[i] = total_corr(adata[adata['files']==id][adata.columns[1:].values].values,
                                        df[df['Image']==id][adata.columns[1:].values].values,
                                        method='KENDALLTAU')

correlation_data = {
    'ROIs': sorted_ids,
    'Pearson Correlation Coef.': [corr for corr in corr_p],
    'Pearson p-value': [corr for corr in pval_p],
    'Spearman Correlation Coef.': [corr for corr in corr_s],
    'Spearman p-value': [corr for corr in pval_s],
    'Kendall Correlation Coef.': [corr for corr in corr_k],
    'Kendall p-value': [corr for corr in pval_k]
}

corr_df = pd.DataFrame(correlation_data)
mean_values = corr_df[corr_df.columns[1:]].mean()
mean_row = pd.DataFrame({'ROIs': 'mean', **mean_values}, index=[0])
std_values = corr_df[corr_df.columns[1:]].std()
std_row = pd.DataFrame({'ROIs': 'std', **std_values}, index=[0])
corr_df = pd.concat([mean_row, std_row, corr_df], ignore_index=True)

plt.table(cellText=corr_df.values, colLabels=corr_df.columns, loc='center')
plt.axis('off')
plt.savefig(os.path.join(out, 'corr_cells_'+adata_name+'.pdf'), bbox_inches='tight')
plt.close()
