import os
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd

measurement_name = 'CRC_measurements.csv'
h5ad_name = '_crc_gat_l1_slow'
figure_dir = 'figures/crc_gat_l1_slow'

subdir = 'corr_plots'
if not (os.path.exists(os.path.join(figure_dir, subdir)) and os.path.isdir(os.path.join(figure_dir, subdir))):
    os.makedirs(os.path.join(figure_dir, subdir))

df = pd.read_csv(os.path.join('data/raw/CRC/', measurement_name))
df['Image'] = df['Image'].apply(lambda x: x.split('.')[0])
df = df.sort_values(by='Image').reset_index(drop=True)

adata = sc.read_h5ad(os.path.join('out/', h5ad_name+'_all.h5ad'))
var_names = adata.var_names.values
tmp = pd.DataFrame()
tmp['files'] = adata.obs['files'].values
tmp[var_names] = adata.X
adata = tmp
adata['files'] = adata['files'].apply(lambda x: x.split('_')[-1].split('.')[0])
adata = adata.sort_values(by='files').reset_index(drop=True)

categories = np.unique(df['Image'].values)
colors = np.linspace(0, 1, len(categories))
colordict = dict(zip(categories, colors))
adata['Color'] = df['Image'].apply(lambda x: colordict[x])

cell_index = np.random.default_rng(42).choice(np.arange(adata.shape[0]), size=50000, replace=False)

for name in var_names:
    plt.scatter(df[name].values[cell_index], adata[name].values[cell_index], c=adata['Color'].values[cell_index], cmap='gist_ncar')
    plt.legend()
    plt.xlabel('True value')
    plt.ylabel('Pred value')
    plt.title(h5ad_name.split('_')[-1])
    plt.savefig(os.path.join(figure_dir, subdir, h5ad_name.split('_')[-1]+name+'.png'))
    plt.close()