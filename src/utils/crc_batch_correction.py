import os
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt


og_df_name = 'CRC_measurements.csv'
save = 'figures/crc_og/'
csv_out = 'data/raw/'

df = pd.read_csv(os.path.join('data/raw/CRC', og_df_name), header=0, sep=',')
adata = sc.AnnData(df[df.columns[4:].values].values)
adata.var_names = df.columns[4:].values
adata.obs['files'] = df['Image'].values
adata.obs['Centroid.X.px'] = df['Centroid.X.px'].values
adata.obs['Centroid.Y.px'] = df['Centroid.Y.px'].values

cell_index = np.random.default_rng(42).choice(np.arange(adata.shape[0]), size=50000, replace=False)

sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
sc.pl.violin(adata, ['total_counts'], show=False)
plt.savefig(os.path.join(save, f'total_counts.png'))
plt.close()
#adata = adata[adata.obs.total_counts < 200 and adata.obs.total_counts > 135,:]

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=10, min_disp=0.5, batch_key = 'files')
print("Highly variable genes intersection: %d"%sum(adata.var.highly_variable_intersection))
print("Number of batches where gene is variable:")
print(adata.var.highly_variable_nbatches.value_counts())

df = pd.DataFrame()
df['Image'] = adata.obs['files'].values
df['Centroid.X.px'] = adata.obs['Centroid.X.px'].values
df['Centroid.Y.px'] = adata.obs['Centroid.Y.px'].values
df['Class'] = ''
df[adata.var_names.values] = adata.X
df.to_csv(os.path.join(csv_out, 'CRC_measurements_nt_rmo.csv'), sep=',', header=True, index=False,)

sc.pp.pca(adata,
          svd_solver='arpack',
          n_comps=adata.X.shape[1]-1 if adata.X.shape[1]-1 < 30 else 30,
          chunked=True,
          chunk_size=20000,
          use_highly_variable=True)
sc.pl.pca_variance_ratio(adata,
                         log=True,
                         n_pcs=adata.X.shape[1]-1 if adata.X.shape[1]-1 < 30 else 30,
                         show=False)
plt.savefig(os.path.join(save, f'crc_og_pca_variance.png'))
plt.close()
s_adata = sc.AnnData(adata.X[cell_index],
                     obs=adata.obs.iloc[cell_index],
                     var=adata.var,
                     uns=adata.uns,
                     varm=adata.varm,
                     obsm={'X_pca': adata.obsm['X_pca'][cell_index]})
sc.pp.neighbors(s_adata,
                n_neighbors=10,
                n_pcs=adata.varm['PCs'].shape[1] if adata.X.shape[1]-1 < 30 else 30)
sc.tl.umap(s_adata)
#sc.tl.tsne(adata, n_pcs=30)

adata_combat = adata.copy()

sc.pp.combat(adata_combat, key='files')

df = pd.DataFrame()
df['Image'] = adata_combat.obs['files'].values
df['Centroid.X.px'] = adata_combat.obs['Centroid.X.px'].values
df['Centroid.Y.px'] = adata_combat.obs['Centroid.Y.px'].values
df['Class'] = ''
df[adata_combat.var_names.values] = adata_combat.X
df.to_csv(os.path.join(csv_out, 'CRC_measurements_combat_nt_rmo.csv'), sep=',', header=True, index=False,)

sc.pp.highly_variable_genes(adata_combat, min_mean=0.0125, max_mean=10, min_disp=0.5,)
sc.pp.pca(adata_combat,
          svd_solver='arpack',
          n_comps=adata.X.shape[1]-1 if adata.X.shape[1]-1 < 30 else 30,
          chunked=True,
          chunk_size=20000,
          use_highly_variable=True)
sc.pl.pca_variance_ratio(adata_combat,
                         log=True,
                         n_pcs=adata.X.shape[1]-1 if adata.X.shape[1]-1 < 30 else 30,
                         show=False)
plt.savefig(os.path.join(save, f'crc_combat_pca_variance.png'))
plt.close()
s_adata_combat = sc.AnnData(adata_combat.X[cell_index],
                     obs=adata_combat.obs.iloc[cell_index],
                     var=adata_combat.var,
                     uns=adata_combat.uns,
                     varm=adata_combat.varm,
                     obsm={'X_pca': adata_combat.obsm['X_pca'][cell_index]})
sc.pp.neighbors(s_adata_combat,
                n_neighbors=10,
                n_pcs=adata.varm['PCs'].shape[1] if adata.X.shape[1]-1 < 30 else 30)
sc.tl.umap(s_adata_combat)
#sc.tl.tsne(adata_combat, n_pcs = 30)

sc.pl.umap(s_adata, color='files', show=False)
plt.savefig(os.path.join(save, f'crc_og_subset_umap_id.png'))
plt.close()
sc.pl.umap(s_adata, color=adata.var_names.values, show=False)
plt.savefig(os.path.join(save, f'crc_og_subset_umap_expr.png'))
plt.close()
sc.pl.umap(s_adata_combat, color='files', show=False)
plt.savefig(os.path.join(save, f'crc_combat_subset_umap_id.png'))
plt.close()
sc.pl.umap(s_adata_combat, color=adata.var_names.values, show=False)
plt.savefig(os.path.join(save, f'crc_combat_subset_umap_expr.png'))
plt.close()
