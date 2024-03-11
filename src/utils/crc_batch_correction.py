import os
import pandas as pd
import scanpy as sc
import numpy as np


og_df_name = 'CRC_measurements.csv'

df = pd.read_csv(os.path.join('data/raw/CRC', og_df_name, header=0, sep=','))
adata = sc.AnnData(df[df.columns[4:].values].values)
adata.var_names = df.columns[4:].values
adata.obs['files'] = df['Image'].values
adata.obs['Centroid.X.px'] = df['Centroid.X.px'].values
adata.obs['Centroid.Y.px'] = df['Centroid.Y.px'].values

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=10, min_disp=0.5, batch_key = 'files')
print("Highly variable genes intersection: %d"%sum(adata.var.highly_variable_intersection))
print("Number of batches where gene is variable:")
print(adata.var.highly_variable_nbatches.value_counts())
var_genes_batch = adata.var.highly_variable_nbatches > 0

sc.pp.pca(adatasvd_solver='arpack', n_comps=adata.X.shape[1]-1 if adata.X.shape[1]-1 < 30 else 30, chunked=True, chunk_size=20000, use_highly_variable=True)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=adata.varm['PCs'].shape[1] if adata.X.shape[1]-1 < 30 else 30)
sc.tl.umap(adata)
#sc.tl.tsne(adata, n_pcs=30)

adata_combat = adata.copy()

sc.pp.combat(adata_combat, key='files')

sc.pp.highly_variable_genes(adata_combat, min_mean=0.0125, max_mean=10, min_disp=0.5,)
sc.pp.pca(adata_combat, svd_solver='arpack', n_comps=adata.X.shape[1]-1 if adata.X.shape[1]-1 < 30 else 30, chunked=True, chunk_size=20000, use_highly_variable=True)
sc.pp.neighbors(adata_combat, n_neighbors=10, n_pcs=adata.varm['PCs'].shape[1] if adata.X.shape[1]-1 < 30 else 30)
sc.tl.umap(adata_combat)
#sc.tl.tsne(adata_combat, n_pcs = 30)

sc.pl.umap(adata, color='files')
sc.pl.umap(adata, color=adata.var_names.values)
sc.pl.umap(adata_combat, color='files')
sc.pl.umap(adata_combat, color=adata.var_names.values)
