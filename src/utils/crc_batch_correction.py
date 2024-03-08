import os
import pandas as pd
import scanpy as sc
import numpy as np


og_df_name = 'CRC_measurements.csv'

df = pd.read_csv(os.path.join('data/raw/CRC', og_df_name, header=0, sep=','))
adata = sc.AnnData(df[df.columns[4:].values].values)
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


