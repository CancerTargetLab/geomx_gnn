import os
from skimage import io
import numpy as np
import pandas as pd
import squidpy as sq
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt

raw_subset_dir = 'CRC_1p'
name_tiff = 'CRC03.ome.tif'
figure_dir = 'figures/crc_lin_1p_long/crc03_s'
vis_name = '_crc_lin_1p_long_all.h5ad'
vis_name_og = '_crc_1p_og_all.h5ad'
proteins = 'Hoechst1,CD3,Ki67,CD4,CD20,CD163,Ecadherin,LaminABC,PCNA,NaKATPase,Keratin,CD45,CD68,FOXP3,Vimentin,Desmin,Ki67_570,CD45RO,aSMA,PD1,CD8a,PDL1,CDX2,CD31,Collagen'
proteins = proteins.replace('.', ' ').split(',')
crop_coord = [(23701, 14729, 27393, 18421)]#[(23801, 14729, 27493, 19421)]

if not os.path.exists(figure_dir) and not os.path.isdir(figure_dir):
    os.makedirs(figure_dir)

#crop_coord=[(22201, 14729, 25893, 18421)]
path = os.path.join('data/raw', raw_subset_dir)
df_path = [os.path.join(path, p) for p in os.listdir(path) if p.endswith(('.csv'))][0]
df = pd.read_csv(df_path, header=0, sep=",")
df = df[["Image", "Centroid.X.px", "Centroid.Y.px"]] #'Class'
df = df[df["Image"] == name_tiff]
df = df.drop("Image", axis=1)
mask = ~df.duplicated(subset=['Centroid.X.px', 'Centroid.Y.px'], keep=False) | ~df.duplicated(subset=['Centroid.X.px', 'Centroid.Y.px'], keep='first')
df = df[mask]

img = np.zeros((24000, 33000, 1), dtype=np.uint16)
counts = np.random.default_rng(42).integers(0, 15, size=(df.shape[0], 1))
coordinates = np.column_stack((df["Centroid.X.px"].to_numpy(), df["Centroid.Y.px"].to_numpy()))
adata = AnnData(counts, obsm={"spatial": coordinates})
sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)

spatial_key = "spatial"
library_id = "tissue42"
adata.uns[spatial_key] = {library_id: {}}
adata.uns[spatial_key][library_id]["images"] = {}
adata.uns[spatial_key][library_id]["images"] = {"hires": img}
adata.uns[spatial_key][library_id]["scalefactors"] = {"tissue_hires_scalef": 1, "spot_diameter_fullres": 0.5,}

cluster = sc.read_h5ad(os.path.join('out/', vis_name_og))
sc.pp.normalize_total(cluster)
sc.pp.log1p(cluster)
cluster.obs['prefix'] = cluster.obs['files'].apply(lambda x: x.split('_')[-1].split('.')[0])
cluster_lin = sc.read_h5ad(os.path.join('out/', vis_name))
sc.pp.normalize_total(cluster_lin)
sc.pp.log1p(cluster_lin)
cluster_lin.obs['prefix'] = cluster_lin.obs['files'].apply(lambda x: x.split('_')[-1].split('.')[0])

for prt in proteins:
    adata.obs[prt] = cluster.X[:,np.argmax(cluster.var_names.values==prt)][cluster.obs['prefix']==name_tiff.split('.')[0]]
    adata.obs[prt+'lin'] = cluster_lin.X[:,np.argmax(cluster_lin.var_names.values==prt)][cluster_lin.obs['prefix']==name_tiff.split('.')[0]]
    adata.obs[prt+'diff'] = adata.obs[prt].values - adata.obs[prt+'lin'].values
    sq.pl.spatial_scatter(adata,
                        color=prt,
                        size=25,
                        img_channel=0,
                        img_alpha=0.,
                        crop_coord=crop_coord,
                        vmin=min(np.min(adata.obs[prt].values), np.min(adata.obs[prt+'lin'].values)),
                        vmax=max(np.max(adata.obs[prt].values), np.max(adata.obs[prt+'lin'].values)))
    plt.savefig(os.path.join(figure_dir, f'cell_expression_og_{prt}_{vis_name}_{name_tiff}.png'), bbox_inches='tight')
    plt.close()
    sq.pl.spatial_scatter(adata,
                        color=prt+'lin',
                        size=25,
                        img_channel=0,
                        img_alpha=0.,
                        crop_coord=crop_coord,
                        vmin=min(np.min(adata.obs[prt].values), np.min(adata.obs[prt+'lin'].values)),
                        vmax=max(np.max(adata.obs[prt].values), np.max(adata.obs[prt+'lin'].values)),
                        title=prt)
    plt.savefig(os.path.join(figure_dir, f'cell_expression_pred_{prt}_{vis_name}_{name_tiff}.png'), bbox_inches='tight')
    plt.close()
    sq.pl.spatial_scatter(adata,
                        color=prt+'diff',
                        size=25,
                        img_channel=0,
                        img_alpha=0.,
                        crop_coord=crop_coord,
                        title=prt + ' Actual - Prediction')
    plt.savefig(os.path.join(figure_dir, f'cell_expression_diff_{prt}_{vis_name}_{name_tiff}.png'), bbox_inches='tight')
    plt.close()
