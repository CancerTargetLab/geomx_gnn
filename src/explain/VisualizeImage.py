import os
import tifffile
from skimage import io
import numpy as np
import pandas as pd
import squidpy as sq
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt

def visualizeImage(raw_subset_dir, name_tiff, figure_dir, vis_name, args):
    """
    Visualize sc expression on Image, and more Image visualisations.

    raw_subset_dir (str): Dir name in data/raw/ containing images
    name_tiff (str): name of image to visualize
    figure_dir (str): Path to save figures to
    vis_name: name of .h5ad file to load scanpy.AnnData and add to figure save name
    args (dict): Arguments
    """
    path = os.path.join('data/raw', raw_subset_dir)
    df_path = [os.path.join(path, p) for p in os.listdir(path) if p.endswith(('.csv'))][0]
    df = pd.read_csv(df_path, header=0, sep=",")
    df = df[["Image", "Centroid.X.px", "Centroid.Y.px"]] #'Class'
    df = df[df["Image"] == name_tiff]
    df = df.drop("Image", axis=1)
    mask = ~df.duplicated(subset=['Centroid.X.px', 'Centroid.Y.px'], keep=False) | ~df.duplicated(subset=['Centroid.X.px', 'Centroid.Y.px'], keep='first')
    df = df[mask]

    img = np.expand_dims(io.imread(os.path.join('data/raw',
                                                raw_subset_dir,
                                                name_tiff),
                                                plugin='tifffile',
                                                key=0),
                                                axis=-1)
    crop_coord = [(args['vis_img_xcoords'][0], args['vis_img_ycoords'][0],
                   args['vis_img_xcoords'][1], args['vis_img_ycoords'][1])]
    if sum(crop_coord[0]) == 0:
        crop_coord = [(0, 0, img.shape[0], img.shape[1])]

    counts = np.random.default_rng(42).integers(0, 15, size=(df.shape[0], 1))
    coordinates = np.column_stack((df["Centroid.X.px"].to_numpy(), df["Centroid.Y.px"].to_numpy()))
    adata = AnnData(counts, obsm={"spatial": coordinates})
    sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)


    spatial_key = "spatial"
    library_id = "tissue42"
    adata.uns[spatial_key] = {library_id: {}}
    adata.uns[spatial_key][library_id]["images"] = {}
    adata.uns[spatial_key][library_id]["images"] = {"hires": img}
    adata.uns[spatial_key][library_id]["scalefactors"] = {"tissue_hires_scalef": 1,
                                                          "spot_diameter_fullres": 0.5,}

    cluster = sc.read_h5ad(os.path.join('out/', vis_name))
    sc.pp.normalize_total(cluster)
    sc.pp.log1p(cluster)
    cluster.obs['prefix'] = cluster.obs['files'].apply(lambda x: x.split('_')[-1].split('.')[0])
    adata.obs['cluster'] = cluster.obs['leiden'][cluster.obs['prefix']==name_tiff.split('.')[0]].apply(lambda x: str(x)).values

    if len(args['vis_name_og']) > 0:
        cluster_og = sc.read_h5ad(os.path.join('out/', args['vis_name_og']))
        sc.pp.normalize_total(cluster_og)
        sc.pp.log1p(cluster_og)
        cluster_og.obs['prefix'] = cluster.obs['files'].apply(lambda x: x.split('_')[-1].split('.')[0])

    if not os.path.exists(figure_dir) and not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)

    pre_name_tiff = name_tiff.split('.')[0]

    sq.pl.spatial_scatter(adata,
                            color="cluster",
                            size=25,
                            img_channel=0,
                            crop_coord=crop_coord)
    plt.savefig(os.path.join(figure_dir,
                             f'cluster_{vis_name}_{pre_name_tiff}.png'),
                             bbox_inches='tight')
    plt.close()

    if len(args['vis_protein']) > 0:
        prot_dir = os.path.join(figure_dir, pre_name_tiff+'_prediction')
        if not os.path.exists(prot_dir) and not os.path.isdir(prot_dir):
            os.makedirs(prot_dir)
        proteins = args['vis_protein'].replace('.', ' ').split(',')
        for prt in proteins:
            adata.obs[prt] = cluster.X[:,np.argmax(cluster.var_names.values==prt)][cluster.obs['prefix']==name_tiff.split('.')[0]]
            if len(args['vis_name_og']) > 0:
                adata.obs[prt+'lin'] = cluster_og.X[:,np.argmax(cluster_og.var_names.values==prt)][cluster_og.obs['prefix']==name_tiff.split('.')[0]]
                adata.obs[prt+'diff'] = adata.obs[prt].values - adata.obs[prt+'lin'].values
                sq.pl.spatial_scatter(adata,
                        color=prt,
                        size=25,
                        img_channel=0,
                        img_alpha=0.,
                        crop_coord=crop_coord,
                        vmin=min(np.min(adata.obs[prt].values), np.min(adata.obs[prt+'lin'].values)),
                        vmax=max(np.max(adata.obs[prt].values), np.max(adata.obs[prt+'lin'].values)))
                plt.savefig(os.path.join(prot_dir,
                                         f'cell_expression_og_{prt}_{vis_name}_{pre_name_tiff}.png'),
                                         bbox_inches='tight')
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
                plt.savefig(os.path.join(prot_dir,
                                         f'cell_expression_pred_{prt}_{vis_name}_{pre_name_tiff}.png'),
                                         bbox_inches='tight')
                plt.close()
                sq.pl.spatial_scatter(adata,
                                    color=prt+'diff',
                                    size=25,
                                    img_channel=0,
                                    img_alpha=0.,
                                    crop_coord=crop_coord,
                                    title=prt + ' Actual - Prediction')
                plt.savefig(os.path.join(prot_dir,
                                         f'cell_expression_diff_{prt}_{vis_name}_{pre_name_tiff}.png'),
                                         bbox_inches='tight')
                plt.close()
            else:
                sq.pl.spatial_scatter(adata,
                            color=prt,
                            size=25,
                            img_channel=0,
                            img_alpha=0.,
                            crop_coord=crop_coord)
                plt.savefig(os.path.join(prot_dir,
                                         f'cell_expression_pred_{prt}_{vis_name}_{pre_name_tiff}.png'),
                                         bbox_inches='tight')
                plt.close()

    sq.pl.spatial_scatter(adata,
                        color="cluster",
                        connectivity_key="spatial_connectivities",
                        edges_color="grey",
                        edges_width=1,
                        size=25,
                        img_channel=0,
                        crop_coord=crop_coord)
    plt.savefig(os.path.join(figure_dir,
                             f'cluster_graph_{vis_name}_{pre_name_tiff}.png'),
                             bbox_inches='tight')
    plt.close()

    if args['vis_all_channels']:
        channel_dir = os.path.join(figure_dir, pre_name_tiff+'_channels')
        if not os.path.exists(channel_dir) and not os.path.isdir(channel_dir):
            os.makedirs(channel_dir)
        n_channel = len(tifffile.TiffFile(os.path.join('data/raw',
                                                       raw_subset_dir,
                                                       name_tiff)).series[0].pages)
        for channel in range(n_channel):
            img = np.expand_dims(io.imread(os.path.join('data/raw',
                                                        raw_subset_dir,
                                                        name_tiff),
                                                        plugin='tifffile',
                                                        key=channel),
                                                        axis=-1)
            adata.uns[spatial_key] = {library_id: {}}
            adata.uns[spatial_key][library_id]["images"] = {}
            adata.uns[spatial_key][library_id]["images"] = {"hires": img}
            adata.uns[spatial_key][library_id]["scalefactors"] = {"tissue_hires_scalef": 1,
                                                                  "spot_diameter_fullres": 0.5,}
            sq.pl.spatial_scatter(adata,
                                img_channel=0,
                                crop_coord=crop_coord)
            plt.savefig(os.path.join(channel_dir,
                                     f'{vis_name}_channel_{channel}_{pre_name_tiff}.png'),
                                     bbox_inches='tight')
            plt.close()
