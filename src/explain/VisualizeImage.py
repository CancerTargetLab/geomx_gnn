import os
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

    img = io.imread(os.path.join('data/raw', raw_subset_dir, name_tiff), plugin='tifffile')
    if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
        img = np.transpose(img, (1,2,0))
    crop_coord = [(args['vis_img_xcoords'][0], args['vis_img_ycoords'][0],
                   args['vis_img_xcoords'][1], args['vis_img_ycoords'][1])]
    if sum(crop_coord[0]) == 0:
        crop_coord = [(0, 0, img.shape[0], img.shape[1])]

    counts = np.random.default_rng(42).integers(0, 15, size=(df.shape[0], 1))

    coordinates = np.column_stack((df["Centroid.X.px"].to_numpy(), df["Centroid.Y.px"].to_numpy()))

    adata = AnnData(counts, obsm={"spatial": coordinates})

    # sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
    # edge_matrix = adata.obsp["spatial_distances"]
    # edge_matrix[edge_matrix > 60] = 0.
    # adata.obsp["spatial_distances"] = edge_matrix
    sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)


    spatial_key = "spatial"
    library_id = "tissue42"
    adata.uns[spatial_key] = {library_id: {}}
    adata.uns[spatial_key][library_id]["images"] = {}
    adata.uns[spatial_key][library_id]["images"] = {"hires": img}
    adata.uns[spatial_key][library_id]["scalefactors"] = {"tissue_hires_scalef": 1, "spot_diameter_fullres": 0.5,}

    cluster = sc.read_h5ad(os.path.join('out/', vis_name))
    sc.pp.normalize_total(cluster)
    sc.pp.log1p(cluster)
    cluster.obs['prefix'] = cluster.obs['files'].apply(lambda x: x.split('_')[-1].split('.')[0])
    adata.obs['cluster'] = cluster.obs['leiden'][cluster.obs['prefix']==name_tiff.split('.')[0]].values

    if not os.path.exists(figure_dir) and not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)

    sq.pl.spatial_scatter(adata,
                            color="cluster",
                            size=25,
                            img_channel=args['vis_channel'],
                            crop_coord=crop_coord)
    plt.savefig(os.path.join(figure_dir, f'cluster_{vis_name}_{name_tiff}.png'), bbox_inches='tight')
    plt.close()

    if len(args['vis_protein']) > 0:
        proteins = args['vis_protein'].replace('.', ' ').split(',')
        for prt in proteins:
            adata.obs[prt] = cluster.X[:,np.argmax(cluster.var_names.values==prt)][cluster.obs['prefix']==name_tiff.split('.')[0]]
        sq.pl.spatial_scatter(adata,
                            color=proteins,
                            size=25,
                            img_channel=args['vis_channel'],
                            img_alpha=0.,
                            crop_coord=crop_coord)
        plt.savefig(os.path.join(figure_dir, f'cell_expression_pred_{vis_name}_{name_tiff}.png'), bbox_inches='tight')
        plt.close()

    sq.pl.spatial_scatter(adata,
                            color="cluster",
                            connectivity_key="spatial_connectivities",
                            edges_color="grey",
                            edges_width=1,
                            size=25,
                            img_channel=args['vis_channel'],
                            crop_coord=crop_coord)
    plt.savefig(os.path.join(figure_dir, f'cluster_graph_{vis_name}_{name_tiff}.png'), bbox_inches='tight')
    plt.close()

    if args['vis_all_channels']:
        for channel in range(img.shape[2]):
            sq.pl.spatial_scatter(adata,
                                img_channel=channel,
                                crop_coord=crop_coord)
            plt.savefig(os.path.join(figure_dir, f'{vis_name}_channel_{channel}_{name_tiff}.png'), bbox_inches='tight')
            plt.close()
