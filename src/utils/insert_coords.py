import numpy as np


def insert_coords(adata, df):
    adata.obs['Image'] = adata.obs['files'].apply(lambda x: x.split('graph_')[-1].split('.pt')[0]+'.tiff')

    adata.obs['x'] = np.nan
    adata.obs['y'] = np.nan

    for img in adata.obs['Image'].unique().tolist():
        adata.obs.loc[adata.obs['Image']==img, ['x', 'y']] = df.loc[df['Image'].isin([img]), ['Centroid.X.px', 'Centroid.Y.px']].values
    
    return adata