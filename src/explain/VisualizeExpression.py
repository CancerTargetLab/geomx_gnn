import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scanpy as sc
import pandas as pd
import os
from tqdm import tqdm

def get_true_graph_expression_dict(path):
    path = os.path.join(os.getcwd(), path)
    graph_paths = [p for p in os.listdir(path) if p.startswith('graph')]
    value_dict = {}
    for graph_p in graph_paths:
        value_dict[graph_p] = {'y': torch.load(os.path.join(path, graph_p), map_location='cpu').y.numpy()}
    return value_dict

def get_predicted_graph_expression(value_dict, path):
    path = os.path.join(os.getcwd(), path)
    roi_pred_paths = [p for p in os.listdir(path) if p.startswith('roi_pred')]
    for roi_pred_p in roi_pred_paths:
        value_dict[roi_pred_p.split('roi_pred_')[1]]['roi_pred'] = torch.load(os.path.join(path, roi_pred_p), map_location='cpu').squeeze().detach().numpy()
    return value_dict

def get_predicted_cell_expression(value_dict, path):
    path = os.path.join(os.getcwd(), path)
    cell_pred_paths = [p for p in os.listdir(path) if p.startswith('cell_pred')]
    for roi_pred_p in cell_pred_paths:
        value_dict[roi_pred_p.split('cell_pred_')[1]]['cell_pred'] = torch.load(os.path.join(path, roi_pred_p), map_location='cpu').squeeze().detach().numpy()
    return value_dict

def get_patient_ids(label_data):
    df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'raw', label_data), header=0, sep=',')
    IDs = np.array(df[~df.duplicated(subset=['ROI'], keep=False) | ~df.duplicated(subset=['ROI'], keep='first')].sort_values(by=['ROI'])['Patient_ID'].values)
    exps = df.columns.values[3:] #TODO: 2: for p2106
    return IDs, exps

def get_bulk_expression_of(value_dict, IDs, exps, key='y'):
    rois = list(value_dict.keys())
    rois.sort()
    rois_np = np.array(rois)
    adata = sc.AnnData(np.zeros((len(rois), value_dict[rois[0]][key].shape[0])))
    adata.obs['ID'] = -1
    adata.var_names = exps
    files = np.array([])

    i = 0
    for id in np.unique(IDs).tolist():
        id_map = IDs==id
        id_keys = rois_np[id_map].tolist()
        for id_key in id_keys:
            adata.X[i] = value_dict[id_key][key]
            adata.obs['ID'][i] = id
            files = np.concatenate((files, np.array([id_key])))
            i += 1
    adata.obs['files'] = files
    return adata

def visualize_bulk_expression(value_dict, IDs, exps, name, key='y'):
    adata = get_bulk_expression_of(value_dict, IDs, exps, key)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pl.highly_variable_genes(adata, save=name+'.png', show=False)
    sc.pp.scale(adata)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=adata.varm['PCs'].shape[1])
    sc.tl.umap(adata)
    sc.tl.leiden(adata)
    sc.pl.umap(adata, color=['ID', 'leiden'], save=name+'.png', show=False)
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon', show=False)
    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save=name+'.png', show=False)

def visualize_cell_expression(value_dict, IDs, exps, name):
    if os.path.exists('out/'+name+'.h5ad'):
        adata = sc.read_h5ad('out/'+name+'.h5ad')
    else:
        rois = list(value_dict.keys())
        rois.sort()
        rois_np = np.array(rois)
        counts = None
        ids = np.array([])
        files = np.array([])

        i = 0
        key = 'cell_pred'
        for id in np.unique(IDs).tolist():
            id_map = IDs==id
            id_keys = rois_np[id_map].tolist()
            for id_key in id_keys:
                if counts is not None:
                    counts = np.concatenate((counts, value_dict[id_key][key]))
                else:
                    counts = value_dict[id_key][key]
                ids = np.concatenate((ids, np.array([id]*value_dict[id_key][key].shape[0])))
                files = np.concatenate((files, np.array([id_key]*value_dict[id_key][key].shape[0])))
                i += 1
        counts = np.array(counts)
        adata = sc.AnnData(counts)
        adata.obs['ID'] = ids
        adata.obs['files'] = files
        adata.var_names = exps
        adata.layers['counts'] = adata.X.copy() 
        sc.pp.log1p(adata)
        adata.layers['logs'] = adata.X.copy()
        adata.X = adata.layers['counts'].copy() 
        sc.pp.normalize_total(adata, exclude_highly_expressed=True, max_fraction=0.15)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        
        sc.pp.scale(adata)
        sc.tl.pca(adata, svd_solver='arpack')
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=adata.varm['PCs'].shape[1])
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=0.5)

        sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon', show=False, layer='logs')

        adata.write('out/'+name+'.h5ad')
    
    #with plt.rc_context():
    sc.pl.highly_variable_genes(adata, show=False, save=name+'.png',)
    #plt.save('figures/')
    plt.close()

    categories = np.unique(adata.obs['ID'])
    colors = np.linspace(0, 1, len(categories))
    colordict = dict(zip(categories, colors))
    adata.obs['Color'] = adata.obs['ID'].apply(lambda x: colordict[x])
    plt.scatter(adata.obsm['X_umap'][:,0], adata.obsm['X_umap'][:,1], c=adata.obs['Color'], alpha=0.4, cmap='gist_ncar', s=1)
    plt.savefig('figures/umap'+name+'_ID.png')

    sc.pl.umap(adata, color='leiden', save=name+'_cluster.png', show=False)
    sc.pl.umap(adata, color='leiden', save=name+'_cluster_named.png',
               show=False, add_outline=True, legend_loc='on data',
               legend_fontsize=12, legend_fontoutline=2,frameon=False)

    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save=name+'.png', show=False)
    sc.pl.rank_genes_groups_heatmap(adata, show_gene_labels=True, show=False, save=name+'_rank_genes.png', layer='logs', n_genes=5)

    sc.pl.heatmap(adata, adata.var_names, groupby='leiden', show=False, save=name+'.png', layer='logs')
    sc.pl.violin(adata, adata.var['highly_variable'].index[adata.var['highly_variable'].values].values, groupby='leiden', show=False, save=name+'.png', layer='logs')
    plt.close()


def visualize_graph_accuracy(value_dict, IDs, exps, name):
    adata_y = get_bulk_expression_of(value_dict, IDs, exps, key='y')
    adata_p = get_bulk_expression_of(value_dict, IDs, exps, key='roi_pred')

    similarity = torch.nn.CosineSimilarity()

    adata_p.obs['cs'] = similarity(torch.from_numpy(adata_p.X), torch.from_numpy(adata_y.X)).squeeze().detach().numpy()
    
    plt.close('all')
    boxplot = plt.boxplot(adata_p.obs['cs'],)# labels=[category])
    outliers = [flier.get_ydata() for flier in boxplot['fliers']]

    for j, outlier_y in enumerate(outliers):
        outlier_x = np.full_like(outlier_y, 1.1)
        #plt.scatter(outlier_x, outlier_y, marker='o', color='red', label='Outliers' if j == 0 else '')

        for x, y, info in zip(outlier_x, outlier_y, adata_p.obs['files']):
            plt.annotate(info, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='red')


    plt.ylabel('Cosine Similarity')
    plt.title('Boxplots of Cosine Similarity')
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'figures/all_boxplot{name}.png')
    plt.close()

    plt.figure(figsize=(30, 5))
    plt.scatter(adata_p.obs['ID'].apply(lambda x: str(x)).values, adata_p.obs['cs'], s=10)
    plt.title('Cosine Similarity of IDs')
    plt.ylabel('Cosine Similarity')
    plt.xticks(rotation=90)  # Rotate x-axis labels vertically
    plt.xlabel('IDs')
    plt.savefig(f'figures/cosine_similarity_IDs{name}.png')
    plt.close()

    df = pd.DataFrame()
    df['cs'] = adata_p.obs['cs'].values
    df['slides'] = adata_p.obs['files'].apply(lambda x: x.split('-')[-1]).values
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, y='cs', x='slides')
    plt.title('Cosine Similarity of Slides')
    plt.ylabel('Cosine Similarity')
    plt.xlabel('Slides')
    plt.xticks(rotation=90)  # Rotate x-axis labels vertically
    plt.savefig(f'figures/cosine_similarity_slides{name}.png')
    plt.close()






    



value_dict = get_true_graph_expression_dict('data/processed/TMA1_preprocessed')
value_dict = get_predicted_graph_expression(value_dict, 'out/TMA1')
value_dict = get_predicted_cell_expression(value_dict, 'out/TMA1')
IDs, exps = get_patient_ids('OC1_all.csv')
# visualize_bulk_expression(value_dict, IDs, exps, '_true', key='y')
# visualize_bulk_expression(value_dict, IDs, exps, '_pred', key='roi_pred')
visualize_cell_expression(value_dict, IDs, exps, '_cells')
visualize_graph_accuracy(value_dict, IDs, exps, '_cells')

