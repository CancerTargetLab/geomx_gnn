import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scanpy as sc
import pandas as pd
from src.utils.stats import per_gene_corr
import os
from tqdm import tqdm

def get_true_graph_expression_dict(path):
    path = os.path.join(os.getcwd(), path)
    graph_paths = [p for p in os.listdir(path) if 'graph' in p and p.endswith('pt')]
    value_dict = {}
    for graph_p in graph_paths:
        graph = torch.load(os.path.join(path, graph_p), map_location='cpu')
        value_dict[graph_p] = {'y': graph.y.numpy()}
        if 'Class' in graph.to_dict().keys():
            value_dict[graph_p]['cell_class'] =graph.Class
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
    num_cells = 0
    for roi_pred_p in cell_pred_paths:
        value_dict[roi_pred_p.split('cell_pred_')[1]]['cell_pred'] = torch.load(os.path.join(path, roi_pred_p), map_location='cpu').squeeze().detach().numpy()
        num_cells += value_dict[roi_pred_p.split('cell_pred_')[1]]['cell_pred'].shape[0]
    cell_shapes = (num_cells, value_dict[roi_pred_p.split('cell_pred_')[1]]['cell_pred'].shape[1])
    return value_dict, cell_shapes

def get_patient_ids(label_data, keys):
    df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'raw', label_data), header=0, sep=',')
    IDs = np.array(df[~df.duplicated(subset=['ROI'], keep=False) | ~df.duplicated(subset=['ROI'], keep='first')].sort_values(by=['ROI'])['Patient_ID'].values)
    exps = df.columns.values[2:]

    if len(keys) != IDs.shape[0] and keys[0] not in df['ROI'].values.tolist():
        keys.sort()
        tmp = np.ndarray((len(keys)), dtype=str)
        for i_key in range(len(keys)):
            tmp[i_key] = df[df['ROI']==keys[i_key].split('_')[-1].split('.')[0]]['Patient_ID'].values[0]
        IDs = tmp
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
    for id in np.unique(IDs).tolist():  #TODO: when subgraphs what then? how to automate instead of manual label creation?
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

def visualize_cell_expression(value_dict, IDs, exps, name, figure_dir, cell_shapes, select_cells=50000):
    if os.path.exists('out/'+name+'.h5ad'):
        adata = sc.read_h5ad('out/'+name+'.h5ad')
    else:
        rois = list(value_dict.keys())
        rois.sort()
        rois_np = np.array(rois)
        counts = np.zeros(cell_shapes, dtype=np.float32)
        cell_class = None
        ids = np.array([])
        files = np.array([])

        i = 0
        num_cells = 0
        key = 'cell_pred'
        for id in np.unique(IDs).tolist():
            id_map = IDs==id
            id_keys = rois_np[id_map].tolist()
            for id_key in id_keys:
                tmp_counts = value_dict[id_key][key]
                counts[num_cells:num_cells+tmp_counts.shape[0],:] = tmp_counts
                num_cells += tmp_counts.shape[0]
                if num_cells != 0:
                    if ('cell_class' in value_dict[id_key].keys()) and cell_class is not None:
                        cell_class = np.concatenate((cell_class, value_dict[id_key]['cell_class']))
                else:
                    if ('cell_class' in value_dict[id_key].keys()):
                        cell_class = value_dict[id_key]['cell_class']
                ids = np.concatenate((ids, np.array([id]*value_dict[id_key][key].shape[0])))
                files = np.concatenate((files, np.array([id_key]*value_dict[id_key][key].shape[0])))
                i += 1
        counts = np.array(counts)

        cell_index = np.arange(counts.shape[0])
        if select_cells and counts.shape[0] > select_cells:
            cell_index = np.random.default_rng(42).choice(np.arange(counts.shape[0]), size=select_cells, replace=False)
        
        adata = sc.AnnData(counts)
        if cell_class is not None:
            cell_class = np.array(cell_class)
            adata.obs['cell_class'] = cell_class
        adata.obs['ID'] = ids
        adata.obs['files'] = files
        adata.var_names = exps
        adata.write('out/'+name+'_all.h5ad')

        adata = sc.AnnData(counts[cell_index])
        if cell_class is not None:
            cell_class = np.array(cell_class)
            adata.obs['cell_class'] = cell_class[cell_index]
        adata.obs['ID'] = ids[cell_index]
        adata.obs['files'] = files[cell_index]
        adata.var_names = exps
        
        adata.layers['counts'] = adata.X.copy()
        sc.pp.log1p(adata)
        adata.layers['logs'] = adata.X.copy()
        adata.X = adata.layers['counts'].copy()
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=10, min_disp=0.5)
        
        sc.pp.scale(adata)
        sc.tl.pca(adata, svd_solver='arpack', n_comps=adata.X.shape[1]-1, chunked=True, chunk_size=50000, use_highly_variable=False)
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=adata.varm['PCs'].shape[1])
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=0.5)

        sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon', show=False)

        adata.write('out/'+name+'.h5ad')
    
    #with plt.rc_context():
    sc.pl.highly_variable_genes(adata, show=False)
    plt.savefig(os.path.join(figure_dir, f'highly_varible_genes{name}.png'))
    plt.close()

    categories = np.unique(adata.obs['ID'])
    colors = np.linspace(0, 1, len(categories))
    colordict = dict(zip(categories, colors))
    adata.obs['Color'] = adata.obs['ID'].apply(lambda x: colordict[x])
    plt.scatter(adata.obsm['X_umap'][:,0], adata.obsm['X_umap'][:,1], c=adata.obs['Color'], alpha=0.4, cmap='gist_ncar', s=1)
    plt.savefig(os.path.join(figure_dir, f'umap{name}_ID.png'))
    plt.close()

    sc.pl.umap(adata, color='leiden', show=False)
    plt.savefig(os.path.join(figure_dir, f'umap{name}_cluster.png'))
    plt.close()
    sc.pl.umap(adata, color='leiden',
               show=False, add_outline=True, legend_loc='on data',
               legend_fontsize=12, legend_fontoutline=2,frameon=False)
    plt.savefig(os.path.join(figure_dir, f'umap{name}_cluster_named.png'))
    plt.close()

    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, show=False)
    plt.savefig(os.path.join(figure_dir, f'rank_genes_group{name}.png'))
    plt.close()
    sc.pl.rank_genes_groups_heatmap(adata, show_gene_labels=True, show=False, layer='logs', n_genes=5)
    plt.savefig(os.path.join(figure_dir, f'rank_genes_group{name}_heatmap.png'))
    plt.close()

    sc.pl.heatmap(adata, adata.var_names, groupby='leiden', show=False, layer='logs')
    plt.savefig(os.path.join(figure_dir, f'heatmap{name}.png'))
    plt.close()
    sc.pl.violin(adata, adata.var['highly_variable'].index[adata.var['highly_variable'].values].values, groupby='leiden', show=False, layer='logs')
    plt.savefig(os.path.join(figure_dir, f'violin_highly_varible{name}.png'))
    plt.close()

    if 'cell_class' in adata.obs.columns.values.tolist():
        confusion_matrix = np.zeros((len(np.unique(adata.obs['leiden'])), len(np.unique(adata.obs['cell_class']))))

        # Fill the matrix based on the relationships between categories
        for i, category_a in enumerate(np.unique(adata.obs['leiden'])):
            for j, category_b in enumerate(np.unique(adata.obs['cell_class'])):
                count = np.sum((adata.obs['leiden'] == category_a) & (adata.obs['cell_class'] == category_b))
                confusion_matrix[i, j] = count

        # Create a heatmap using seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues',
                    xticklabels=np.unique(adata.obs['cell_class']), yticklabels=np.unique(adata.obs['leiden']))
        plt.xlabel('Categories')
        plt.ylabel('Leiden Clusters')
        plt.title('Relationship Between predicted Cell Clusters and Categories')
        plt.savefig(os.path.join(figure_dir, f'Cell_Class_label_heatmap{name}.png'))

def visualize_graph_accuracy(value_dict, IDs, exps, name, figure_dir):
    adata_y = get_bulk_expression_of(value_dict, IDs, exps, key='y')
    adata_p = get_bulk_expression_of(value_dict, IDs, exps, key='roi_pred')

    similarity = torch.nn.CosineSimilarity()

    adata_p.obs['cs'] = similarity(torch.from_numpy(adata_p.X), torch.from_numpy(adata_y.X)).squeeze().detach().numpy()

    outlier_map = adata_p.obs['cs'] < 0.78
    outlier_cs = adata_p.obs['cs'][outlier_map]
    outlier_f = adata_p.obs['files'][outlier_map]
    for i in range(outlier_cs.shape[0]):
        print(f'ROI: {outlier_f[i]}; Cosine Similarity: {outlier_cs[i]:.4f}')
    
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
    plt.savefig(os.path.join(figure_dir, f'all_boxplot{name}.png'))
    plt.close()

    plt.figure(figsize=(30, 5))
    plt.scatter(adata_p.obs['ID'].apply(lambda x: str(x)).values, adata_p.obs['cs'], s=10)
    plt.title('Cosine Similarity of IDs')
    plt.ylabel('Cosine Similarity')
    plt.xticks(rotation=90)  # Rotate x-axis labels vertically
    plt.xlabel('IDs')
    plt.savefig(os.path.join(figure_dir, f'cosine_similarity_IDs{name}.png'))
    plt.close()

    df = pd.DataFrame()
    df['cs'] = adata_p.obs['cs'].values
    df['slides'] = adata_p.obs['files'].apply(lambda x: x.split('-')[-1]).values
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df, y='cs', x='slides')
    plt.title('Cosine Similarity of Slides')
    plt.ylabel('Cosine Similarity')
    plt.xlabel('Slides')
    plt.xticks(rotation=90)  # Rotate x-axis labels vertically
    plt.savefig(os.path.join(figure_dir, f'cosine_similarity_slides{name}.png'))
    plt.close()

def visualize_per_gene_corr(value_dict, IDs, exps, name, figure_dir):
    adata_y = get_bulk_expression_of(value_dict, IDs, exps, key='y')
    adata_p = get_bulk_expression_of(value_dict, IDs, exps, key='roi_pred')

    pred = adata_p.X
    y = adata_y.X

    p_statistic, p_pval = per_gene_corr(pred, y, mean=False, method='pearsonr')
    s_statistic, s_pval = per_gene_corr(pred, y, mean=False, method='pearsonr')
    k_statistic, k_pval = per_gene_corr(pred, y, mean=False, method='pearsonr')
        
    correlation_data = {
        'Variable': adata_p.var_names.values,
        'Pearson Correlation Coef.': [corr for corr in p_statistic],
        'Pearson p-value': [corr for corr in p_pval],
        'Spearman Correlation Coef.': [corr for corr in s_statistic],
        'Spearman p-value': [corr for corr in s_pval],
        'Kendall Correlation Coef.': [corr for corr in k_statistic],
        'Kendall p-value': [corr for corr in k_pval]
    }

    corr_df = pd.DataFrame(correlation_data)

    plt.figure(figsize=(10, 5))
    plt.table(cellText=corr_df.values, colLabels=corr_df.columns, loc='center')
    plt.axis('off')
    plt.savefig(os.path.join(figure_dir, f'corr_area{name}.pdf'))
    plt.close()

def visualizeExpression(processed_dir='TMA1_processed',
                        embed_dir='out/',
                        label_data='label_data.csv',
                        figure_dir='figures/',
                        name='_cells',
                        select_cells=50000):
    value_dict = get_true_graph_expression_dict(os.path.join('data/processed', processed_dir))
    value_dict = get_predicted_graph_expression(value_dict, embed_dir)
    value_dict, cell_shapes = get_predicted_cell_expression(value_dict, embed_dir)
    IDs, exps = get_patient_ids(label_data, list(value_dict.keys()))
    if not os.path.exists(figure_dir) and not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)
    # visualize_bulk_expression(value_dict, IDs, exps, '_true', key='y')
    # visualize_bulk_expression(value_dict, IDs, exps, '_pred', key='roi_pred')
    visualize_cell_expression(value_dict, IDs, exps, name, figure_dir, cell_shapes, select_cells)
    visualize_graph_accuracy(value_dict, IDs, exps, name, figure_dir)
    visualize_per_gene_corr(value_dict, IDs, exps, name, figure_dir)
