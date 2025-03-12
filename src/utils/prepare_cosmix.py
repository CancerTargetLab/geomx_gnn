import glob
import pandas as pd
import numpy as np

tif_files = glob.glob('RawFiles/*/*/CellStatsDir/Morphology2D/*.TIF')
exprMat_files = glob.glob('flatFiles/*/*exprMat_file*')
metaData_files = glob.glob('flatFiles/*/*metadata_file*')

experiment_name_path_idx = 1
ignore_cell_id = [0]

num_cells = 0
gene_names = []
fov = []
cell_ID = []
experiment = []


for exprMat in exprMat_files:
    fov_id_df = pd.read_csv(exprMat, usecols=['fov', 'cell_ID'])
    if ignore_cell_id:
        fov_id_df = fov_id_df[~fov_id_df['cell_ID'].isin(ignore_cell_id)]
    num_cells += fov_id_df.shape[0]
    fov_id_df = fov_id_df.sort_values(by=['fov', 'cell_ID'], kind='stable', ignore_index=True)
    fov.extend(fov_id_df['fov'].values.tolist())
    cell_ID.extend(fov_id_df['cell_ID'].values.tolist())
    experiment.extend([exprMat.split('/')[experiment_name_path_idx]]*fov_id_df.shape[0])
    col_names = pd.read_csv(exprMat, index_col=0, nrows=0)
    col_names = col_names[col_names.columns.values[~col_names.columns.str.contains('^(SystemControl|cell_ID|Negative|fov|NegPrb)')]]
    if len(gene_names) == 0:
        gene_names = col_names.columns.tolist()
    else:
        if not set(gene_names) == set(col_names.columns.tolist()):
            raise Exception(f'Some Gene names in {exprMat} do not exists in other experiments runs! Fix pls')

cell_expr = np.zeros((num_cells, len(gene_names)), dtype=np.uint16)

start = 0
gene_names_cell_id = gene_names.copy()
gene_names_cell_id.append('cell_ID')
for exprMat in exprMat_files:
    expr = pd.read_csv(exprMat, usecols=gene_names_cell_id)
    if ignore_cell_id:
        expr = expr[~expr['cell_ID'].isin(ignore_cell_id)]
    cell_expr[start:start+expr.shape[0],:] = expr[gene_names].values
    start += expr.shape[0]

x = np.zeros(len(experiment))
y = np.zeros(len(experiment))
cell_ID = np.array(cell_ID)
fov = np.array(fov)
experiment = np.array(experiment)

for metaData in metaData_files:
    exp = metaData.split('/')[experiment_name_path_idx]
    idx = experiment == exp
    df = pd.read_csv(metaData, usecols=['fov', 'cell_ID', 'CenterX_local_px', 'CenterY_local_px'])
    if ignore_cell_id:
        df = df[~df['cell_ID'].isin(ignore_cell_id)]
    df = df.sort_values(by=['fov', 'cell_ID'], kind='stable', ignore_index=True)
    x[idx] = df['CenterX_local_px'].values
    y[idx] = df['CenterY_local_px'].values
    if not (df['fov'].values == fov[idx]).all() or not (df['cell_ID'].values == cell_ID[idx]).all():
        raise Exception('fov or cell_ID did not match when getting coords to fov or cell_ID of expr pattern')

df = pd.DataFrame()
df['experiment'] = experiment
df['fov'] = fov
df['cell_ID'] = cell_ID
df['Centroid.X.px'] = x
df['Centroid.Y.px'] = y
df = pd.concat([df, pd.DataFrame(data=cell_expr, columns=gene_names)], axis=1)
df.to_csv('cosmix_measurements_data.csv', index=False, header=True, sep=',')
