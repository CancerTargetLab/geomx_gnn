import scanpy as sc
import pandas as df
import pandas as pd
import numpy as np

measurements = 'data/raw/cosmix_measurements_flipped_y.csv'
out = 'data/raw/cosmix_measurements_flipped_y_dca.csv'
df = pd.read_csv(measurements)
adata = sc.AnnData(df.loc[np.sum(df[df.columns[4:]].values, axis=-1) > 0, df.columns[4:]].values)
adata_var_names = df.columns[4:].values

sc.external.pp.dca(adata, ae_type='zinb-conddisp', verbose=True, return_info=True, learning_rate=0.0005, batch_size=256, hidden_size=(256, 128, 64, 128, 256))

df.loc[np.sum(df[df.columns[4:]].values, axis=-1) > 0, df.columns[4:].values] = adata.X.astype(np.float32)
df.to_csv(out, index=False, header=True, sep=',')
