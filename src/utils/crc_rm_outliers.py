import pandas as pd
import numpy as np
import os

m = 'CRC_measurements.csv'
dir = 'data/raw/CRC'
upper_p = 99
lower_p = 1
out_dir = 'data/raw'
out_name = 'CRC_1p_measurements.csv'

df = pd.read_csv(os.path.join(dir, m))

cell_sum = np.sum(df[df.columns.values[4:]].values, axis=1)
upper_p_num = np.percentile(cell_sum, upper_p)
lower_p_num = np.percentile(cell_sum, lower_p)
cell_select = (cell_sum < upper_p_num) & (cell_sum > lower_p_num)

df = df[cell_select]

strings_to_match = ['A488', 'A555', 'A647']
cols_to_remove = [col for col in df.columns if any(substring in col for substring in strings_to_match)]
df = df.drop(columns=cols_to_remove)

df.to_csv(os.path.join(out_dir, out_name), sep=',', index=False, header=True)
