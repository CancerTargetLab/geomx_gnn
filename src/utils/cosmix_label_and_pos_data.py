import pandas as pd
import numpy as np
import glob

tif_files = glob.glob('RawFiles/*/*/CellStatsDir/Morphology2D/*.TIF')

experiment_name_path_idx = 1

df = pd.read_csv('cosmix_measurements_data.csv')

tif_set = set(map((lambda x: x.split('/')[experiment_name_path_idx]), tif_files))
tif_dict = {}
for exp in tif_set:
    tif_dict[exp] = {}

for file in tif_files:
    tif_dict[file.split('/')[experiment_name_path_idx]][int(file.split('/')[-1].split('.')[0].split('F')[-1])] = file.split('/')[-1]

def get_file_name(exp, fov):
    if exp in tif_dict.keys():
        return tif_dict[exp][fov]
    else:
        return ''
     

df['Image'] = df.apply(lambda x: get_file_name(x['experiment'], x['fov']), axis=1)

measurements = pd.DataFrame()
measurements['Image'] = df['Image']
measurements['Centroid.X.px'] = df['Centroid.X.px']
measurements['Centroid.Y.px'] = df['Centroid.Y.px']
measurements['Class'] = ''

gene_names = df.columns.values[~df.columns.str.contains('^(Image|Centroid.X.px|Centroid.Y.px|fov|cell_ID|experiment)')].tolist()
measurements = pd.concat([measurements, df[gene_names]], axis=1)

measurements.to_csv('cosmix_measurements.csv', index=False, header=True, sep=',')
del measurements

label = pd.DataFrame()
for tif in df['Image'].unique().tolist():
    tmp = pd.DataFrame()
    tmp['ROI'] = [tif.split('.')[0]]
    tmp_df = df[df['Image']==tif]
    exp_name  = tmp_df['experiment'].unique().tolist()[0]
    if exp_name in tif_dict.keys():
        tmp['Patient_ID'] = exp_name
        tmp[gene_names] = np.sum(tmp_df[gene_names].values, axis=0)	#concat instead maybe
        if label.shape[0] > 0:
            label = pd.concat([label, tmp], ignore_index=True)
        else:
            label = tmp.copy(deep=True)
    else:
        print(f'{exp_name} in measurements not in tiff files!')

label.to_csv('cosmix_label.csv', index=False, header=True, sep=',')
