import pandas as pd
import numpy as np
import os

csv_dir = os.path.join(os.getcwd(), 'data/raw/CRC/')
csvs = [os.path.join(csv_dir,p) for p in os.listdir(csv_dir) if p.endswith('.csv')]

pos_df = pd.DataFrame()
label_df = pd.DataFrame()

for csv in csvs:
    df = pd.read_csv(csv)
    df['Image'] = np.array([csv.split('/')[-1].split('.')[0]+'.ome.tif']*df.shape[0])

    gene_names = np.concatenate((np.array(['Hoechst1']),df.columns[9:-10].values))
    genes_sum = np.sum(df[gene_names].values, axis=0)
    if pos_df.shape[0]>0:
        tmp_df = pd.DataFrame()
        tmp_df['Image'] = df['Image']
        tmp_df['Centroid.X.px'] = df['Xt'] / 0.65
        tmp_df['Centroid.Y.px'] = df['Yt'] / 0.65
        tmp_df['Class'] = ''
        tmp_df[gene_names] = df[gene_names].values
        pos_df = pd.concat([pos_df, tmp_df], ignore_index=True)

        tmp_df = pd.DataFrame()
        tmp_df['ROI'] = [csv.split('.')[0].split('/')[-1]]
        tmp_df['Patient_ID'] = [csv.split('.')[0][-2:]]
        tmp_df[gene_names] = [genes_sum]
        label_df = pd.concat([label_df, tmp_df], ignore_index=True)
    else:
        pos_df['Image'] = df['Image']
        pos_df['Centroid.X.px'] = df['Xt'] / 0.65
        pos_df['Centroid.Y.px'] = df['Yt'] / 0.65
        pos_df['Class'] = ''
        pos_df[gene_names] = df[gene_names].values

        label_df['ROI'] = [csv.split('.')[0].split('/')[-1]]
        label_df['Patient_ID'] = [csv.split('.')[0][-2:]]
        label_df[gene_names] = [genes_sum]

pos_df.to_csv('data/raw/CRC/CRC_measurements.csv', sep=',', header=True, index=False,)
label_df.to_csv('data/raw/CRC/CRC_label.csv', sep=',', header=True, index=False,)

def plot_gene_histo(gene):
    import matplotlib.pyplot as plt

    csv_dir = os.path.join(os.getcwd(), 'data/raw/CRC/')
    csvs = [os.path.join(csv_dir,p) for p in os.listdir(csv_dir) if p.endswith('.csv')]
    for csv in csvs:
        print(f'Plotting {csv} ...')
        df = pd.read_csv(csv)
        plt.hist(df[gene].values, bins='auto', color='blue',)
        plt.savefig(csv.split('/')[-1].split('.')[0]+'.png')
        plt.close()
        area = df['AREA'].values
        print('Area Max: ',np.max(area))
        print('Area Mean: ',np.mean(area))
        print('Area Median: ', np.median(area))
    print('DONE')
