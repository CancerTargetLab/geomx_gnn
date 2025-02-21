import torch
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm

num_samples = 10000
max_clusters = 100
n_reps = 5
raw_dir = 'CRC'
split = 'train'
figure_dir = 'figures/KMeans'
name = 'crc_optimal_num_clusters'

if not os.path.isdir(figure_dir):
    os.makedirs(figure_dir)

paths = [os.path.join('data/raw', raw_dir, split, p) for p in os.listdir(os.path.join('data/raw', raw_dir, split)) if p.endswith('_cells.npy')]
csv_path = [os.path.join('data/raw', raw_dir, p) for p in os.listdir(os.path.join('data/raw', raw_dir)) if p.endswith('.csv')][0]
cell_number = pd.read_csv(csv_path, header=0, sep=',', usecols=['Centroid.X.px']).shape[0]
img_shape = np.load(paths[0]).shape
x = np.zeros((cell_number, img_shape[1]), dtype=np.float32)
_xcenter_p = int(img_shape[-2]*0.15)
_ycenter_p = int(img_shape[-1]*0.15)

last_idx = 0
with tqdm(paths, total=len(paths), desc='Load Channels Means...') as paths:
    for path in paths:
        tmp = np.load(path)
        if tmp.shape[-1] > 50:
            x[last_idx:tmp.shape[0]+last_idx] = np.max(tmp[:,:,int(tmp.shape[-2]/2)-_xcenter_p:int(tmp.shape[-2]/2)+_xcenter_p:,int(tmp.shape[-1]/2)-_ycenter_p:int(tmp.shape[-1]/2)+_ycenter_p],
                                                    axis=(-2,-1))
        else:
            x[last_idx:tmp.shape[0]+last_idx] = tmp[:,:,int(tmp.shape[-2]/2),int(tmp.shape[-1]/2)]
        last_idx += tmp.shape[0]
        del tmp

sil_mean = []
sil_max = []
sil_min = []

with tqdm(range(2, max_clusters+1), total=max_clusters, desc='KMeans')as cluster_n:
    for k in cluster_n:
        sil = []
        for rep in range(n_reps):
            rand_index = np.random.randint(low=0, high=x.shape[0], size=num_samples)
            kmeans = KMeans(n_clusters=k, n_init=1).fit(x[rand_index])
            centroids = kmeans.cluster_centers_
            pred_clusters = kmeans.labels_
            sil.append(np.sum((x[rand_index] - centroids[pred_clusters])**2)/num_samples)
            #sil.append(silhouette_score(x[rand_index], kmeans.labels_, metric = 'euclidean'))
        sil = np.array(sil)
        sil_mean.append(np.mean(sil))
        sil_max.append(np.max(sil))
        sil_min.append(np.min(sil))


plt.plot(list(range(2, max_clusters+1)), sil_mean)
plt.fill_between(list(range(2, max_clusters+1)), sil_max, sil_min, alpha=0.3)
plt.title(f'KMeans Clusters')
plt.ylabel('WSS score')
plt.xlabel('Num Clusters')
plt.savefig(os.path.join(figure_dir, f'{name}.png'))
plt.close()
