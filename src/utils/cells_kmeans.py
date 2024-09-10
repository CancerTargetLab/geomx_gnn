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
figure_dir = 'figures/KMeans'
name = 'crc_optimal_num_clusters'

if not os.path.isdir(figure_dir):
    os.makedirs(figure_dir)

paths = [torch.load(os.path.join('data/raw', raw_dir, p)) for p in os.listdir(os.path.join('data/raw', raw_dir)) if p.endswith('_cells.pt')]
csv_path = [os.path.join('data/raw', raw_dir, p) for p in os.listdir(os.path.join('data/raw', raw_dir)) if p.endswith('.csv')][0]
cell_number = pd.read_csv(csv_path, header=0, sep=',').shape[0]
img_shape = torch.load(paths[0]).shape
x = torch.zeros((cell_number, img_shape[1]), dtype=torch.float32)

last_idx = 0
for path in paths:
    tmp = torch.load(tmp)
    x[last_idx:tmp.shape[0]+last_idx] = torch.mean(tmp, axis=(2, 3))
    del tmp
x = x.numpy()

sil_mean = []
sil_max = []
sil_min = []

with tqdm(range(2, max_clusters+1), total=max_clusters, desc='KMeans')as cluster_n:
    for k in cluster_n:
        sil = []
        for rep in range(n_reps):
            rand_index = np.random.randint(low=0, high=x.shape[0], size=num_samples)
            kmeans = KMeans(n_clusters=k, n_init=1).fit(x[rand_index])
            sil.append(silhouette_score(x[rand_index], kmeans.labels_, metric = 'euclidean'))
        sil = np.array(sil)
        sil_mean.append(np.mean(sil))
        sil_max.append(np.max(sil))
        sil_min.append(np.min(sil))


plt.plot(list(range(2, max_clusters+1)), sil_mean)
plt.fill_between(list(range(2, max_clusters+1)), sil_max, sil_min, alpha=0.3)
best_cluster = np.argmax(np.array(sil_mean)) + 2
plt.title(f'KMeans Clusters best with k={best_cluster}')
plt.ylabel('SIL score')
plt.xlabel('Num Clusters')
plt.savefig(os.path.join(figure_dir, f'{name}.png'))
plt.close()
