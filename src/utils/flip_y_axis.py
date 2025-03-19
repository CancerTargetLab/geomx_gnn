import pandas as pd
import skimage.io as io
import os

path = 'data/raw/nanostring/'

images_train = [img for img in os.listdir(os.path.join(path, 'train')) if img.upper().endswith(('.TIFF', '.TIF'))]
images_test = [img for img in os.listdir(os.path.join(path, 'test')) if img.upper().endswith(('.TIFF', '.TIF'))]

df = pd.read_csv(os.path.join(path, [csv for csv in os.listdir(path) if csv.endswith('.csv')][0]))

for imgf in images_train:
    print(imgf)
    img = io.imread(os.path.join(path, 'train', imgf))
    idx = df['Image'] == imgf
    df.loc[idx, 'Centroid.Y.px'] = img.shape[1] - df.loc[idx, 'Centroid.Y.px'].values - 1

for imgf in images_test:
    print(imgf)
    img = io.imread(os.path.join(path, 'test', imgf))
    idx = df['Image'] == imgf
    df.loc[idx, 'Centroid.Y.px'] = img.shape[1] - df.loc[idx, 'Centroid.Y.px'].values - 1

df.to_csv(os.path.join(path, [csv for csv in os.listdir(path) if csv.endswith('.csv')][0].split('.')[0]+'_flipped_y.csv'),
           index=False, header=True, sep=',')
print('DONE')
