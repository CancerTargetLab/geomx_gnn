import numpy as np
import os
from skimage import io
import torch
import torchvision.transforms as T
import pandas as pd
from tqdm import tqdm

#Implemented through adapting NaroNets Image processing.
#Original Implementation: https://github.com/djimenezsanchez/NaroNet/blob/main/src/NaroNet/Patch_Contrastive_Learning/preprocess_images.py#L145

def load_img(path, img_channels):
    if path.endswith(('.tiff', '.tif')):
        img = io.imread(path, plugin='tifffile')
        if not type(img_channels) == str:
            img = np.take(img, indices=img_channels, axis=2)
    else:
        file_end = path.split('/')[-1]
        print(f"Do not support opening of files of type {file_end}.")
        print(f"Could not open file {path}")
    
    return img

def calc_mean_std(image_paths, max_img=2**16, img_channels=''):
    global_hist = None
    for img_p in tqdm(image_paths, desc='Calculating mean and std for ROIs'):
        img = load_img(img_p, img_channels=img_channels)
        local_hist = [np.histogram(img[:,:,channel], bins=max_img+2, range=(0,max_img+2)) for channel in range(img.shape[2])]
        if global_hist:
            global_hist = [np.concatenate((global_hist[channel], local_hist[channel])) for channel in range(len(local_hist))]
        else:
            global_hist = local_hist
    
    mean = np.array([np.mean(hist_chan[0]*hist_chan[1][:hist_chan[1].shape[0]-1]) for hist_chan in global_hist], dtype=np.float32)
    std = np.array([np.std(hist_chan[0]*hist_chan[1][:hist_chan[1].shape[0]-1]) for hist_chan in global_hist], dtype=np.float32)
    return mean, std

def zscore(image_paths, mean, std):
    for img_p in tqdm(image_paths, desc='ZScore normalisation of ROIs'):
        img = torch.load(img_p.split('.')[0]+'_cells.pt')
        img = torch.permute(img, (0, 2, 3, 1))
        # following 4 lines copied and adapted from naronet preprocess_imagrs.py line 107-110 commit 7c419bc
        cells,x,y,chan = img.shape[0],img.shape[1],img.shape[2],img.shape[3]
        img = torch.reshape(img,(cells*x*y,chan))
        img = (img-mean)/(std+1e-16)
        img = torch.reshape(img,(cells,x,y,chan))
        img = torch.permute(img, (0, 3, 1, 2))

        torch.save(img.to(torch.float32), img_p.split('.')[0]+'_cells.pt')

def cell_seg(df_path, image_paths, img_channels=''):
    df = pd.read_csv(df_path, header=0, sep=',')
    df['Centroid.X.px'] = df['Centroid.X.px'].round().astype(int)
    df['Centroid.Y.px'] = df['Centroid.Y.px'].round().astype(int)
    for image in tqdm(image_paths, desc='Segmenting Cells'):
        img = torch.from_numpy(load_img(image, img_channels).astype(np.int16))
        df_img = df[df['Image']==image.split('/')[-1]]
        x = df_img['Centroid.X.px'].values
        y = df_img['Centroid.Y.px'].values
        all_cells = torch.Tensor().to(torch.int16)
        if x.shape[0] < 1:
            raise Exception(f'No coordinates in {df_path} for {image}!!!') 
        try:
            for cell in list(range(x.shape[0])):
                delta_x1 = 10 if x[cell] >= 10 else x[cell]
                delta_y1 = 10 if y[cell] >= 10 else y[cell]
                delta_x2 = 10 if img.shape[1]-x[cell] >= 10 else img.shape[1]-x[cell]
                delta_y2 = 10 if img.shape[0]-y[cell] >= 10 else img.shape[0]-y[cell]
                cell_img = img[y[cell]-delta_y1:y[cell]+delta_y2,x[cell]-delta_x1:x[cell]+delta_x2,:]
                cell_img = T.Resize((20, 20), antialias=None)(torch.moveaxis(cell_img, 2, 0))
                all_cells = torch.cat((all_cells, torch.unsqueeze(cell_img, axis=0)), axis=0)
        except Exception as e:
            print(f'Something went wrong when segmenting cells for {image}')
            print(e)
        torch.save(all_cells, os.path.join(image.split('.')[0]+'_cells.pt'))

def image_preprocess(path, max_img=2**16, img_channels='', path_mean_std=''):
    df_path = [os.path.join(path, p) for p in os.listdir(path) if p.endswith(('.csv'))][0]
    img_paths = [os.path.join(path, p) for p in os.listdir(path) if p.endswith(('.tiff', '.tif'))]

    img_channels = img_channels if len(img_channels) == 0 else np.array([int(channel) for channel in img_channels.split(',')])
    if len(path_mean_std) == 0:
        mean, std = calc_mean_std(img_paths, max_img=max_img, img_channels=img_channels)
        np.save(os.path.join(path, 'mean.npy'), mean)
        np.save(os.path.join(path, 'std.npy'), std)
    else: 
        mean = np.load(os.path.join(path_mean_std, 'mean.npy'))
        std = np.load(os.path.join(path_mean_std, 'std.npy'))
    cell_seg(df_path, img_paths, img_channels=img_channels)
    zscore(img_paths, torch.from_numpy(mean), torch.from_numpy(std))
