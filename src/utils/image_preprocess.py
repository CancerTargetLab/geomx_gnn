import numpy as np
import os
from skimage import io
import torch
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import torchvision.transforms as T
import pandas as pd
from tqdm import tqdm

#Implemented through adapting NaroNets Image processing.
#Original Implementation: https://github.com/djimenezsanchez/NaroNet/blob/main/src/NaroNet/Patch_Contrastive_Learning/preprocess_images.py#L145

def load_img(path,
             img_channels):
    if path.endswith(('.tiff', '.tif')):
        img = io.imread(path, plugin='tifffile')
        if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            img = np.transpose(img, (1,2,0))
        if not type(img_channels) == str:
            img = img[:, :, img_channels]
    else:
        file_end = path.split('/')[-1]
        print(f"Do not support opening of files of type {file_end}.")
        print(f"Could not open file {path}")
    
    return img

def calc_mean_std(image_paths,
                  max_img=2**16,
                  img_channels=''):
    global_hist = None
    
    for img_p in tqdm(image_paths, desc='Calculating mean and std for ROIs'):
        img = load_img(img_p, img_channels=img_channels)
        if global_hist is None:
            global_hist = np.zeros((img.shape[2], max_img+1))
        for channel in range(img.shape[2]):
            hist, _ = np.histogram(img[:,:,channel], bins=max_img+1, range=(0,max_img))
            global_hist[channel] += hist

    pixel_count = np.sum(global_hist[0])
    mean = np.sum(np.arange(max_img+1) * global_hist, axis=1) / pixel_count
    std_sq = np.sum(((np.arange(max_img+1) - mean[:, np.newaxis])**2) * global_hist, axis=1) / pixel_count
    std = np.sqrt(std_sq)
    
    return mean.astype(np.float32), std.astype(np.float32)

def zscore(image_paths,
           mean,
           std):
    for img_p in tqdm(image_paths, desc='ZScore normalisation of ROIs'):
        img = torch.load(img_p.split('.')[0]+'_cells.pt')
        img = torch.permute(img, (0, 2, 3, 1))
        # following 4 lines copied and adapted from naronet preprocess_imagrs.py line 107-110 commit 7c419bc
        cells,x,y,chan = img.shape[0],img.shape[1],img.shape[2],img.shape[3]
        img = torch.reshape(img,(cells*x*y,chan))
        img = (img-mean)/(std+1e-16)
        img = torch.reshape(img,(cells,x,y,chan))
        img = torch.permute(img, (0, 3, 1, 2))

        torch.save(img.to(torch.float16), img_p.split('.')[0]+'_cells.pt')

def process_cells(img,
                  x,
                  y,
                  cell_cutout,
                  all_results,
                  process_idx):
    for cell in list(range(x.shape[0])):
        delta_x1 = int(cell_cutout/2) if x[cell] >= int(cell_cutout/2) else x[cell]
        delta_y1 = int(cell_cutout/2) if y[cell] >= int(cell_cutout/2) else y[cell]
        delta_x2 = int(cell_cutout/2) if img.shape[1]-x[cell] >= int(cell_cutout/2) else img.shape[1]-x[cell]
        delta_y2 = int(cell_cutout/2) if img.shape[0]-y[cell] >= int(cell_cutout/2) else img.shape[0]-y[cell]
        cell_img = img[y[cell]-delta_y1:y[cell]+delta_y2,x[cell]-delta_x1:x[cell]+delta_x2,:]
        cell_img = T.Resize((cell_cutout, cell_cutout), antialias=None)(torch.moveaxis(cell_img, 2, 0))
        all_results[process_idx][cell] = cell_img

#https://github.com/pytorch/pytorch/issues/17199#issuecomment-833226969
# sometimes issues arise were processes deadlock when using fork instead of spawn, as more threads are used then available
def process_cells_wrapped(img,
                  x,
                  y,
                  cell_cutout,
                  all_results,
                  process_idx):
    return ThreadPoolExecutor().submit(process_cells,
                                       img=img,
                                       x=x,
                                       y=y,
                                       cell_cutout=cell_cutout,
                                       all_results=all_results,
                                       process_idx=process_idx).result()

def cell_seg(df_path,
             image_paths,
             img_channels='',
             cell_cutout=20,
             num_processes=1):
    df = pd.read_csv(df_path, header=0, sep=',')
    df['Centroid.X.px'] = df['Centroid.X.px'].round().astype(int)
    df['Centroid.Y.px'] = df['Centroid.Y.px'].round().astype(int)
    for image in tqdm(image_paths, desc='Segmenting Cells'):
        img = torch.from_numpy(load_img(image, img_channels).astype(np.int32)) # Needed to set to int32, as torch does not support uint16
        df_img = df[df['Image']==image.split('/')[-1]]
        x = df_img['Centroid.X.px'].values
        y = df_img['Centroid.Y.px'].values
        img.share_memory_()
        
        if x.shape[0] < 1:
            raise Exception(f'No coordinates in {df_path} for {image}!!!')
        
        # Split coordinates into chunks for parallel processing
        chunk_size = (len(x) + num_processes - 1) // num_processes
        chunks = [list(range(i, min(i + chunk_size, len(x)))) for i in range(0, len(x), chunk_size)]

        all_results = [torch.zeros((len(chunk),img.shape[-1],cell_cutout,cell_cutout)) for chunk in (chunks)]
        for result in all_results:
            result.share_memory_()
        all_cells = []
        num_processes = num_processes - 1 if num_processes >= 2 else 1
        processes = []
        
        for process_idx, chunk in enumerate(chunks):
            process = mp.Process(target=process_cells_wrapped, args=(img,
                                                             x[chunk],
                                                             y[chunk],
                                                             cell_cutout,
                                                             all_results,
                                                             process_idx))
            process.start()
            processes.append(process)
        
        for process in processes:
            process.join()
        
        for result in all_results:
            all_cells.extend(result)
        
        all_cells = torch.stack(all_cells)
        torch.save(all_cells, os.path.join(image.split('.')[0]+'_cells.pt'))
        del img
        del all_cells
        del df_img

def image_preprocess(path,
                     max_img=2**16,
                     img_channels='',
                     path_mean_std='',
                     cell_cutout=20,
                     num_processes=1):
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
    cell_seg(df_path, img_paths, img_channels=img_channels, cell_cutout=cell_cutout, num_processes=num_processes)
    zscore(img_paths, torch.from_numpy(mean), torch.from_numpy(std))
