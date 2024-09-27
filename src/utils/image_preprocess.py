import numpy as np
import os
from skimage import io
import torch
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from tqdm import tqdm

#Implemented through adapting NaroNets Image processing.
#Original Implementation: https://github.com/djimenezsanchez/NaroNet/blob/main/src/NaroNet/Patch_Contrastive_Learning/preprocess_images.py#L145

def load_img(path,
             img_channels):
    """
    Load tifffile images.

    Parameters:
    path (str): Path of image
    img_channels (list): List containing indices of Channels to extract

    Returns:
    np.array, Image
    """

    if path.endswith(('.tiff', '.tif')):
        img = io.imread(path, plugin='tifffile')
        if img.shape[2] < img.shape[1] and img.shape[2] < img.shape[0]:
            img = np.transpose(img, (2,0,1))
        if not type(img_channels) == str:
            img = img[img_channels,:,:]
    else:
        file_end = path.split('/')[-1]
        print(f"Do not support opening of files of type {file_end}.")
        print(f"Could not open file {path}")
    
    return img

def calc_mean_std(image_paths,
                  max_img=2**16):
    """
    Calculate mean and std per channel over all images.

    Paramters:
    image_paths (list): List containing str paths to tifffile images
    max_img (int): Max possible pixel intensitiy of images

    Returns:
    (np.array, np.array), tuple containing 1D arrays of mean,std values for each channel extracted
    """

    global_hist = None
    
    # Sum channel-wise histograms of pixel values over all images,TODO: do stuff in torch?
    for img_p in tqdm(image_paths, desc='Calculating mean and std for ROIs'):
        img = torch.from_numpy(np.load(img_p.split('.')[0]+'_cells.npy'))   #TODO: change when torch support save/indexing for uint16
        if global_hist is None:
            global_hist = np.zeros((img.shape[1], max_img+1))
        for channel in range(img.shape[1]):
            hist, _ = np.histogram(img[:,channel,:,:], bins=max_img+1, range=(0,max_img))
            global_hist[channel] += hist

    pixel_count = np.sum(global_hist[0])
    mean = np.sum(np.arange(max_img+1) * global_hist, axis=1) / pixel_count
    std_sq = np.sum(((np.arange(max_img+1) - mean[:, np.newaxis])**2) * global_hist, axis=1) / pixel_count
    std = np.sqrt(std_sq)
    
    return mean.astype(np.float32), std.astype(np.float32)

def process_cells(img,
                  x,
                  y,
                  chunk,
                  cell_cutout,
                  all_results):
    """
    Cut out cells of image.

    Parameters:
    img (torch.tensor): image
    x (list): List of floats or ints containing pixel x cell positions
    y (list): List of floats or ints containing pixel y cell positions
    chunk (list): List of ints containing index position of cell cutouts
    cell_cutout (int): length/width of cut out
    all_results (list): List containing torch.tensor of shape (num_cells, cell_cutout,cell_cutout,channels)
    """
    l_w = int(cell_cutout/2)
    for cell in list(range(x.shape[0])):
        delta_x1 = l_w if x[cell] >= l_w else x[cell]
        delta_y1 = l_w if y[cell] >= l_w else y[cell]
        delta_x2 = l_w if img.shape[2]-x[cell] >= l_w else img.shape[2]-x[cell]
        delta_y2 = l_w if img.shape[1]-y[cell] >= l_w else img.shape[1]-y[cell]
        all_results[chunk[cell],:,l_w-delta_y1:l_w+delta_y2,l_w-delta_x1:l_w+delta_x2] = img[:,y[cell]-delta_y1:y[cell]+delta_y2,x[cell]-delta_x1:x[cell]+delta_x2]

#https://github.com/pytorch/pytorch/issues/17199#issuecomment-833226969
# sometimes issues arise were processes deadlock when using fork instead of spawn, as more threads are used then available
def process_cells_wrapped(img,
                  x,
                  y,
                  chunk,
                  cell_cutout,
                  all_results):
    """
    Wrapper for process_cells.

    Parameters:
    img (torch.tensor): image
    x (list): List of floats or ints containing pixel x cell positions
    y (list): List of floats or ints containing pixel y cell positions
    chunk (list): List of ints containing index position of cell cutouts
    cell_cutout (int): length/width of cut out
    all_results (list): List containing torch.tensor of shape (num_cells, cell_cutout,cell_cutout,channels)
    """

    return ThreadPoolExecutor().submit(process_cells,
                                       img=img,
                                       x=x,
                                       y=y,
                                       chunk=chunk,
                                       cell_cutout=cell_cutout,
                                       all_results=all_results).result()

def cell_seg(df_path,
             image_paths,
             img_channels='',
             cell_cutout=20,
             num_processes=1):
    """
    Cut out all provided cells from all images.

    Parameters:
    df_path (str): Path to pandas DataFrame containing cell positions of images
    image_paths (list): List of str paths to images
    img_channels (list): List of indices of image channels to extract
    cell_cutout (int): length/width of cut out
    num_processes (int): number of processes to use
    """

    df = pd.read_csv(df_path, header=0, sep=',')
    df['Centroid.X.px'] = df['Centroid.X.px'].round().astype(int)
    df['Centroid.Y.px'] = df['Centroid.Y.px'].round().astype(int)
    for image in tqdm(image_paths, desc='Segmenting Cells'):
        img = torch.from_numpy(load_img(image, img_channels))
        df_img = df[df['Image']==image.split('/')[-1]]
        x = df_img['Centroid.X.px'].values
        y = df_img['Centroid.Y.px'].values
        img.share_memory_()
        
        if x.shape[0] < 1:
            raise Exception(f'No coordinates in {df_path} for {image}!!!')
        
        # Split coordinates into chunks for parallel processing
        chunk_size = (len(x) + num_processes - 1) // num_processes
        chunks = [list(range(i, min(i + chunk_size, len(x)))) for i in range(0, len(x), chunk_size)]

        # Create between processes shared list of tensors to save cut outs
        all_results = torch.zeros((len(x),img.shape[0],cell_cutout,cell_cutout), dtype=img.dtype)
        all_results.share_memory_()
        num_processes = num_processes - 1 if num_processes >= 2 else 1
        processes = []
        
        for process_idx, chunk in enumerate(chunks):
            process = mp.Process(target=process_cells_wrapped, args=(img,
                                                             x[chunk],
                                                             y[chunk],
                                                             chunk,
                                                             cell_cutout,
                                                             all_results))
            process.start()
            processes.append(process)
        
        for process in processes:
            process.join()
        
        np.save(os.path.join(image.split('.')[0]+'_cells.npy'), all_results.numpy())
        #torch.save(all_results, os.path.join(image.split('.')[0]+'_cells.pt'))
        del img
        del all_results
        del df_img

def image_preprocess(path,
                     max_img=2**16,
                     img_channels='',
                     path_mean_std='',
                     cell_cutout=20,
                     num_processes=1):
    """
    Preprocess all tiffile images in directory.

    Parameters:
    path (str): Path to image directory
    max_img (int): Max possible pixel intensitiy of images
    img_channels (list): List of indices of image channels to extract
    path_mean_std (str): Path to dir containing saved numpy array of 
                         calculated channel-wise mean/std, if empty calculate
    cell_cutout (int): length/width of cut out
    num_processes (int): number of processes to use
    """
    raw_dir = os.path.abspath(os.path.join(path, '..'))
    df_path = [os.path.join(raw_dir, p) for p in os.listdir(raw_dir) if p.endswith(('.csv'))][0]
    img_paths = [os.path.join(path, p) for p in os.listdir(path) if p.endswith(('.tiff', '.tif'))]

    img_channels = img_channels if len(img_channels) == 0 else np.array([int(channel) for channel in img_channels.split(',')])
    cell_seg(df_path, img_paths, img_channels=img_channels, cell_cutout=cell_cutout, num_processes=num_processes)
    if len(path_mean_std) == 0:
        mean, std = calc_mean_std(img_paths, max_img=max_img)
        np.save(os.path.join(raw_dir, 'mean.npy'), mean)
        np.save(os.path.join(raw_dir, 'std.npy'), std)
