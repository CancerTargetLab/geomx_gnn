import numpy as np
import os
import skimage.io as io
import torch

#Implemented through adapting NaroNets Image processing.
#Original Implementation: https://github.com/djimenezsanchez/NaroNet/blob/main/src/NaroNet/Patch_Contrastive_Learning/preprocess_images.py#L145

def load_img(path):
    if path.endswith('.tiff', '.tif'):
        img = io.imread(path, pluggin='tifffile')
    else:
        file_end = path.split('/')[-1]
        print(f"Do not support opening of files of type {file_end}.")
        print(f"Could not open file {path}")
    
    return img

def calc_mean_std(image_paths, max_img=2**16):
    global_hist = None
    for img_p in image_paths:
        img = load_img(img_p)
        local_hist = [np.histogram(img[:,:,channel], bins=max_img+2, range=(0,max_img+2)) for channel in range(img.shape[2])]
        if global_hist:
            global_hist = [np.concatenate((global_hist[channel], local_hist[channel])) for channel in range(len(local_hist))]
        else:
            global_hist = local_hist
    
    mean = np.array([np.mean(hist_chan[0]*hist_chan[1][:hist_chan[1].shape[0]-1]) for hist_chan in global_hist])
    std = np.array([np.std(hist_chan[0]*hist_chan[1][:hist_chan[1].shape[0]-1]) for hist_chan in global_hist])
    return mean, std

def zscore(image_paths, mean, std):
    for img_p in image_paths:
        img = load_img(img_p)
        # following 4 lines copied from naronet preprocess_imagrs.py line 107-110 commit 7c419bc
        x,y,chan = img.shape[0],img.shape[1],img.shape[2] 
        img = np.reshape(img,(x*y,chan))
        img = (img-mean)/(std+1e-16)
        img = np.reshape(img,(x,y,chan))

        torch.save(torch.from_numpy(img), img_p.split('.')[0]+'.pt')

def image_preprocess(path, max_img=2**16):
    img_paths = [p for p in os.listdir(path) if p.endswith('.tiff', '.tif')]
    preprocessed_paths = [p for p in os.listdir(path) if p.endswith('.pt')]

    if len(img_paths) != len(preprocessed_paths):
        mean, std = calc_mean_std(img_paths, max_img=max_img)
        zscore(img_paths, mean, std)
