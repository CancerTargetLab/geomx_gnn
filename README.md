# Image2Count: From MultiPlex Imaging to Single-Cell Spatialomics Data to Investigate the Tumor Immune Microenvironment  

## Environment setup

Our models are trained with nvidia GPUs. To run on GPUs appropiate CUDA versions must be installed. Installing via conda can result in cuda version mismatches or in the installation of CPU versions. Appropiate versions can be found [here](https://data.pyg.org/whl/).  
Setup via Anaconda environment:
```
conda install -c anaconda python=3.11.5  
pip install torch -f https://data.pyg.org/whl/torch-2.1.0+cu121.html  
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu121.html  
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html  
pip install torchvision -f https://data.pyg.org/whl/torch-2.1.0+cu121.html  
conda install -c conda-forge squidpy  
```

## Required Data
The following data is required to start training:  
1. .tiff images  
These are images of tissue to investigate. We recommend to save them in a seperate directory, and to link them to a working dir via `ln -s /path/to/images/*.tiff data/raw/{EXPERIMENT NAME}/`.
2. {measurements}.csv  
This CSV should be in the same dir as the linked images: `data/raw/{EXPERIMENT NAME}/`. The csv contains Cellpostion information of all images. It consists out of the following columns with these exact names: The first column is named `ROI` and contains the name of the images, minus everything after the first `.`(`CRC02.ome.tif`->`CRC02`). The second and third column should be named `Centroid.X.px` and `Centroid.Y.px`, which contain Cell positions in pixel(!) values. The 4th column should be named `Class` and contains Celltype information, if existing. This can also be empty. Every column after this is optional, and represents count data of the cells. This can be used to create smaller areas of the image to train on. This is only useful when already spatialomics count data for cells exists(e.g. to investigate model performace). This can be normalized counts, but o not use log normalized counts.
3. {label}.csv  
This .csv should be located in `data/raw/`. The csv contains information of the overall count data for an image. It consists out of the following columns: The first column is named `ROI` and contains the name of the image, following the same procedure as mentioned above in (2). The second column is named `Patient_ID` and contains some ID to which the Image corresponds. Every column after this is interpreted as Count data. Each entry has the whole count data of the whole Image. This can be normalized counts, but o not use log normalized counts.  

Now we are ready to go!

## Preprocessing
The first process is to normalize Image data and uniformly cut out Cells:  
```py
python -m --image_preprocess --preprocess_dir 'data/raw/{EXPERIMENT NAME}/' --cell_cutout 34 --preprocess_workers 26 --preprocess_channels 0,10,14,19 --preprocess_mean_std_dir 'data/raw/{EXPERIMENT NAME}/'
```

## Cell Contrastive Learning

## Learning to predict SC Expression

## Visualizing Model runs

## Visualizing Expression Data

## Visualizing Spatialomics

## Tutorial