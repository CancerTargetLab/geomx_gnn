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
```sh
python -m --image_preprocess --preprocess_dir 'data/raw/{EXPERIMENT NAME}/' --cell_cutout 34 --preprocess_workers 26 --preprocess_channels 0,10,14,19 --preprocess_mean_std_dir 'data/raw/{EXPERIMENT NAME}/'
```
- `--image_preprocess`:  
Wether or not to run image preprocessing.
- `--preprocess_dir`:  
Directory path which contains .tiff images and `{measurements}.csv`.
- `--cell_cutout`:  
Pixel Height und Width to cutout around Cell postions.
- `--preprocess_workers`:  
Number of Threads to use to cutout Cells. Default 1, Optional.
- `--preprocess_channels`: 
Channels to preprocess. If not specified use all Channels. Seperate Channel indicies, starting from 0, with commas.
- `--preprocess_mean_std_dir`:  
Directory in which `mean.npy` and `std.npy` of Channels can be found, if already calculated. Calculation of mean and std gets skipped when specified.  

Preprocessing produces files for each Image: `{IMAGE NAME}_cells.pt` in the shape of `(Number of Cells, Channels, --cell_cutout, --cell_cutout)`.

## General arguments

- `--deterministic`:  
Wether or not to run NNs deterministicly.
- `--seed`:  
Seed for random computations.

## Cell Contrastive Learning

Next, we learn Visual Representations of Cells via Contrastive learning:  
```sh
python -m main --train_image_model --embed_image_data --output_name_image 'out/models/image_contrast_model.pt' --image_dir 'data/raw/{EXPERIMENT NAME}/' --resnet_model '18' --batch_size_image 4096 --epochs_image 100 --warmup_epochs_image 10 --num_workers_image 25 --lr_image 0.1 --embedding_size_image 32 --contrast_size_image 16 --early_stopping_image 100 --crop_factor 0.2 --train_ratio_image 0.6 --val_ratio_image 0.2
```

- `--train_image_model`:  
Wether or not to train the Image model.
- `--embed_image_data`:  
Wether or not to embed data with a given Image model.
- `--output_name_image`:  
Name of model,
- `--image_dir`:  
Directory in which preproccessed Images lay.
- `--resnet_model`:  
What ResNet model to choose, on of '18', '34', '50' and '101', default '18'.
- `--batch_size_image`:  
Number of Cell Images per Batch.
- `--epochs_image`:  
Number of epochs for which to train.
- `--warmup_epochs_image`:  
Number of Epochs in which learning rate gets increased.
- `--num_workers_image`:  
Number of worker processes to be used(loading data etc).
- `--lr_image`:  
Max learning rate of model
- `--embedding_size_image`:  
Linear net size used to embed data.
- `--contrast_size_image`:  
Linear net size on which to calculate the contrast loss.
- `--early_stopping_image`:  
Number of epochs after which to stop model run without improvement to val contrast loss.
- `--crop_factor`:  
Cell Image crop factor for Image augmentation.
- `--train_ratio_image`:  
Ratio of Cell Images upon which to train.
- `--val_ratio_image`:  
Ratio of Cell Images upon which to Validate.  

We extract visual representations after training a model to learn visual representations. Embedding produces files for each `{IMAGE NAME}_cells.pt` in the same directory: `{IMAGE NAME}_cells_embed.pt` in the shape of `(Number of Cells, --embedding_size_image)`.

## Learning to predict SC Expression

Next, we learn what each Cell contributes to the Count Data of an Image:  
```sh
python -m main --train_gnn --embed_gnn_data --output_name_graph 'out/models/image_graph_model.pt' --output_graph_embed '/out/graph_model/' --init_image_model 'out/models/image_contrast_model.pt' --init_graph_model 'out/models/graph_model.pt' --graph_dir 'data/' --graph_raw_subset_dir '{EXPERIMENT NAME}' --graph_label_data '{label}.csv'  --batch_size_graph 64 --epochs_graph 1000 --num_workers_graph 12 --lr_graph 0.005 --early_stopping_graph 50 --train_ratio_graph 0.6 --val_ratio_graph 0.2 --node_dropout 0.0 --edge_dropout 0.3 --cell_pos_jitter 40 --cell_n_knn 6 --subgraphs_per_graph 0 --num_hops_subgraph 11 --graph_model_type 'GAT' --graph_mse_mult 1 --graph_cos_sim_mult 1 --graph_entropy_mult 1 --layers_graph 3 --num_node_features 32 --num_edge_features 1 --num_embed_features 128 --heads_graph 4 --embed_dropout_graph 0.1 --conv_dropout_graph 0.1
```

- `--train_gnn`:  
Wether or not to train the Graph Model.
- `--embed_gnn_data`:  
Wether or not to embed predicted Cell Expression.
- `--output_name_graph`:  
Name of model/ where to save model.
- `--output_graph_embed`:  
Dir in which to embed Cell Expressions.
- `--init_image_model`:  
Name of pre-trained Image model to load. If not used, train from scratch. Only used when IMAGE in modeltype.
- `--init_graph_model`:  
Name of pre-trained Graph model to load. If not used, train from scratch. Only used when IMAGE in modeltype.
- `--graph_dir`:  
Where to find the `raw/` and `processed/` dirs.
- `--graph_raw_subset_dir`:  
How the subdir in raw/ and processed/ is called(`{EXPERIMENT NAME}`).
- `--graph_label_data`:  
`{label}.csv` label data in the raw dir containing count data.
- `--batch_size_graph`:  
Number of Graphs per Batch.
- `--epochs_graph`:  
Number of epochs for which to train.
- `--num_workers_graph`:  
Number of worker processes to be used(loading data etc).
- `--lr_graph`:  
Learning rate of model.
- `--early_stopping_graph`:  
Number of epochs without validation loss improvement after which to stop training.
- `--train_ratio_graph`:  
Ratio of Patients used for training.
- `--val_ratio_graph`:  
Ratio of Patients which are used for validation.
- `--node_dropout`:  
Probability of Graph Node dropout during training.
- `--edge_dropout`:  
Probability of Graph Edge dropout during training.
- `--cell_pos_jitter`:  
Positional Jittering during training of cells in pixel dist.
- `-cell_n_knn`:  
Number of Nearest Neighbours to calculate for each cell in graph.
- `--subgraphs_per_graph`:  
Number of Subgraphs per Graph to use for training, should be a quadratic integer. If 0, train with entire graph. Can only be non 0 if `{measurements}.csv` contains Cell count data.
- `--num_hops_subgraph`:  
Number of hops to create subgraph neighborhoods.
- `--graph_model_type`:  
Type of Model to train, one of {IMAGE}GAT+{_ph , _NB , _ZINB}, {IMAGE}LINT+{_ph , _nb , _zinb}. When IMAGE in name, then model is trained together with an Image Model. When _ph, _nb, _zinb or name, entropy Loss, NB Loss or ZiNB Loss gets calculated on predicted Cell Expression.
- `--graph_mse_mult`:  
Multiplier for L1 Loss.
- `--graph_cos_sim_mult`:  
Multiplier for Cosine Similarity Loss.
- `--graph_entropy_mult`:  
Multiplier for Entropy Loss, NB Loss or ZiNB Loss, depending on Model type.
- `--layers_graph`:  
Number of Layers in Graph.
- `-num_node_features`:  
Size of initial Node features.
- `--num_edge_features`:  
Size of edge features.
- `--num_embed_features`:  
Size to embed initial Node features to.
- `--heads_graph`:  
Number of Attention Heads for the Graph NN.
- `--embed_dropout_graph`:  
Percentage of embedded feature dropout chance.
- `--conv_dropout_graph`:  
Percentage of dropout chance between layers.

After training a model to predict the Expression of Single Cells, the predicted Expression of all Single Cells in the specified directory in which is trained get embeded in the specified output graph embed directory. Each graph gets embedded seperatly in shape `(Number of Cells, Number of Genes/Transcripts/Proteins/...)`. It is important to note that models trained on subgraphs will embed subgraphs, if the embedding is not done in a seperate call where `--subgraphs_per_graph` is set to 0. Subgraphs are stored in `processed/subgraphs/` if created.
## Visualizing Model runs
Model runs can be visualized as follows:  
```sh
python -m main --visualize_model_run --model_path 'out/models/graph_model.pt' --output_model_name 'Image Contrast Model' --figure_model_dir 'figures/graph_model/' --is_cs
```

- `--visualize_model_run`:  
Wether or not to Visualize statistics of model run.
- `--model_path`:  
Path and name of model save.
- `--output_model_name`:  
Name of model in figures.
- `--figure_model_dir`:  
Path to output figures to.
- `--is_cs`:  
Wether or not Cosine Similarity is used or Contrast Loss.

## Visualizing Expression Data

The predicted single cell Expression can be visualized:
```sh
python -m main --visualize_expression --vis_label_data '{label}.csv' --processed_subset_dir '{EXPERIMENT NAME}' --figure_dir 'figures/graph_model/' --embed_dir 'out/graph_model/' --vis_select_cells 50000 --vis_name '_graph_model'
```

- `--visualize_expression`:  
Wether or not to visualize predicted sc expression.
- `--vis_label_data`:  
Count data of Images, linked with Patient IDs.
- `--processed_subset_dir`:  
Subset directory of processed/ and raw/ of data.
- `--figure_dir`:  
Path to save images to
- `--embed_dir`:  
Path to predicted single cell data per Graph/Image.
- `--vis_select_cells`:  
Number of cells to perform dim reduction on. If 0, then all cells get reduced.
- `--vis_name`:  
Name added to figures name, saves processed data as `{NAME}.h5ad` in `out/`. Additional unproccesed save with all cells is named `{NAME}_all.h5ad`.

## Visualizing Spatialomics

The predicted single cell Expression can be also visualized on the Images themselfs:  
```sh
python -m main --visualize_image --vis_img_raw_subset_dir '{EXPERIMENT NAME}' --name_tiff 'CRC02.ome.tif' --figure_img_dir 'figures/graph_model/' --vis_protein 'CD45,CD8,Keratin,Ki67,Fibronectin,Some.Name' --vis_img_xcoords (0,0) --vis_img_ycoords (0,0) --vis_channel 0 --vis_all_channels --vis_name '_graph_model'
```

- `--visualize_image`:  
Wether or not to Visualize an Image.
- `--vis_img_raw_subset_dir`:  
Name of raw/ subsetdir which contains .tiff Images to visualize.
- `--name_tiff`:  
Name of .tiff Image to visualize.
- `--figure_img_dir`:  
Path to output figures to.
- `--vis_protein`:  
Proteins to visualize Expression over Image of, seperated by ,; . converts to space.
- `--vis_img_xcoords`:  
Image x coords, smaller first.
- `--vis_img_ycoords`:  
Image y coords, smaller first.
- `--vis_channel`:  
Image channel to visualize as background.
- `--vis_all_channels`:  
Wether or not to visualize all Image channels on their own.
- `--vis_name`:  
Name of `{NAME}.h5ad` produced via visualizing expression, needs to be given.

## Reproduce

Tutorials to repreduce our results can be found in the `reproduce/` directory.