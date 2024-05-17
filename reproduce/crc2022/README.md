# CRC Atlas 2022

Follow the instructions to intall required packages and become familiar Image2Count.  

Lets start by creating a directory to download images into and then download the data we used(~1.4TB):
```sh
mkdir data/raw/CRC
cd data/raw/CRC
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC02.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC03.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC04.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC05.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC06.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC07.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC08.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC09.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC10.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC11.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC12.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC13.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC14.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC15.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC16.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC17.ome.tif
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC02-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC03-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC04-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC05-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC06-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC07-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC08-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC09-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC10-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC11-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC12-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC13-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC14-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC15-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC16-features.zip
wget http://lin-2021-crc-atlas.s3.amazonaws.com/data/CRC17-features.zip
wget "http://lin-2021-crc-atlas.s3.amazonaws.com/metadata/CRC202105 HTAN channel metadata.csv"
mkdir metadata
mkdir spatial
mv *metadata.csv metadata
cd ../../../
```

As a next step we create the `label.csv` and `measurements.csv` files and clean up the directory:
```sh
python -m src/utils/crc_label_and_pos_data.py
mv data/raw/CRC/*features* data/raw/CRC/spatial
mv data/raw/CRC/CRC_label.csv data/raw/
```
If your setup is different to the one described here you might need to modif where to load data from and where to save it to in the utils script.  


If we want we can now also remove outlier cells with total counts in the top or bottom 1% and setup a subdir for this data:
```sh
mkdir data/raw/CRC_1p
ln -s /path/to/data/raw/CRC/*.tif data/raw/CRC_1p
python -m src/utils/crc_rm_outliers.py
mv data/raw/CRC_1p_measurements.csv data/raw/CRC_1p/
```
If paths differ from how described here, or another percentage threshold should be used, adjust the utils script.  

We can now progress and preprocess the images by ZScore normalisation and cutting out cells:
```sh
python -m main --image_preprocess --preprocess_dir data/raw/CRC/ --cell_cutout 34 --preprocess_workers 26 --preprocess_channels 0,10,14,19
```
The channels that we investigate are the first DNA channel, CD45, Keratin and CD8a. To see what channel is used for what marker, look at the metadata in `data/raw/CRC/metadata/`, markers that failed QC are not included in the Image, as well as 3 controll markers.  

If we have also removed outliers:  

```sh
python -m main --image_preprocess --preprocess_dir data/raw/CRC_1p/ --cell_cutout 34 --preprocess_workers 26 --preprocess_channels 0,10,14,19 --preprocess_mean_std_dir 'data/raw/CRC/'
```

In the next step we train a ResNet50 in contrastive fashion, following the implementation of SimCLR and oriented after NaroNet, and save embeddings of each cell(one file per ROI containing all cells) in the same dir as the image data:

```sh
python -m main --train_image_model --embed_image_data --output_name_image out/models/CRC_image_contrast_50_256_32.pt --embedding_size_image 256 --contrast_size_image 32 --crop_factor 0.2 --resnet_model '50'  --num_workers_image 14 --lr_image 0.002 --epochs_image 100 --warmup_epochs_image 10 --batch_size_image 1024 --image_dir data/raw/CRC_1p
```

Next we train a GAT and MLP model on the visual embedings to predict cell expression and save predictions in a specified dir:

```sh
python -m main --train_gnn --embed_gnn_data --graph_model_type GAT --graph_raw_subset_dir CRC_1p  --train_ratio_graph 0.6 --val_ratio_graph 0.2 --batch_size_graph 64 --epochs_graph 3000 --num_workers_graph 8 --num_node_features 256 --num_embed_features 128 --lr_graph 0.001 --layers_graph 3 --output_name_graph out/models/CRC_1p_gat.pt --output_graph_embed out/CRC_1p_gat/  --graph_label_data CRC_1p_label.csv --early_stopping_graph 50 --heads_graph 16 --node_dropout 0.0 --edge_dropout 0.3 --subgraphs_per_graph 49 --num_hops 11 --seed 44  
python -m main --train_gnn --embed_gnn_data --graph_model_type LIN --graph_raw_subset_dir CRC_1p  --train_ratio_graph 0.6 --val_ratio_graph 0.2 --batch_size_graph 64 --epochs_graph 3000 --num_workers_graph 8 --num_node_features 256 --num_embed_features 128 --lr_graph 0.001 --layers_graph 6 --output_name_graph out/models/CRC_1p_lin.pt --output_graph_embed out/CRC_1p_lin/  --graph_label_data CRC_1p_label.csv --early_stopping_graph 50 --node_dropout 0.0 --subgraphs_per_graph 49 --num_hops 11 --seed 44
```

Single Cell analysis of the predictions and saving plots was done as follows:

```sh
python -m main --vis_select_cells 50000 --vis_label_data CRC_1p_label.csv --processed_subset_dir CRC_1p --figure_dir figures/CRC_1p_gat/ --embed_dir out/CRC_1p_gat/ --vis_name _CRC_1p_gat --visualize_expression  
python -m main --vis_select_cells 50000 --vis_label_data CRC_1p_label.csv --processed_subset_dir CRC_1p --figure_dir figures/CRC_1p_lin/ --embed_dir out/CRC_1p_lin/ --vis_name _CRC_1p_lin --visualize_expression
```

We are able to calculate single cell correlation between predicted and actual cell expression of each gene by changing initial varibles in `src/explain/correlation.py` to match the outputs for the `_all.h5ad` file and the coressponding `--vis_name` as well as `--graph_raw_subset_dir` andthen executing the python script.  
We change the size of each created subgraph to `[1, 2, 3, 5, 8, 11]` hop subgraphs and embed model predictions, investigating the change in correlation value when comparing expression of differently sized "spots" instead of single cells:

```sh
python -m main --embed_gnn_data --graph_model_type GAT --graph_raw_subset_dir CRC_1p --batch_size_graph 64 --num_workers_graph 8 --num_node_features 256 --num_embed_features 128 --lr_graph 0.001 --layers_graph 3 --output_name_graph out/models/CRC_1p_gat.pt --output_graph_embed out/CRC_1p_gat_900_1/  --graph_label_data CRC_1p_label.csv --heads_graph 16 --subgraphs_per_graph 900 --num_hops 1 --seed 44
python -m main --vis_select_cells 50000 --vis_label_data CRC_1p_label.csv --processed_subset_dir CRC_1p --figure_dir figures/CRC_1p_gat/900_1/ --embed_dir out/CRC_1p_gat_900_1/ --vis_name _CRC_1p_gat_900_1 --visualize_expression 

python -m main -embed_gnn_data --graph_model_type LIN --graph_raw_subset_dir CRC_1p --batch_size_graph 64 --num_workers_graph 8 --num_node_features 256 --num_embed_features 128 --lr_graph 0.001 --layers_graph 6 --output_name_graph out/models/CRC_1p_lin.pt --output_graph_embed out/CRC_1p_lin_8001/  --graph_label_data CRC_1p_label.csv --subgraphs_per_graph 900 --num_hops 1 --seed 44
python -m main --vis_select_cells 50000 --vis_label_data CRC_1p_label.csv --processed_subset_dir CRC_1p --figure_dir figures/CRC_1p_lin/900_1/ --embed_dir out/CRC_1p_lin_900_1/ --vis_name _CRC_1p_lin_900_1 --visualize_expression
```
We repeat this procedure for all numbers in `[1, 2, 3, 5, 8, 11]` and change naming/ number of hops accordingly.  
To visualize predicted expression on images and save visualizations we execute:

```sh
python -m main --visualize_image --vis_name _crc_1p_gat_all.h5ad --vis_img_raw_subset_dir CRC_1p --vis_channel 0 --name_tiff CRC03.ome.tif --figure_img_dir figures/crc_1p_gat/crc03/ --vis_protein Hoechst1,CD3,Ki67,CD4,CD20,CD163,Ecadherin,LaminABC,PCNA,NaKATPase,Keratin,CD45,CD68,FOXP3,Vimentin,Desmin,Ki67_570,CD45RO,aSMA,PD1,CD8a,PDL1,CDX2,CD31,Collagen --vis_img_xcoords 22201 25893 --vis_img_ycoords 14729 18421 --vis_all_channels
python -m main --visualize_image --vis_name _crc_1p_lin_all.h5ad --vis_img_raw_subset_dir CRC_1p --vis_channel 0 --name_tiff CRC03.ome.tif --figure_img_dir figures/crc_1p_lin/crc03/ --vis_protein Hoechst1,CD3,Ki67,CD4,CD20,CD163,Ecadherin,LaminABC,PCNA,NaKATPase,Keratin,CD45,CD68,FOXP3,Vimentin,Desmin,Ki67_570,CD45RO,aSMA,PD1,CD8a,PDL1,CDX2,CD31,Collagen --vis_img_xcoords 22201 25893 --vis_img_ycoords 14729 18421 --vis_all_channels
```

Its is important to rememer to use the same `--vis_name` as previously with and appended `_all.h5ad`, as this is the file was saved when visualizing single cell anlysis that contains all cell predictions with file names, IDs, cell location.  
Lastly, we visualize model metrics:

```sh
python -m main --visualize_model_run --model_path out/models/crc_1p_gat.pt --output_model_name crc_1p_gat --figure_model_dir figures/crc_1p_gat
python -m main --visualize_model_run --model_path out/models/crc_1p_gat.pt --output_model_name crc_1p_gat --figure_model_dir figures/crc_1p_gat
```

We also applied single cell analysis and image visualisation. This was done manualy through loading the `CRC_1p_measurements.csv` file in pandas, converting to a scanpy object, saving as `.h5ad` object to execute `--visualize_image` on, and manuly performing sc analysis following the steps in `src/explain/VisualizeExpression.py`.
