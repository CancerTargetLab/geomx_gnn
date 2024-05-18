# OC GeoMx

Follow the instructions to intall required packages and become familiar with Image2Count.  

We have Image data of 636 ROIs with corresponding GeoMx count data previously normalized through geometric mean of H3, cell postions, and the Image data of the 1C54 TMA. Installing QuPath by following the [instructions](https://qupath.readthedocs.io/en/stable/docs/reference/building.html) to enable GPU usage, and install the Stardist extension following the [instructions](https://qupath.readthedocs.io/en/stable/docs/deep/stardist.html), loading extensions is done via letting QuPath execute a modified version of the script in `src/utils/qupath_include_extension.groovy` that points to a directory which has a subdir called extension, this subdir contains extension `.jar`. Segmentation is done with the script `src/utils/qupath_segment.groovy`, modify path to stardist model and channel name of `.tiff` files to segment. Script can be executed as follows:

```sh
./qupath/build/dist/QuPath/bin/QuPath script -p=1C-54/project.qpproj segment.groovy
```

With `1C-54/` being the directory containing the QuPath project.  
ROI image data is exported through executing the script `src/utils/qupath_create_roi_tiffs.groovy`. Cell postions get exported manualy and transformed to the correct format. Cell count information and cell segmentationdata of the 636 GeoMx ROIs is manualy transformed into the correct format, also utilizing script `src/utils/getRelevantCSVData.py`. For visual representation learning we combine label and measurements `.csv` of ROIs and 1C54 and create a directory holding the combined tiff files. We cut out cells and normalize image data:

```sh
python -m main --image_preprocess --preprocess_dir data/raw/1C54_hkgmh3/ --cell_cutout 20 --preprocess_workers 26 
```
We then learn representations of cells, and save learned representations of each cell:
```sh
python -m main --train_image_model --output_name_image out/models/1C54_hkgmh3_image_contrast_18_32_16.pt --embedding_size_image 32 --contrast_size_image 16 --crop_factor 0.2 --resnet_model '18'  --num_workers_image 14 --lr_image 0.1 --epochs_image 100 --warmup_epochs_image 10 --batch_size_image 1024 --image_dir data/raw/1C54_hkgmh3/
python -m main --embed_image_data --output_name_image out/models/1C54_hkgmh3_image_contrast_18_32_16.pt --embedding_size_image 32 --contrast_size_image 16  --resnet_model '18'  --num_workers_image 14 --batch_size_image 1024 --image_dir data/raw/hkgmh3/
python -m main --embed_image_data --output_name_image out/models/1C54_hkgmh3_image_contrast_18_32_16.pt --embedding_size_image 32 --contrast_size_image 16  --resnet_model '18'  --num_workers_image 14 --batch_size_image 1024 --image_dir data/raw/1C54/
```

We do not use RandomGaussianNoise per image channel for this training run.  
Next we train a GAT and MLP model on the visual embedings to predict cell expression and save predictions in a specified dir:

```sh
python -m main --train_gnn --embed_gnn_data --graph_model_type GAT --graph_raw_subset_dir hkgmh3  --train_ratio_graph 0.6 --val_ratio_graph 0.2 --batch_size_graph 64 --epochs_graph 3000 --num_workers_graph 8 --num_node_features 32 --num_embed_features 128 --lr_graph 0.001 --layers_graph 3 --output_name_graph out/models/hkgmh3_gat.pt --output_graph_embed out/hkgmh3_gat/  --graph_label_data hk_geomean_h3.csv --early_stopping_graph 50 --heads_graph 8 --node_dropout 0.0 --edge_dropout 0.3 --seed 44 --data_use_log_graph  
python -m main --train_gnn --embed_gnn_data --graph_model_type LIN --graph_raw_subset_dir hkgmh3  --train_ratio_graph 0.6 --val_ratio_graph 0.2 --batch_size_graph 64 --epochs_graph 3000 --num_workers_graph 8 --num_node_features 256 --num_embed_features 128 --lr_graph 0.001 --layers_graph 6 --output_name_graph out/models/hkgmh3_lin.pt --output_graph_embed out/CRC_1p_lin/  --graph_label_data hk_geomean_h3.csv --early_stopping_graph 50 --node_dropout 0.0 --seed 44 --data_use_log_graph
python -m main --embed_gnn_data --graph_model_type GAT --graph_raw_subset_dir 1C54  --train_ratio_graph 0.6 --val_ratio_graph 0.2 --batch_size_graph 64 --epochs_graph 3000 --num_workers_graph 8 --num_node_features 32 --num_embed_features 128 --lr_graph 0.001 --layers_graph 3 --output_name_graph out/models/1C54_gat.pt --output_graph_embed out/1C54_gat/  --graph_label_data 1C54_label.csv --early_stopping_graph 50 --heads_graph 8 --node_dropout 0.0 --edge_dropout 0.3 --seed 44  
python -m main --embed_gnn_data --graph_model_type LIN --graph_raw_subset_dir 1C54  --train_ratio_graph 0.6 --val_ratio_graph 0.2 --batch_size_graph 64 --epochs_graph 3000 --num_workers_graph 8 --num_node_features 256 --num_embed_features 128 --lr_graph 0.001 --layers_graph 6 --output_name_graph out/models/1C54_lin.pt --output_graph_embed out/1C54_lin/  --graph_label_data 1C54_label.csv --early_stopping_graph 50 --node_dropout 0.0 --seed 44
```

Single Cell analysis of the predictions and saving plots was done as follows, also done for 1C54 ROIs for which we do not have data:

```sh
python -m main --vis_select_cells 50000 --vis_label_data hk_geomean_h3.csv --processed_subset_dir hkgmh3 --figure_dir figures/hkgmh3_gat/ --embed_dir out/hkgmh3_gat/ --vis_name _hkgmh3_gat --visualize_expression  
python -m main --vis_select_cells 50000 --vis_label_data hk_geomean_h3.csv --processed_subset_dir hkgmh3 --figure_dir figures/hkgmh3_lin/ --embed_dir out/hkgmh3_lin/ --vis_name _hkgmh3_lin --visualize_expression
python -m main --vis_select_cells 50000 --vis_label_data 1C54_label.csv --processed_subset_dir 1C54 --figure_dir figures/hkgmh3_gat/1C54/ --embed_dir out/1C54_gat/ --vis_name _1C54_gat --visualize_expression  
python -m main --vis_select_cells 50000 --vis_label_data 1C54_label.csv --processed_subset_dir 1C54 --figure_dir figures/hkgmh3_lin/1C54/ --embed_dir out/1C54_lin/ --vis_name _1C54_lin --visualize_expression
```

We also visualize predicted sc expression on Images:

```sh
python -m main --visualize_image --vis_name _hkgmh3_lin_all.h5ad --vis_img_raw_subset_dir hkgmh3 --vis_channel 0 --name_tiff 002-3CII33.tiff --figure_img_dir figures/hkgmh3_lin/002-3CII33/ --vis_protein PanCk,Ki-67,SMA,CD8,CD4,CD3,PD-1,CD11c,CD68,CD45,Fibronectin,Pan-AKT,BCL6,BCLXL,BAD,p53,CD163,CD14,CD34,CD45RO,STING,B7-H3,CD44,CD127 --vis_all_channels
python -m main --visualize_image --vis_name _hkgmh3_gat_all.h5ad --vis_img_raw_subset_dir hkgmh3 --vis_channel 0 --name_tiff 002-3CII33.tiff --figure_img_dir figures/hkgmh3_gat/002-3CII33/ --vis_protein PanCk,Ki-67,SMA,CD8,CD4,CD3,PD-1,CD11c,CD68,CD45,Fibronectin,Pan-AKT,BCL6,BCLXL,BAD,p53,CD163,CD14,CD34,CD45RO,STING,B7-H3,CD44,CD127  --vis_all_channels
python -m main --visualize_image --vis_name _1C54_lin_all.h5ad --vis_img_raw_subset_dir 1C54 --vis_channel 0 --name_tiff 010-1C54.tiff --figure_img_dir figures/hkgmh3_lin/010-1C54/ --vis_protein PanCk,Ki-67,SMA,CD8,CD4,CD3,PD-1,CD11c,CD68,CD45,Fibronectin,Pan-AKT,BCL6,BCLXL,BAD,p53,CD163,CD14,CD34,CD45RO,STING,B7-H3,CD44,CD127 --vis_all_channels
python -m main --visualize_image --vis_name _1C54_gat_all.h5ad --vis_img_raw_subset_dir 1C54 --vis_channel 0 --name_tiff 010-1C54.tiff --figure_img_dir figures/hkgmh3_gat/010-1C54/ --vis_protein PanCk,Ki-67,SMA,CD8,CD4,CD3,PD-1,CD11c,CD68,CD45,Fibronectin,Pan-AKT,BCL6,BCLXL,BAD,p53,CD163,CD14,CD34,CD45RO,STING,B7-H3,CD44,CD127  --vis_all_channels
```

Lastly, we visualize metrics of the model run:

```sh
python -m main --visualize_model_run --model_path out/models/hkgmh3_lin.pt --output_model_name hkgmh3_lin --figure_model_dir figures/hkgmh3_lin
python -m main --visualize_model_run --model_path out/models/hkgmh3_gat_log.pt --output_model_name hkgmh3_gat --figure_model_dir figures/hkgmh3_gat
```
