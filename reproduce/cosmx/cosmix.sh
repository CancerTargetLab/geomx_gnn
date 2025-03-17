module load Anaconda3
conda activate geomx

python -m main --image_preprocess --preprocess_dir data/raw/nanostring/train/ --cell_cutout 166 --preprocess_workers 15 --calc_mean_std
python -m main --image_preprocess --preprocess_dir data/raw/nanostring/test/ --cell_cutout 166 --preprocess_workers 15

python src/utils/cells_kmeans.py

python -m main --train_image_model --output_name out/models/nanostring_image_model_166_516_128.pt --early_stopping 1000 --embed 516 --contrast 128 --resnet '50'  --num_workers 30 --epochs 300 --warmup_epochs 30 --crop_factor 0.5 --batch_size 512 --raw_subset_dir nanostring --n_clusters_image 30 --lr 0.1

python -m main --visualize_model_run --model_path out/models/nanostring_image_model_166_516_128.pt --output_model_name nanostring_image_model_166_516_128 --figure_model_dir figures/image_contrast/
python -m main --embed_image_data --output_name out/models/nanostring_image_model_166_516_128.pt --early_stopping 1000 --embed 516 --contrast 128 --resnet '50'  --num_workers 30 --epochs 300 --warmup_epochs 30 --crop_factor 0.5 --batch_size 512 --raw_subset_dir nanostring

python -m main --model_type Image2Count --raw_subset_dir nanostring --batch_size 64 --epochs 300 --num_workers 4 \
        --num_node_features 516 --num_gat_features 64 --num_embed_features 128 --lr 0.0001 --lin_layers 6 --gat_layers 6 \
        --output_name out/models/nanostring_6_6.pt  --label_data nanostring_label.csv \
        --early_stopping 25 --heads 8 --node_dropout 0.05 --embed_dropout 0.5 --edge_dropout 0.3 --data_use_log_graph \
        --train_gnn --num_cfolds 6 --subgraphs_per_graph 4 --num_hops 11  --weight_decay 5e-3

python -m main --model_type Image2Count --raw_subset_dir nanostring --batch_size 64 --epochs 300 --num_workers 4 \
        --num_node_features 516 --num_gat_features 64 --num_embed_features 128 --lr 0.0001 --lin_layers 17 --gat_layers 0 \
        --output_name out/models/nanostring_17_0.pt  --label_data nanostring_label.csv \
        --early_stopping 25 --heads 8 --node_dropout 0.05 --embed_dropout 0.5 --edge_dropout 0.3 --data_use_log_graph \
        --train_gnn --num_cfolds 6 --subgraphs_per_graph 4 --num_hops 11  --weight_decay 5e-3

python -m main --model_type LIN --raw_subset_dir nanostring --batch_size 64 --epochs 300 --num_workers 4 \
        --num_node_features 516 --lr 0.005 --output_name out/models/nanostring_lin.pt  --label_data nanostring_label.csv \
        --early_stopping 25 --node_dropout 0.05 --embed_dropout 0.5 --edge_dropout 0.3 --data_use_log_graph \
        --train_gnn --num_cfolds 6 --subgraphs_per_graph 4 --num_hops 11 --weight_decay 5e-3

for m in '0' '1' '2' '3' '4' '5' '6' '7' '8' '9';
do

    python -m main --model_type Image2Count --raw_subset_dir nanostring --batch_size 64 --num_workers 8 --num_node_features 516 \
        --num_embed_features 128  --lin_layers 6 --gat_layers 6 --output_name out/models/nanostring_6_6/$m.pt --output_graph_embed out/nanostring_6_6/$m/ \
        --label_data nanostring_label.csv --subgraphs_per_graph 0 --num_gat_features 64 --heads 8 --data_use_log_graph --embed_gnn_data --num_cfolds 0
    python -m main --vis_select_cells 50000 --vis_label_data nanostring_label.csv --processed_subset_dir nanostring/test --figure_dir figures/nanostring/6_6/$m/ \
            --embed_dir out/nanost0ring_6_6/$m/ --vis_name nanostring_6_6_$m --visualize_expression --has_expr_data --raw_subset_dir nanostring
    python -m main --visualize_model_run --model_path out/models/nanostring_6_6/$m.pt --output_model_name nanostring_"$m"_6_6 --figure_model_dir figures/nanostring/6_6/$m/

    python -m main --model_type Image2Count --raw_subset_dir nanostring --batch_size 64 --num_workers 8 --num_node_features 124 --num_gat_features 64 \
        --num_embed_features 128  --lin_layers 17 --gat_layers 0 --output_name out/models/nanostring_17_0/$m.pt --output_graph_embed out/nanostring_17_0/$m/ \
        --label_data nanostring_label.csv --subgraphs_per_graph 0 --data_use_log_graph --embed_gnn_data --num_cfolds 0
    python -m main --vis_select_cells 50000 --vis_label_data nanostring_label.csv --processed_subset_dir nanostring/test --figure_dir figures/nanostring/17_0/$m/ \
            --embed_dir out/nanostring_17_0/$m/ --vis_name nanostring_17_0_$m --visualize_expression --has_expr_data --raw_subset_dir nanostring
    python -m main --visualize_model_run --model_path out/models/nanostring17_0/$m.pt --output_model_name nanostring_"$m"_17_0 --figure_model_dir figures/nanostring/17_0/$m/

    python -m main --model_type LIN --raw_subset_dir nanostring --batch_size 64 --num_workers 8 --num_node_features 32 \
        --output_name out/models/nanostring_lin/$m.pt --output_graph_embed out/nanostring_lin/$m/ \
        --label_data nanostring_label.csv --subgraphs_per_graph 0 --data_use_log_graph --embed_gnn_data --num_cfolds 0
    python -m main --vis_select_cells 50000 --vis_label_data nanostring_label.csv --processed_subset_dir nanostring/test --figure_dir figures/nanostring/lin/$m/ \
            --embed_dir out/nanostring_lin/$m/ --vis_name nanostring_lin_$m --visualize_expression --has_expr_data --raw_subset_dir nanostring
    python -m main --visualize_model_run --model_path out/models/nanostring_lin/$m.pt --output_model_name nanostring_"$m"_lin --figure_model_dir figures/nanostring/lin/$m/

done

python -m main --vis_label_data nanostring_label.csv --processed_subset_dir nanostring/test --figure_dir figures/nanostring/6_6/mean/ --merge \
        --embed_dir out/nanostring_6_6/ --vis_name nanostring_6_6_mean --visualize_expression --has_expr_data --raw_subset_dir nanostring

python -m main --vis_label_data nanostring_label.csv --processed_subset_dir nanostring/test --figure_dir figures/nanostring/17_0/mean/ --merge \
        --embed_dir out/nanostring_17_0/ --vis_name nanostring_17_0_mean --visualize_expression --has_expr_data --raw_subset_dir nanostring

python -m main --vis_label_data nanostring_label.csv --processed_subset_dir nanostring/test --figure_dir figures/nanostring/lin/mean/ --merge \
        --embed_dir out/nanostring_lin/ --vis_name nanostring_lin_mean --visualize_expression --has_expr_data --raw_subset_dir nanostring
