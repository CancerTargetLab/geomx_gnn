module load Anaconda3
conda activate geomx

python -m main --vis_label_data cosmx_label.csv --processed_subset_dir cosmx/test --figure_dir figures/cosmx/6_6/mean/dca/ --merge \
        --embed_dir out/cosmx_6_6/ --vis_name cosmx_6_6_mean --visualize_expression --has_expr_data --raw_subset_dir cosmx_dca

python -m main --vis_label_data cosmx_label.csv --processed_subset_dir cosmx/test --figure_dir figures/cosmx/17_0/mean/dca/ --merge \
        --embed_dir out/cosmx_17_0/ --vis_name cosmx_17_0_mean --visualize_expression --has_expr_data --raw_subset_dir cosmx_dca

python -m main --vis_label_data cosmx_label.csv --processed_subset_dir cosmx/test --figure_dir figures/cosmx/lin/mean/dca/ --merge \
        --embed_dir out/cosmx_lin/ --vis_name cosmx_lin_mean --visualize_expression --has_expr_data --raw_subset_dir cosmx_dca
