import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for image and GNN models")

    # Arguments for Image preprocessing
    parser.add_argument("--preprocess_dir", type=str, default="data/raw/p2106")
    parser.add_argument("--image_preprocess", action="store_true", default=False)

    # Arguments for image model
    parser.add_argument("--image_dir", type=str, default="data/raw/TMA1_preprocessed")
    parser.add_argument("--batch_size_image", type=int, default=256)
    parser.add_argument("--epochs_image", type=int, default=100)
    parser.add_argument("--warmup_epochs_image", type=int, default=10)
    parser.add_argument("--num_workers_image", type=int, default=1)
    parser.add_argument("--lr_image", type=float, default=0.1)
    parser.add_argument("--embedding_size_image", type=int, default=256)
    parser.add_argument("--contrast_size_image", type=int, default=124)
    parser.add_argument("--early_stopping_image", type=int, default=100)
    parser.add_argument("--crop_factor", type=float, default=0.5)
    parser.add_argument("--train_ratio_image", type=float, default=0.6)
    parser.add_argument("--val_ratio_image", type=float, default=0.2)
    parser.add_argument("--resnet_model", type=str, default="18")
    parser.add_argument("--output_name_image", type=str, default="out/image_contrast.pt")
    parser.add_argument("--train_image_model", action="store_true", default=False)
    parser.add_argument("--embed_image_data", action="store_true", default=False)

    # Arguments for the GNN model
    parser.add_argument("--graph_dir", type=str, default="data/")
    parser.add_argument("--graph_raw_subset_dir", type=str, default="TMA1_preprocessed")
    parser.add_argument("--graph_label_data", type=str, default="OC1_all.csv")
    parser.add_argument("--data_is_log_graph", action="store_true", default=False)
    parser.add_argument("--batch_size_graph", type=int, default=1)
    parser.add_argument("--epochs_graph", type=int, default=100)
    parser.add_argument("--warmup_epochs_graph", type=int, default=10)
    parser.add_argument("--num_workers_graph", type=int, default=1)
    parser.add_argument("--lr_graph", type=float, default=0.005)
    parser.add_argument("--early_stopping_graph", type=int, default=10)
    parser.add_argument("--train_ratio_graph", type=float, default=0.6)
    parser.add_argument("--val_ratio_graph", type=float, default=0.2)
    parser.add_argument("--node_dropout", type=float, default=0.3)
    parser.add_argument("--edge_dropout", type=float, default=0.5)
    parser.add_argument("--graph_model_type", type=str, default="GAT")
    parser.add_argument("--graph_mse_mult", type=float, default=1.0)
    parser.add_argument("--graph_cos_sim_mult", type=float, default=1.0)
    parser.add_argument("--graph_entropy_mult", type=float, default=1.0)
    #parser.add_argument("--num_phenotypes_graph", type=int, default=15)
    #parser.add_argument("--num_phenotypes_layers_graph", type=int, default=2)
    parser.add_argument("--layers_graph", type=int, default=1)
    parser.add_argument("--num_node_features", type=int, default=256)
    parser.add_argument("--num_edge_features", type=int, default=1)
    parser.add_argument("--num_embed_features", type=int, default=128)
    parser.add_argument("--heads_graph", type=int, default=1)
    parser.add_argument("--embed_dropout_graph", type=float, default=0.1)
    parser.add_argument("--conv_dropout_graph", type=float, default=0.1)
    parser.add_argument("--output_name_graph", type=str, default="out/ROI.pt")
    parser.add_argument("--train_gnn", action="store_true", default=False)
    parser.add_argument("--embed_gnn_data", action="store_true", default=False)
    parser.add_argument("--output_graph_embed", type=str, default="out/")

    # Arguments for the TME model
    parser.add_argument("--tme_dir", type=str, default="data/")
    parser.add_argument("--tme_raw_subset_dir", type=str, default="TMA1_preprocessed")
    parser.add_argument("--tme_label_data", type=str, default="OC1_all.csv")
    parser.add_argument("--data_is_log_tme", action="store_true", default=False)
    parser.add_argument("--batch_size_tme", type=int, default=4)
    parser.add_argument("--subgraphs_per_batch_tme", type=int, default=64)
    parser.add_argument("--epochs_tme", type=int, default=100)
    parser.add_argument("--warmup_epochs_tme", type=int, default=10)
    parser.add_argument("--num_workers_tme", type=int, default=1)
    parser.add_argument("--lr_tme", type=float, default=0.0001)
    parser.add_argument("--early_stopping_tme", type=int, default=10)
    parser.add_argument("--train_ratio_tme", type=float, default=0.6)
    parser.add_argument("--val_ratio_tme", type=float, default=0.2)
    parser.add_argument("--node_dropout_tme", type=float, default=0.3)
    parser.add_argument("--edge_dropout_tme", type=float, default=0.5)
    parser.add_argument("--num_hops_tme", type=int, default=2)
    parser.add_argument("--layers_tme", type=int, default=2)
    parser.add_argument("--num_node_features_tme", type=int, default=256)
    parser.add_argument("--num_edge_features_tme", type=int, default=1)
    parser.add_argument("--num_embed_features_tme", type=int, default=128)
    parser.add_argument("--num_out_features_tme", type=int, default=64)
    parser.add_argument("--heads_tme", type=int, default=1)
    parser.add_argument("--embed_dropout_tme", type=float, default=0.1)
    parser.add_argument("--conv_dropout_tme", type=float, default=0.1)
    parser.add_argument("--output_name_tme", type=str, default="out/test.pt")
    parser.add_argument("--train_tme", action="store_true", default=False)
    parser.add_argument("--embed_tme_data", action="store_true", default=False)
    parser.add_argument("--output_tme_embed", type=str, default="out/")

    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)

    # Visualize Expression 
    parser.add_argument("--visualize_expression", action="store_true", default=False)
    parser.add_argument("--vis_label_data", type=str, default="OC1_all.csv")
    parser.add_argument("--processed_subset_dir", type=str, default="TMA1_preprocessed")
    parser.add_argument("--figure_dir", type=str, default="figures/")
    parser.add_argument("--embed_dir", type=str, default="out/")
    parser.add_argument("--vis_name", type=str, default="_cells")   #alo for visualize image

    # Visualize Image
    parser.add_argument("--visualize_image", action="store_true", default=False)
    parser.add_argument("--vis_img_raw_subset_dir", type=str, default="TMA1_preprocessed")
    parser.add_argument("--name_tiff", type=str, default="027-2B27.tiff")
    parser.add_argument("--figure_img_dir", type=str, default="figures/")
    parser.add_argument("--vis_protein", type=str, default="")
    parser.add_argument("--vis_channel", type=int, default=0)
    parser.add_argument("--vis_all_channels", action="store_true", default=False)

    # Visualize Model Run
    parser.add_argument("--visualize_model_run", action="store_true", default=False)
    parser.add_argument("--model_path", type=str, default="out/models/ROI.pt")
    parser.add_argument("--output_model_name", type=str, default="ROI")
    parser.add_argument("--figure_model_dir", type=str, default="figures/")
    parser.add_argument("--is_cs", action="store_true", default=False)

    return parser.parse_args()


def main(**args):
    if args['image_preprocess']:
        from src.utils.image_preprocess import image_preprocess as ImagePreprocess
        ImagePreprocess(path=args['preprocess_dir'])
    if args['train_image_model']:
        from src.run.CellContrastTrain import train as ImageTrain
        ImageTrain(image_dir=args['image_dir'],
                   output_name=args['output_name_image'],
                   args=args)
    if args['embed_image_data']:
        from src.run.CellContrastEmbed import embed as CellContrastEmbed
        CellContrastEmbed(image_dir=args['image_dir'],
                          model_name=args['output_name_image'],
                          args=args)
    if args['train_gnn']:
        from src.run.GraphTrain import train as GraphTrain
        GraphTrain(raw_subset_dir=args['graph_raw_subset_dir'],
                   label_data=args['graph_label_data'],
                   output_name=args['output_name_graph'],
                   args=args)
    if args['embed_gnn_data']:
        from src.run.GraphEmbed import embed as GraphEmbed
        GraphEmbed(raw_subset_dir=args['graph_raw_subset_dir'],
                   label_data=args['graph_label_data'],
                   model_name=args['output_name_graph'],
                   output_dir=args['output_graph_embed'],
                   args=args)
    if args['train_tme']:
        from src.run.TMETrain import train as TMETrain
        TMETrain(raw_subset_dir=args['tme_raw_subset_dir'],
                label_data=args['tme_label_data'],
                output_name=args['output_name_tme'],
                args=args)
    # if args['embed_tme_data']:
    #     from src.run.TMEEmbed import embed as TMEEmbed
    #     TMEEmbed(args)
    if args['visualize_expression']:
        from src.explain.VisualizeExpression import visualizeExpression
        visualizeExpression(processed_dir=args['processed_subset_dir'],
                            embed_dir=args['embed_dir'],
                            label_data=args['vis_label_data'],
                            figure_dir=args['figure_dir'],
                            name=args['vis_name'])
    if args['visualize_image']:
        from src.explain.VisualizeImage import visualizeImage
        visualizeImage(raw_subset_dir=args['vis_img_raw_subset_dir'],
                       name_tiff=args['name_tiff'],
                       figure_dir=args['figure_img_dir'],
                       vis_name=args['vis_name'],
                       args=args)
    if args['visualize_model_run']:
        from src.utils.per_epoch_metrics import epochMetrics
        epochMetrics(model_path=args['model_path'],
                     figure_dir=args['figure_model_dir'],
                     is_cs=args['is_cs'],
                     name=args['output_model_name'])


if __name__ == '__main__':
    args = vars(parse_args())
    main(**args)
