import argparse
from src.utils.image_preprocess import image_preprocess
from src.run.CellContrastTrain import train as ImageTrain
from src.run.CellContrastEmbed import embed as CellContrastEmbed
from src.run.GraphTrain import train as GraphTrain
from src.run.GraphEmbed import embed as GraphEmbed
from src.run.TMETrain import train as TMETrain
#from src.run.TMEEmbed import embed as TMEEmbed

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
    parser.add_argument("--num_workers_image", type=int, default=8)
    parser.add_argument("--lr_image", type=float, default=0.1)
    parser.add_argument("--embedding_size_image", type=int, default=256)
    parser.add_argument("--contrast_size_image", type=int, default=124)
    parser.add_argument("--early_stopping_image", type=int, default=10)
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
    parser.add_argument("--data_is_log_graph", type=bool, default=False)
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
    parser.add_argument("--data_is_log_tme", type=bool, default=False)
    parser.add_argument("--batch_size_tme", type=int, default=1)
    parser.add_argument("--subgraphs_per_batch_tme", type=int, default=10)
    parser.add_argument("--epochs_tme", type=int, default=100)
    parser.add_argument("--warmup_epochs_tme", type=int, default=10)
    parser.add_argument("--num_workers_tme", type=int, default=0)
    parser.add_argument("--lr_tme", type=float, default=0.005)
    parser.add_argument("--early_stopping_tme", type=int, default=10)
    parser.add_argument("--train_ratio_tme", type=float, default=0.6)
    parser.add_argument("--val_ratio_tme", type=float, default=0.2)
    parser.add_argument("--node_dropout_tme", type=float, default=0.3)
    parser.add_argument("--edge_dropout_tme", type=float, default=0.5)
    parser.add_argument("--num_hops_tme", type=int, default=1)
    parser.add_argument("--layers_tme", type=int, default=1)
    parser.add_argument("--num_node_features_tme", type=int, default=256)
    parser.add_argument("--num_edge_features_tme", type=int, default=1)
    parser.add_argument("--num_embed_features_tme", type=int, default=64)
    parser.add_argument("--num_out_features_tme", type=int, default=32)
    parser.add_argument("--heads_tme", type=int, default=1)
    parser.add_argument("--embed_dropout_tme", type=float, default=0.1)
    parser.add_argument("--conv_dropout_tme", type=float, default=0.1)
    parser.add_argument("--output_name_tme", type=str, default="out/test.pt")
    parser.add_argument("--train_tme", action="store_true", default=True)
    parser.add_argument("--embed_tme_data", action="store_true", default=False)
    parser.add_argument("--output_tme_embed", type=str, default="out/")

    parser.add_argument("--seed", type=int, default=44)

    return parser.parse_args()


def main(args):
    if args['image_preprocess']:
        image_preprocess(args['preprocess_dir'])
    if args['train_image_model']:
        ImageTrain(args)
    if args['embed_image_data']:
        CellContrastEmbed(args)
    if args['train_gnn']:
        GraphTrain(args)
    if args['embed_gnn_data']:
        GraphEmbed(args)
    if args['train_tme']:
        TMETrain(args)
    # if args['embed_tme_data']:
    #     TMEEmbed(args)


if __name__ == '__main__':
    args = vars(parse_args())
    main(args)
