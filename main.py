import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for image and GNN models")

    # Arguments for Image preprocessing
    parser.add_argument("--preprocess_dir", type=str, default="data/raw/p2106",
                        help="Directory in which .tiff files are for preprocessing")
    parser.add_argument("--preprocess_channels", type=str, default="",
                        help="Indices of channels to preprocess, seperated by , and empty if all channels")
    parser.add_argument("--preprocess_mean_std_dir", type=str, default="",
                        help="Directory in which to find already calculated mean.npy and std.npy per channel, empty if not used")
    parser.add_argument("--cell_cutout", type=int, default=20,
                        help="Size*Size cutout of cell, centered on Centroid Cell position")
    parser.add_argument("--preprocess_workers", type=int, default=1,
                        help="Number of Workers to use for cell cutout")
    parser.add_argument("--image_preprocess", action="store_true", default=False,
                        help="Wether or not to preprocess images via ZScore normalisation")

    # Arguments for image model
    parser.add_argument("--image_dir", type=str, default="data/raw/TMA1_preprocessed",
                        help="Directory in which preproccessed Images lay")
    parser.add_argument("--batch_size_image", type=int, default=256,
                        help="Number of Cell Images per Batch")
    parser.add_argument("--epochs_image", type=int, default=100,
                        help="Number of epochs for which to train")
    parser.add_argument("--warmup_epochs_image", type=int, default=10,
                        help="Number of Epochs in which learning rate gets increased")
    parser.add_argument("--num_workers_image", type=int, default=1,
                        help="Number of worker processes to be used(loading data etc)")
    parser.add_argument("--lr_image", type=float, default=0.1,
                        help="Learning rate of model")
    parser.add_argument("--embedding_size_image", type=int, default=256,
                        help="Linear net size used to embed data")
    parser.add_argument("--contrast_size_image", type=int, default=124,
                        help="Linear net size on which to calculate the contrast loss")
    parser.add_argument("--early_stopping_image", type=int, default=100,
                        help="Number of epochs after which to stop model run without improvement to val contrast loss")
    parser.add_argument("--crop_factor", type=float, default=0.5,
                        help="Cell Image crop factor for Image augmentation")
    parser.add_argument("--train_ratio_image", type=float, default=0.6,
                        help="Ratio of Cell Images upon which to train")
    parser.add_argument("--val_ratio_image", type=float, default=0.2,
                        help="Ratio of Cell Images upon which to Validate")
    parser.add_argument("--resnet_model", type=str, default="18",
                        help="What ResNet model to choose, on of 18, 34, 50 and 101")
    parser.add_argument("--output_name_image", type=str, default="out/image_contrast.pt",
                        help="Name of model")
    parser.add_argument("--train_image_model", action="store_true", default=False,
                        help="Wether or not to train the Image model")
    parser.add_argument("--embed_image_data", action="store_true", default=False,
                        help="Wether or not to embed data with a given Image model")

    # Arguments for the GNN model
    parser.add_argument("--graph_dir", type=str, default="data/",
                        help="Where to find the raw/ and processed/ dirs")
    parser.add_argument("--graph_raw_subset_dir", type=str, default="TMA1_preprocessed",
                        help="How the subdir in raw/ and processed/ is called")
    parser.add_argument("--graph_label_data", type=str, default="OC1_all.csv",
                        help=".csv label data in the raw dir containing count data")
    parser.add_argument("--data_is_log_graph", action="store_true", default=False,
                        help="Wether or not the count data is log or not")
    parser.add_argument("--batch_size_graph", type=int, default=1,
                        help="Number of Graphs per Batch")
    parser.add_argument("--epochs_graph", type=int, default=100,
                        help="Number of epochs for which to train")
    #parser.add_argument("--warmup_epochs_graph", type=int, default=10)
    parser.add_argument("--num_workers_graph", type=int, default=1,
                        help="Number of worker processes to be used(loading data etc)")
    parser.add_argument("--lr_graph", type=float, default=0.005,
                        help="Learning rate of model")
    parser.add_argument("--early_stopping_graph", type=int, default=10,
                        help="Number of epochs without validation loss improvement after which to stop training")
    parser.add_argument("--train_ratio_graph", type=float, default=0.6,
                        help="Ratio of Patients used for training")
    parser.add_argument("--val_ratio_graph", type=float, default=0.2,
                        help="Ratio of Patients which are used for validation")
    parser.add_argument("--node_dropout", type=float, default=0.3,
                        help="Probability of Graph Node dropout during training")
    parser.add_argument("--edge_dropout", type=float, default=0.5,
                        help="Probability of Graph Edge dropout during training")
    parser.add_argument("--cell_pos_jitter", type=int, default=40,
                        help="Positional Jittering during training of cells in pixel dist")
    parser.add_argument("--cell_n_knn", type=int, default=6,
                        help="Number of Nearest Neighbours to calculate for each cell in graph")
    parser.add_argument("--subgraphs_per_graph", type=int, default=0,
                        help="Number of Subgraphs per Graph to use for training. If 0, train with entire graph")
    parser.add_argument("--num_hops_subgraph", type=int, default=10,
                        help="Number of hops to create subgraph neighborhoods")
    parser.add_argument("--graph_model_type", type=str, default="GAT",
                        help="Type of Model to train, one of GAT, GAT_ph, LIN, LIN_ph")
    parser.add_argument("--graph_mse_mult", type=float, default=1.0,
                        help="Multiplier for MSE Loss")
    parser.add_argument("--graph_cos_sim_mult", type=float, default=1.0,
                        help="Multiplier for Cosine Similarity Loss")
    parser.add_argument("--graph_entropy_mult", type=float, default=1.0,
                        help="Multiplier for Entropy Loss")
    parser.add_argument("--layers_graph", type=int, default=1,
                        help="Number of Layers in Graph")
    parser.add_argument("--num_node_features", type=int, default=256,
                        help="Size of initial Node features")
    parser.add_argument("--num_edge_features", type=int, default=1,
                        help="Size of edge features")
    parser.add_argument("--num_embed_features", type=int, default=128,
                        help="Size to embed initial Node features to")
    parser.add_argument("--heads_graph", type=int, default=1,
                        help="Number of Attention Heads for the Graph NN")
    parser.add_argument("--embed_dropout_graph", type=float, default=0.1,
                        help="Percentage of embedded feature dropout chance")
    parser.add_argument("--conv_dropout_graph", type=float, default=0.1,
                        help="Percentage of dropout chance between layers")
    parser.add_argument("--output_name_graph", type=str, default="out/ROI.pt",
                        help="Name of model")
    parser.add_argument("--output_graph_embed", type=str, default="out/",
                        help="Dir in which to embed Cell Expressions")
    parser.add_argument("--init_image_model", type=str, default="",
                        help="Dir in which to embed Cell Expressions")
    parser.add_argument("--init_graph_model", type=str, default="",
                        help="Dir in which to embed Cell Expressions")
    parser.add_argument("--train_gnn", action="store_true", default=False,
                        help="Wther or not to train the Graph Model")
    parser.add_argument("--embed_gnn_data", action="store_true", default=False,
                        help="Wether or not to embed predicted Cell Expression")

    # Arguments for the TME model
    # parser.add_argument("--tme_dir", type=str, default="data/")
    # parser.add_argument("--tme_raw_subset_dir", type=str, default="TMA1_preprocessed")
    # parser.add_argument("--tme_label_data", type=str, default="OC1_all.csv")
    # parser.add_argument("--data_is_log_tme", action="store_true", default=False)
    # parser.add_argument("--batch_size_tme", type=int, default=4)
    # parser.add_argument("--subgraphs_per_batch_tme", type=int, default=64)
    # parser.add_argument("--epochs_tme", type=int, default=100)
    # parser.add_argument("--warmup_epochs_tme", type=int, default=10)
    # parser.add_argument("--num_workers_tme", type=int, default=1)
    # parser.add_argument("--lr_tme", type=float, default=0.0001)
    # parser.add_argument("--early_stopping_tme", type=int, default=10)
    # parser.add_argument("--train_ratio_tme", type=float, default=0.6)
    # parser.add_argument("--val_ratio_tme", type=float, default=0.2)
    # parser.add_argument("--node_dropout_tme", type=float, default=0.3)
    # parser.add_argument("--edge_dropout_tme", type=float, default=0.5)
    # parser.add_argument("--num_hops_tme", type=int, default=2)
    # parser.add_argument("--layers_tme", type=int, default=2)
    # parser.add_argument("--num_node_features_tme", type=int, default=256)
    # parser.add_argument("--num_edge_features_tme", type=int, default=1)
    # parser.add_argument("--num_embed_features_tme", type=int, default=128)
    # parser.add_argument("--num_out_features_tme", type=int, default=64)
    # parser.add_argument("--heads_tme", type=int, default=1)
    # parser.add_argument("--embed_dropout_tme", type=float, default=0.1)
    # parser.add_argument("--conv_dropout_tme", type=float, default=0.1)
    # parser.add_argument("--output_name_tme", type=str, default="out/test.pt")
    # parser.add_argument("--train_tme", action="store_true", default=False)
    # parser.add_argument("--embed_tme_data", action="store_true", default=False)
    # parser.add_argument("--output_tme_embed", type=str, default="out/")

    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Wether or not to run NNs deterministicly")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for random computations")

    # Visualize Expression 
    parser.add_argument("--visualize_expression", action="store_true", default=False,
                        help="Wether or not to visualize predicted sc expression")
    parser.add_argument("--vis_label_data", type=str, default="OC1_all.csv",
                        help="Count data of Images, linked with Patient IDs")
    parser.add_argument("--processed_subset_dir", type=str, default="TMA1_preprocessed",
                        help="Subset directory of processed/ and raw/ of data")
    parser.add_argument("--figure_dir", type=str, default="figures/",
                        help="Path to save images to")
    parser.add_argument("--embed_dir", type=str, default="out/",
                        help="Path to predicted single cell data per Graph/Image")
    parser.add_argument("--vis_select_cells", type=int, default=0,
                        help="Number of cells to perform dim reduction on. If 0, then all cells get reduced")
    parser.add_argument("--vis_name", type=str, default="_cells",
                        help="Name added to figures name, saves processed data as NAME.h5ad")   #alo for visualize image

    # Visualize Image
    parser.add_argument("--visualize_image", action="store_true", default=False,
                        help="Wether or not to Visualize an Image")
    parser.add_argument("--vis_img_raw_subset_dir", type=str, default="TMA1_preprocessed",
                        help="Name of raw/ subsetdir which contains .tiff Images to visualize")
    parser.add_argument("--name_tiff", type=str, default="027-2B27.tiff",
                        help="Name of .tiff Image to visualize")
    parser.add_argument("--figure_img_dir", type=str, default="figures/",
                        help="Path to output figures to")
    parser.add_argument("--vis_protein", type=str, default="",
                        help="Proteins to visualize Expression over Image of, seperated by ,; . converts to space")
    parser.add_argument("--vis_img_xcoords", type=tuple, default=(0,0),
                        help="Image x coords, smaller first")
    parser.add_argument("--vis_img_ycoords", type=tuple, default=(0,0),
                        help="Image y coords, smaller first")
    parser.add_argument("--vis_channel", type=int, default=0,
                        help="Image channel to visualize as background")
    parser.add_argument("--vis_all_channels", action="store_true", default=False,
                        help="Wether or not to visualize all Image channels on their own")

    # Visualize Model Run
    parser.add_argument("--visualize_model_run", action="store_true", default=False,
                        help="Wether or not to Visualize statistics of model run")
    parser.add_argument("--model_path", type=str, default="out/models/ROI.pt",
                        help="Path and name of model save")
    parser.add_argument("--output_model_name", type=str, default="ROI",
                        help="Name of model in figures")
    parser.add_argument("--figure_model_dir", type=str, default="figures/",
                        help="Path to output figures to")
    parser.add_argument("--is_cs", action="store_true", default=False,
                        help="Wether or not Cosine Similarity is used or Contrast Loss")

    return parser.parse_args()


def main(**args):
    if args['image_preprocess']:
        from src.utils.image_preprocess import image_preprocess as ImagePreprocess
        ImagePreprocess(path=args['preprocess_dir'], 
                        img_channels=args['preprocess_channels'],
                        path_mean_std=args['preprocess_mean_std_dir'],
                        cell_cutout=args['cell_cutout'],
                        num_processes=args['preprocess_workers'])
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
    # if args['train_tme']:
    #     from src.run.TMETrain import train as TMETrain
    #     TMETrain(raw_subset_dir=args['tme_raw_subset_dir'],
    #             label_data=args['tme_label_data'],
    #             output_name=args['output_name_tme'],
    #             args=args)
    # if args['embed_tme_data']:
    #     from src.run.TMEEmbed import embed as TMEEmbed
    #     TMEEmbed(args)
    if args['visualize_expression']:
        from src.explain.VisualizeExpression import visualizeExpression
        visualizeExpression(processed_dir=args['processed_subset_dir'],
                            embed_dir=args['embed_dir'],
                            label_data=args['vis_label_data'],
                            figure_dir=args['figure_dir'],
                            name=args['vis_name'],
                            select_cells=args['vis_select_cells'])
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
