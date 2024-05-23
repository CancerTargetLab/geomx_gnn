import torch
from torch_geometric.nn import GATv2Conv, Sequential
import torch_geometric
from collections import OrderedDict
from src.models.CellContrastModel import ContrastiveLearning


class MeanAct(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return torch.clamp(torch.exp(x), 1e-5, 1e6)


class DispAct(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softplus = torch.nn.Softplus()
    
    def forward(self, x):
        return torch.clamp(self.softplus(x), 1e-4, 1e4)

def init_weights(layer):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
    elif isinstance(layer, torch_geometric.nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
    elif isinstance(layer, torch_geometric.nn.GATv2Conv):
        torch.nn.init.xavier_uniform_(layer.lin_l.weight)
        torch.nn.init.xavier_uniform_(layer.lin_r.weight)
        torch.nn.init.xavier_uniform_(layer.lin_edge.weight)


class GATBlock(torch.nn.Module):
    """
    A Graph Attention Network (GAT) block for processing graph data.
    """
    def __init__(self,
                num_edge_features=1,
                num_embed_features=10, 
                heads=1,
                conv_dropout=0.1,
                fill_value=0.0):
        """
        Initializes the GATBlock with the given parameters.

        Parameters:
        num_edge_features (int): Number of edge features.
        num_embed_features (int): Number of embedding features.
        heads (int): Number of attention heads.
        conv_dropout (float): Dropout rate for the convolution layer.
        fill_value (float): Fill value for the GAT layer.
        """

        super().__init__()
        self.gat = GATv2Conv(num_embed_features, 
                            num_embed_features, 
                            edge_dim=num_edge_features,
                            heads=heads,
                            dropout=conv_dropout,
                            fill_value=fill_value)
        self.norm1 = torch_geometric.nn.norm.LayerNorm(num_embed_features*heads)
        self.lin = torch.nn.Linear(heads*num_embed_features, num_embed_features)
        self.norm2 = torch_geometric.nn.norm.LayerNorm(num_embed_features)
        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the GATBlock.

        Parameters:
        x (torch.Tensor): Node feature tensor.
        edge_index (torch.Tensor): Edge indices.
        edge_attr (torch.Tensor): Edge attributes.

        Returns:
        torch.Tensor: The output tensor after processing through the GAT block.
        """
         
        identity = x
        x = self.gat(x, edge_index, edge_attr=edge_attr)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.lin(x)
        x = self.norm2(x)
        x += identity
        x = self.relu(x)
        return x

class ProjectionHead(torch.nn.Module):
    """
    Implemented in torch following the example of 
    https://github.com/google-research/simclr/blob/master/model_util.py#L141
    TODO: graph batch data
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_layers=2,
                 use_bn=True,
                 use_relu=True):
        """
        Initializes the ProjectionHead with the given parameters.

        Parameters:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        num_layers (int): Number of layers in the projection head.
        use_bn (bool): Whether to use batch normalization.
        use_relu (bool): Whether to use ReLU activation.
        """

        super().__init__()
        self.layers = torch.nn.ModuleList()
        dim = input_dim
        for j in range(num_layers):
            if j != num_layers - 1:
                # For the middle layers, use bias and ReLU activation.
                bias_relu = True
                next_dim = input_dim
            else:
                # For the final layer, neither bias nor ReLU is used.
                bias_relu = False
                next_dim = output_dim

            linear_layer = torch.nn.Linear(dim, next_dim, bias=bias_relu)
            self.layers.append(linear_layer)
            if use_bn and bias_relu:
                bn_layer = torch.nn.LayerNorm(next_dim)
                self.layers.append(bn_layer)
            
            if bias_relu and use_relu:
                relu_layer = torch.nn.ReLU(inplace=True)
                self.layers.append(relu_layer)

            dim = next_dim
        
        self.layers.apply(init_weights)

    def forward(self, hiddens):
        """
        Forward pass of the ProjectionHead.

        Parameters:
        hiddens (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output tensor after passing through the projection head.
        """

        for layer in self.layers:
            hiddens = layer(hiddens)
        
        return hiddens


class LinearBlock(torch.nn.Module):
    """
    A block of linear layers with normalization and activation.
    """
    def __init__(self,
                 input_dim):
        """
        Initializes the LinearBlock with the given input dimension.

        Parameters:
        input_dim (int): Dimension of the input features.
        """

        super().__init__()
        self.lin = torch.nn.Linear(input_dim,
                                   input_dim)
        self.norm = torch.nn.LayerNorm(input_dim)
        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Forward pass of the LinearBlock.

        Parameters:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output tensor after passing through the linear block.
        """

        identity = x
        x = self.lin(x)
        x = self.norm(x)
        x += identity
        x = self.relu(x)
        return x

class GraphLearning(torch.nn.Module):
    """
    A graph learning module using GAT blocks.
    """
    def __init__(self,
                 layers=1,
                 num_node_features=100, 
                 num_edge_features=1,
                 num_embed_features=10,
                 heads=1,
                 embed_dropout=0.1,
                 conv_dropout=0.1,):
        """
        Initializes the GraphLearning module with the given parameters.

        Parameters:
        layers (int): Number of GAT layers.
        num_node_features (int): Number of node features.
        num_edge_features (int): Number of edge features.
        num_embed_features (int): Number of embedding features.
        heads (int): Number of attention heads.
        embed_dropout (float): Dropout rate for the embedding layer.
        conv_dropout (float): Dropout rate for the convolution layer.
        """

        super().__init__()

        self.heads = heads
        # TODO: GraphNorm ?
        self.node_embed = torch.nn.Sequential(
            torch.nn.Linear(num_node_features, num_embed_features),
            torch.nn.LayerNorm(num_embed_features),
            torch.nn.Dropout(p=embed_dropout, inplace=True),
            torch.nn.ReLU(inplace=True)
            )

        blocks = []
        for _ in range(layers):
            blocks.append(GATBlock(num_embed_features=num_embed_features,
                                    num_edge_features=num_edge_features,
                                    heads=heads,
                                    conv_dropout=conv_dropout,
                                    fill_value=0.0))
        self.convs = torch.nn.Sequential(*blocks)
        
        self.conv_skip = torch_geometric.nn.models.JumpingKnowledge(mode='max')
        
        self.node_embed.apply(init_weights)
        self.convs.apply(init_weights)
        self.conv_skip.apply(init_weights)


    def forward(self, data):#x, edge_index, edge_attr):
        """
        Forward pass of the GraphLearning module.

        Parameters:
        data (torch_geometric.data.Data): Graph data containing node features, edge indices, and edge attributes.

        Returns:
        torch.Tensor: Output tensor after applying the graph learning module.
        """

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h_i = self.node_embed(x)
        # h = h_i.clone()
        h = [h_i]

        for conv in list(range(len(self.convs))):
            h_i = self.convs[conv](h_i, edge_index, edge_attr=edge_attr)

            h.append(h_i)
        
        h = self.conv_skip(h)

        return h


class ROIExpression(torch.nn.Module):
    """
    A PyTorch module for predicting sc expressions.
    """
    def __init__(self,
                 layers=1,
                 num_node_features=256, 
                 num_edge_features=1,
                 num_embed_features=128,
                 num_out_features=128,
                 heads=1,
                 embed_dropout=0.1,
                 conv_dropout=0.1,
                 mtype=False):
        """
        Initializes the ROIExpression module with the specified parameters.

        Parameters:
        layers (int): Number of GAT blocks.
        num_node_features (int): Number of node features.
        num_edge_features (int): Number of edge features.
        num_embed_features (int): Number of embedding features.
        num_out_features (int): Number of output features.
        heads (int): Number of attention heads.
        embed_dropout (float): Dropout rate for the embedding layers.
        conv_dropout (float): Dropout rate for the convolutional layers.
        mtype (bool): Model type indicating whether or not to use zero-inflated negative binomial (ZINB) or negative binomial (NB) distribution.
        """

        super().__init__()

        self.mtype = mtype
        self.nb = False
        self.zinb = False
        if mtype.endswith('_zinb'):
            self.zinb = True
            self.nb = True
        if mtype.endswith('_nb'):
            self.nb = True

        self.gnn = GraphLearning(layers=layers,
                                num_node_features=num_node_features, 
                                num_edge_features=num_edge_features,
                                num_embed_features=num_embed_features,
                                heads=heads,
                                embed_dropout=embed_dropout,
                                conv_dropout=conv_dropout,)
        
        self.pool = torch_geometric.nn.pool.global_add_pool

        self.project = ProjectionHead(input_dim=num_embed_features, 
                                      output_dim=num_out_features,
                                      num_layers=2)
        
        self.mean_act = MeanAct()

        if self.nb:
            self.mean = torch.nn.Sequential(
                OrderedDict([
                ('linear_m', torch.nn.Linear(num_embed_features, num_out_features)),
                ('meanact', MeanAct())
                ]))

            self.disp = torch.nn.Sequential(
                OrderedDict([
                ('linear_di', torch.nn.Linear(num_embed_features, num_out_features)),
                ('dispact', DispAct())
                ]))
            
            self.mean.apply(init_weights)
            self.disp.apply(init_weights)

        if self.zinb: 
            self.drop = torch.nn.Sequential(
                OrderedDict([
                ('linear_dr', torch.nn.Linear(num_embed_features, num_out_features)),
                ('sigmoid', torch.nn.Sigmoid())
                ]))
            
            self.drop.apply(init_weights)

    def forward(self, data, return_cells=False, return_mean=False):#x, edge_index, edge_attr, batch):
        """
        Forward pass of the ROIExpression module.

        Parameters:
        data (torch_geometric.data.Data): Graph data containing node features, edge indices, and edge attributes.
        return_cells (bool): Flag indicating whether to return cell-wise outputs.
        return_mean (bool): Flag indicating whether to return mean outputs.

        Returns:
        (torch.Tensor|tuple): Output tensor/tesnor tuple after applying the ROI expression module.
        """

        x = self.gnn(data)#x, edge_index, edge_attr)
        pred = self.project(x)
        if return_cells:
            if self.nb and return_mean:
                return self.mean(x)
            return self.mean_act(pred)
        else:
            if self.nb:
                mean = self.mean(x)
                disp = self.disp(x)
            if self.zinb:
                drop = self.drop(x)
            
            if self.nb and not self.zinb:
                return self.pool(self.mean_act(pred), batch=data.batch), self.mean_act(pred), mean, disp
            elif self.zinb and self.nb:
                return self.pool(self.mean_act(pred), batch=data.batch), self.mean_act(pred), mean, disp, drop
            else:
                return self.pool(self.mean_act(pred), batch=data.batch)


class ROIExpression_lin(torch.nn.Module):
    """
    A PyTorch module for predicting sc expressions.
    """
    def __init__(self,
                layers=1,
                num_node_features=256,
                num_embed_features=128,
                num_out_features=128,
                embed_dropout=0.1,
                conv_dropout=0.1,
                mtype=False):
        """
        Initializes the ROIExpression_lin module with the specified parameters.

        Parameters:
        layers (int): Number of linear blocks.
        num_node_features (int): Number of node features.
        num_embed_features (int): Number of embedding features.
        num_out_features (int): Number of output features.
        embed_dropout (float): Dropout rate for the embedding layers.
        conv_dropout (float): Dropout rate for the convolutional layers.
        mtype (bool): Model type indicating whether or not to use zero-inflated negative binomial (ZINB) or negative binomial (NB) distribution.
        """

        super().__init__()

        self.mtype = mtype
        self.nb = False
        self.zinb = False
        if mtype.endswith('_zinb'):
            self.zinb = True
            self.nb = True
        if mtype.endswith('_nb'):
            self.nb = True

        self.node_embed = torch.nn.Sequential(
            torch.nn.Linear(num_node_features, num_embed_features),
            torch.nn.LayerNorm(num_embed_features),
            torch.nn.Dropout(p=embed_dropout, inplace=True),
            torch.nn.ReLU(inplace=True)
            )
        
        blocks = []
        for _ in range(layers):
            blocks.append(LinearBlock(input_dim=num_embed_features))
        self.lin = torch.nn.Sequential(*blocks)
        
        self.project = ProjectionHead(input_dim=num_embed_features, 
                                      output_dim=num_out_features,
                                      num_layers=2)
        
        self.pool = torch_geometric.nn.pool.global_add_pool

        self.mean_act = MeanAct()

        if self.nb:
            self.mean = torch.nn.Sequential(
                OrderedDict([
                ('linear_m', torch.nn.Linear(num_embed_features, num_out_features)),
                ('meanact', MeanAct())
                ]))

            self.disp = torch.nn.Sequential(
                OrderedDict([
                ('linear_di', torch.nn.Linear(num_embed_features, num_out_features)),
                ('dispact', DispAct())
                ]))
            
            self.mean.apply(init_weights)
            self.disp.apply(init_weights)

        if self.zinb: 
            self.drop = torch.nn.Sequential(
                OrderedDict([
                ('linear_dr', torch.nn.Linear(num_embed_features, num_out_features)),
                ('sigmoid', torch.nn.Sigmoid())
                ]))
            
            self.drop.apply(init_weights)
    
    def forward(self, data, return_cells=False, return_mean=False):
        """
        Forward pass of the ROIExpression_lin module.

        Parameters:
        data (torch_geometric.data.Data): Graph data containing node features, edge indices, and edge attributes.
        return_cells (bool): Flag indicating whether to return cell-wise outputs.
        return_mean (bool): Flag indicating whether to return mean outputs.

        Returns:
        (torch.Tensor|tuple): Output tensor/tesnor tuple after applying the ROI expression module.
        """

        x = self.lin(self.node_embed(data.x))
        pred = self.project(x)
        if return_cells:
            if self.nb and return_mean:
                return self.mean(x)
            return self.mean_act(pred)
        else:
            if self.nb:
                mean = self.mean(x)
                disp = self.disp(x)
            if self.zinb:
                drop = self.drop(x)
            
            if self.nb and not self.zinb:
                return self.pool(self.mean_act(pred), batch=data.batch), self.mean_act(pred), mean, disp
            elif self.zinb and self.nb:
                return self.pool(self.mean_act(pred), batch=data.batch), self.mean_act(pred), mean, disp, drop
            else:
                return self.pool(self.mean_act(pred), batch=data.batch)
        

class ROIExpression_Image_gat(torch.nn.Module):
    """
    A PyTorch module combining image-based learning and GAT-based graph learning for predicting sc expressions.
    """
    def __init__(self,
                 channels,
                 embed=256,
                 contrast=124,
                 mode='train_combined',
                 resnet='101',
                 layers=1,
                 num_edge_features=1,
                 num_embed_features=128,
                 num_out_features=128,
                 heads=1,
                 embed_dropout=0.1,
                 conv_dropout=0.1,
                 mtype=False,
                 path_image_model='',
                 path_graph_model='') -> None:
        """
        Initializes the ROIExpression_Image_gat module with the specified parameters.

        Parameters:
        channels (int): Number of input channels for the image model.
        embed (int): Embedding size for the image model.
        contrast (int): Contrastive size for the image model.
        mode (str): Mode of the image model (e.g., 'train_combined').
        resnet (str): ResNet model variant (e.g., '101').
        layers (int): Number of GAT blocks.
        num_edge_features (int): Number of edge features.
        num_embed_features (int): Number of embedding features.
        num_out_features (int): Number of output features.
        heads (int): Number of attention heads.
        embed_dropout (float): Dropout rate for the embedding layers.
        conv_dropout (float): Dropout rate for the convolutional layers.
        mtype (bool): Model type indicating whether to use zero-inflated negative binomial (ZINB) or negative binomial (NB) distribution.
        path_image_model (str): Path to the pretrained image model.
        path_graph_model (str): Path to the pretrained graph model.
        """

        super().__init__()
        self.image = ContrastiveLearning(channels=channels,
                                        embed=embed,
                                        contrast=contrast,
                                        mode=mode,
                                        resnet=resnet)
        self.graph = ROIExpression(layers=layers,
                                num_node_features=embed,
                                num_edge_features=num_edge_features,
                                num_embed_features=num_embed_features,
                                num_out_features=num_out_features,
                                heads=heads,
                                embed_dropout=embed_dropout,
                                conv_dropout=conv_dropout,
                                mtype=mtype)
        if path_image_model:
            self.image.load_state_dict(torch.load(path_image_model)['model'])
        if path_graph_model:
            self.graph.load_state_dict(torch.load(path_graph_model)['model'])
        
    def forward(self, data, return_cells=False, return_mean=False):
        """
        Forward pass of the ROIExpression_Image_gat module.

        Parameters:
        data (torch_geometric.data.Data): Graph data containing node features, edge indices, and edge attributes.
        return_cells (bool): Flag indicating whether to return cell-wise outputs.
        return_mean (bool): Flag indicating whether to return mean outputs.

        Returns:
        Output tensor after applying the ROI expression module.
        """

        data.x = self.image.forward(data.x)
        return self.graph.forward(data, return_cells=return_cells, return_mean=return_mean)


class ROIExpression_Image_lin(torch.nn.Module):
    """
    A PyTorch module combining image-based learning and linear block-based graph learning for predicting sc expressions.
    """
    def __init__(self,
                 channels,
                 embed=256,
                 contrast=124,
                 mode='train_combined',
                 resnet='101',
                 layers=1,
                 num_embed_features=128,
                 num_out_features=128,
                 embed_dropout=0.1,
                 conv_dropout=0.1,
                 mtype=False,
                 path_image_model='',
                 path_graph_model='') -> None:
        """
        Initializes the ROIExpression_Image_lin module with the specified parameters.

        Parameters:
        channels (int): Number of input channels for the image model.
        embed (int): Embedding size for the image model.
        contrast (int): Contrastive size for the image model.
        mode (str): Mode of the image model (e.g., 'train_combined').
        resnet (str): ResNet model variant (e.g., '101').
        layers (int): Number of linear blocks.
        num_embed_features (int): Number of embedding features.
        num_out_features (int): Number of output features.
        embed_dropout (float): Dropout rate for the embedding layers.
        conv_dropout (float): Dropout rate for the convolutional layers.
        mtype (bool): Model type indicating whether to use zero-inflated negative binomial (ZINB) or negative binomial (NB) distribution.
        path_image_model (str): Path to the pretrained image model.
        path_graph_model (str): Path to the pretrained graph model.
        """

        super().__init__()
        self.image = ContrastiveLearning(channels=channels,
                                        embed=embed,
                                        contrast=contrast,
                                        mode=mode,
                                        resnet=resnet)
        self.graph = ROIExpression_lin(layers=layers,
                                        num_node_features=embed,
                                        num_embed_features=num_embed_features,
                                        num_out_features=num_out_features,
                                        embed_dropout=embed_dropout,
                                        conv_dropout=conv_dropout,
                                        mtype=mtype)
        if path_image_model:
            self.image.load_state_dict(torch.load(path_image_model)['model'])
        if path_graph_model:
            self.graph.load_state_dict(torch.load(path_graph_model)['model'])
        
    def forward(self, data, return_cells=False, return_mean=False):
        """
        Forward pass of the ROIExpression_Image_lin module.

        Parameters:
        data (torch_geometric.data.Data): Graph data containing node features, edge indices, and edge attributes.
        return_cells (bool): Flag indicating whether to return cell-wise outputs.
        return_mean (bool): Flag indicating whether to return mean outputs.

        Returns:
        torch.Tensor: Output tensor after applying the ROI expression module.
        """
        
        data.x = self.image.forward(data.x)
        return self.graph.forward(data, return_cells=return_cells, return_mean=return_mean)
        