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


def create_GATv2_conv(num_layers=1,
                      num_edge_features=1,
                      num_embed_features=10, 
                      heads=1,
                      conv_dropout=0.1,
                      embed_dropout=0.1,
                      fill_value=0.0):

            layers = torch.nn.ModuleList()

            for _ in list(range(num_layers)):
                if heads == 1:
                    gat_layer = Sequential('x, edge_index, edge_attr', [
                        (GATv2Conv(num_embed_features, 
                                        num_embed_features, 
                                        edge_dim=num_edge_features,
                                        heads=heads,
                                        dropout=conv_dropout,
                                        fill_value=fill_value), 'x, edge_index, edge_attr -> x1'),
                        (torch_geometric.nn.norm.LayerNorm(num_embed_features*heads)),
                        (torch.nn.ReLU(inplace=True))
                    ])
                    layers.append(gat_layer)
                else:
                    gat_layer = Sequential('x, edge_index, edge_attr', [
                        (GATv2Conv(num_embed_features, 
                                        num_embed_features, 
                                        edge_dim=num_edge_features,
                                        heads=heads,
                                        dropout=conv_dropout,
                                        fill_value=fill_value), 'x, edge_index, edge_attr -> x1'),
                        (torch_geometric.nn.norm.LayerNorm(num_embed_features*heads)),
                        (torch.nn.ReLU(inplace=True)),
                        (torch.nn.Linear(heads*num_embed_features, num_embed_features)),
                        (torch.nn.LayerNorm(num_embed_features)),
                        #(torch.nn.Dropout(p=embed_dropout, inplace=True)),
                        (torch.nn.ReLU(inplace=True))
                        ])
                    layers.append(gat_layer)

            return layers


class ProjectionHead(torch.nn.Module):
    """
    Implemented in torch following the example of 
    https://github.com/google-research/simclr/blob/master/model_util.py#L141
    TODO: graph batch data
    """
    def __init__(self, input_dim, output_dim, num_layers=2, use_bn=True, use_relu=True):
        super(ProjectionHead, self).__init__()
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
                bn_layer = torch_geometric.nn.norm.BatchNorm(next_dim)
                self.layers.append(bn_layer)
            
            if bias_relu and use_relu:
                relu_layer = torch.nn.ReLU()
                self.layers.append(relu_layer)

            dim = next_dim
        
        self.layers.apply(init_weights)

    def forward(self, hiddens):
        for layer in self.layers:
            hiddens = layer(hiddens)
        
        return hiddens


class GraphLearning(torch.nn.Module):
    def __init__(self,
                 layers=1,
                 num_node_features=100, 
                 num_edge_features=1,
                 num_embed_features=10,
                 heads=1,
                 embed_dropout=0.1,
                 conv_dropout=0.1,):
        super().__init__()

        self.heads = heads
        # TODO: GraphNorm ?
        self.node_embed = torch.nn.Sequential(
            torch.nn.Dropout(p=embed_dropout, inplace=True),
            torch.nn.Linear(num_node_features, num_embed_features),
            torch.nn.LayerNorm(num_embed_features),
            torch.nn.ReLU(inplace=True)
            )

        self.convs = create_GATv2_conv(num_layers=layers,
                                       num_embed_features=num_embed_features,
                                       num_edge_features=num_edge_features,
                                       heads=heads,
                                       conv_dropout=conv_dropout,
                                       embed_dropout=embed_dropout,
                                       fill_value=0.0)
        #TODO: skip connec as simple add as well?
        self.conv_skip = torch_geometric.nn.models.JumpingKnowledge(mode='max')
        
        self.node_embed.apply(init_weights)
        self.convs.apply(init_weights)
        self.conv_skip.apply(init_weights)


    def forward(self, data):#x, edge_index, edge_attr):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h_i = self.node_embed(x)
        # h = h_i.clone()
        h = [h_i]

        for conv in list(range(len(self.convs))):
            h_i = self.convs[conv](h_i, edge_index, edge_attr=edge_attr)

            # h = torch.concat((h, h_i), dim=1)
            h.append(h_i)
        
        h = self.conv_skip(h)

        return h


class ROIExpression(torch.nn.Module):
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
        x = self.gnn(data)#x, edge_index, edge_attr)
        pred = self.project(x)
        if return_cells:
            if self.nb and return_mean:
                return self.mean(x)
            return torch.abs(pred)
        else:
            if self.nb:
                mean = self.mean(x)
                disp = self.disp(x)
            if self.zinb:
                drop = self.drop(x)
            
            if self.nb and not self.zinb:
                return self.pool(torch.abs(pred), batch=data.batch), torch.abs(pred), mean, disp
            elif self.zinb and self.nb:
                return self.pool(torch.abs(pred), batch=data.batch), torch.abs(pred), mean, disp, drop
            else:
                return self.pool(torch.abs(pred), batch=data.batch)


class ROIExpression_lin(torch.nn.Module):
    def __init__(self,
                layers=1,
                num_node_features=256,
                num_embed_features=128,
                num_out_features=128,
                embed_dropout=0.1,
                conv_dropout=0.1,
                mtype=False):
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
        
        self.lin = ProjectionHead(input_dim=num_embed_features, 
                                    output_dim=num_embed_features,
                                    num_layers=layers)
        
        self.project = torch.nn.Linear(num_embed_features, num_out_features)
        
        self.pool = torch_geometric.nn.pool.global_add_pool

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
        x = self.lin(self.node_embed(data.x))
        pred = self.project(x)
        if return_cells:
            if self.nb and return_mean:
                return self.mean(x)
            return torch.abs(pred)
        else:
            if self.nb:
                mean = self.mean(x)
                disp = self.disp(x)
            if self.zinb:
                drop = self.drop(x)
            
            if self.nb and not self.zinb:
                return self.pool(torch.abs(pred), batch=data.batch), torch.abs(pred), mean, disp
            elif self.zinb and self.nb:
                return self.pool(torch.abs(pred), batch=data.batch), torch.abs(pred), mean, disp, drop
            else:
                return self.pool(torch.abs(pred), batch=data.batch)
        

class ROIExpression_Image_gat(torch.nn.Module):
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
        super().__init()
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
        data.x = self.image.forward(data.x)
        return self.graph.forward(data, return_cells=return_cells, return_mean=return_mean)


class ROIExpression_Image_lin(torch.nn.Module):
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
        super().__init()
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
        data.x = self.image.forward(data.x)
        return self.graph.forward(data, return_cells=return_cells, return_mean=return_mean)
        