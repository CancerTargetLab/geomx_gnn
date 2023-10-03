import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Sequential
import torch_geometric


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
                        (torch.nn.LeakyReLU(inplace=True))
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
                        (torch.nn.LeakyReLU(inplace=True)),
                        (torch.nn.Linear(heads*num_embed_features, num_embed_features)),
                        (torch.nn.LayerNorm(num_embed_features)),
                        (torch.nn.Dropout(p=embed_dropout, inplace=True)),
                        (torch.nn.LeakyReLU(inplace=True))
                        ])
                    layers.append(gat_layer)

            return layers


class GraphLearning(torch.nn.Module):
    def __init__(self,
                 layers=1,
                 num_node_features=100, 
                 num_edge_features=1,
                 num_embed_features=10,
                 num_out_features=10,
                 heads=1,
                 embed_dropout=0.1,
                 conv_dropout=0.1,
                 skip_dropout=0.1):
        super().__init__()

        self.heads = heads

        self.node_embed = torch.nn.Sequential(
            torch.nn.Linear(num_node_features, num_embed_features),
            torch.nn.LayerNorm(num_embed_features),
            torch.nn.Dropout(p=embed_dropout, inplace=True),
            torch.nn.LeakyReLU(inplace=True)
            )

        self.convs = create_GATv2_conv(num_layers=layers,
                                       num_embed_features=num_embed_features,
                                       num_edge_features=num_edge_features,
                                       heads=heads,
                                       conv_dropout=conv_dropout,
                                       embed_dropout=embed_dropout,
                                       fill_value=0.0)
        
        self.conv_skip = torch.nn.Sequential(
            torch.nn.Linear((layers+1)*num_embed_features, num_out_features),
            torch.nn.LayerNorm(num_embed_features),
            torch.nn.Dropout(p=skip_dropout, inplace=True),
            torch.nn.LeakyReLU(inplace=True)
            )
        
        self.node_embed.apply(init_weights)
        self.convs.apply(init_weights)
        self.conv_skip.apply(init_weights)


    def forward(self, data):
        x, edge_index, edge_feature = data.x, data.edge_index, data.edge_weight

        h_i = self.node_embed(x)
        h = h_i.clone()

        for conv in list(range(len(self.convs))):
            h_i = self.convs[conv](h_i, edge_index, edge_attr=edge_feature)

            h = torch.concat((h, h_i), dim=1)
        
        h = self.conv_skip(h)

        return h


class AutoEncodeEmbedding(torch.nn.Module):
    def __init__(self, embed_size=100):
        super().__init__()

        self.encode = torch.nn.Sequential(
             torch.nn.Linear(2048, 1024),
             torch.nn.BatchNorm1d(1024),
             torch.nn.ReLU(),
             torch.nn.Linear(1024, 512),
             torch.nn.BatchNorm1d(512),
             torch.nn.ReLU(),
            #  torch.nn.Linear(512, 256),
            #  torch.nn.BatchNorm1d(256),
            #  torch.nn.ReLU(),
            #  torch.nn.Linear(256, embed_size),
            #  torch.nn.BatchNorm1d(embed_size),
            #  torch.nn.ReLU(),
        )

        self.decode = torch.nn.Sequential(
            #  torch.nn.Linear(embed_size, 256),
            #  torch.nn.BatchNorm1d(256),
            #  torch.nn.ReLU(),
            #  torch.nn.Linear(256, 512),
            #  torch.nn.BatchNorm1d(512),
            #  torch.nn.ReLU(),
             torch.nn.Linear(512, 1024),
             torch.nn.BatchNorm1d(1024),
             torch.nn.ReLU(),
             torch.nn.Linear(1024, 2048),
             torch.nn.BatchNorm1d(2048),
             torch.nn.ReLU(),
        )

    def forward(self, data, return_encoding=False):
         enc = self.encode(data)
         dec = self.decode(enc)

         if return_encoding:
              return dec, enc
         else:
              return dec



class ProjectionHead(nn.Module):
    """
    Implemented in torch following the example of 
    https://github.com/google-research/simclr/blob/master/model_util.py#L141
    """
    def __init__(self, input_dim, output_dim, num_layers=2, use_bn=True, use_relu=True):
        super(ProjectionHead, self).__init__()
        self.layers = nn.ModuleList()
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

            linear_layer = nn.Linear(dim, next_dim, bias=bias_relu)
            self.layers.append(linear_layer)
            if use_bn and bias_relu:
                bn_layer = nn.BatchNorm1d(next_dim)
                self.layers.append(bn_layer)
            
            if bias_relu and use_relu:
                relu_layer = nn.ReLU()
                self.layers.append(relu_layer)

            dim = next_dim

    def forward(self, hiddens):
        for layer in self.layers:
            hiddens = layer(hiddens)
        
        return hiddens


class ContrastiveLearning(torch.nn.Module):
    def __init__(self, channels, embed=256, contrast=124, mode='train'):
        super().__init__()
        self.mode = mode
        from torchvision.models import resnet101, ResNet101_Weights
        self.res = resnet101(weights=ResNet101_Weights.DEFAULT)
        for param in self.res.parameters():
             param.requires_grad = False
        self.res.conv1 = torch.nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        self.embed = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, embed)
            )
        self.head = ProjectionHead(embed, contrast)

    def forward(self, data):
        data = self.res.conv1(data)
        data = self.res.bn1(data)
        data = self.res.relu(data)
        data = self.res.layer1(data)
        data = self.res.layer2(data)
        data = self.res.layer3(data)
        data = self.res.layer4(data)
        data = self.res.avgpool(data)
        embed = self.embed(data.squeeze())    #TODO: make this a projection head as well?
        if self.mode == 'train':
            out = self.head(embed)
            return out
        else:
            return embed
