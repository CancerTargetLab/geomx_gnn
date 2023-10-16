import torch
import torch.nn as nn


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
    def __init__(self, channels, embed=256, contrast=124, mode='train', resnet='101'):
        super().__init__()
        self.mode = mode
        if resnet == '101':
            from torchvision.models import resnet101, ResNet101_Weights
            self.res = resnet101(weights=ResNet101_Weights.DEFAULT)
        elif resnet == '50':
            from torchvision.models import resnet50, ResNet50_Weights
            self.res = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif resnet == '34':
            from torchvision.models import resnet34, ResNet34_Weights
            self.res = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif resnet == '18':
            from torchvision.models import resnet18, ResNet18_Weights
            self.res = resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in self.res.parameters():
             param.requires_grad = False
        self.res.conv1 = torch.nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=1, padding='same', bias=False)
        self.embed = nn.Sequential(
            nn.BatchNorm1d(self.res.fc.in_features, ),
            nn.ReLU(),
            nn.Linear(self.res.fc.in_features, embed)
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
