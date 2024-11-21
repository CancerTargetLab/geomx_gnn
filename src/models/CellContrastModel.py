import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    Implemented in torch following the example of 
    https://github.com/google-research/simclr/blob/master/model_util.py#L141
    MLP
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
    """
    A PyTorch module for contrastive learning using a ResNet backbone.

    Attributes:
    mode (str): The mode of the model, either 'train' or 'eval'.
    embed_size (int): The size of the embedding layer.
    contrast_size (int): The size of the contrastive layer.
    resnet_model (str): The ResNet model variant to use ('101', '50', '34', '18').
    res (torchvision.models): The ResNet model.
    embed (nn.Sequential): The embedding layer.
    head (ProjectionHead): The projection head for contrastive learning.
    """
    def __init__(self, channels, embed=256, contrast=124, mode='train', resnet='101'):
        """
        Initializes the ContrastiveLearning model with the given parameters.

        Parameters:
        channels (int): Number of input channels.
        embed (int): Size of the embedding layer. Default is 256.
        contrast (int): Size of the contrastive layer. Default is 124.
        mode (str): Mode of the model, either 'train' or 'eval'. Default is 'train'.
        resnet (str): ResNet model variant to use ('101', '50', '34', '18'). Default is '101'.
        """
         
        super().__init__()
        self.mode = mode
        self.embed_size = embed
        self.contrast_size = contrast
        self.resnet_model = resnet
        if resnet == '101':
            from torchvision.models import resnet101
            self.res = resnet101()
        elif resnet == '50':
            from torchvision.models import resnet50
            self.res = resnet50()
        elif resnet == '34':
            from torchvision.models import resnet34
            self.res = resnet34()
        elif resnet == '18':
            from torchvision.models import resnet18
            self.res = resnet18()
        self.res.conv1 = torch.nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=1, padding='same', bias=False)
        self.res.fc = nn.Sequential(
            nn.Linear(self.res.fc.in_features, self.res.fc.in_features),
            nn.BatchNorm1d(self.res.fc.in_features),
            nn.ReLU(),
            nn.Linear(self.res.fc.in_features, embed),#bias False,
            )
        self.head = nn.Sequential(
            #nn.BatchNorm1d(embed),
            ProjectionHead(embed, contrast)
            )

    def forward(self, data):
        """
        Forward pass of the model.

        Parameters:
        data (torch.Tensor): The input data tensor.

        Returns:
        torch.Tensor: The output of the model. If in 'train' mode, returns the projection head output. Otherwise, returns the embeddings.
        """

        data = self.res.conv1(data)
        data = self.res.bn1(data)
        data = self.res.relu(data)
        data = self.res.layer1(data)
        data = self.res.layer2(data)
        data = self.res.layer3(data)
        data = self.res.layer4(data)
        data = self.res.avgpool(data)
        data = data.squeeze()
        if len(data.shape) == 1:
            data = data.unsqueeze(0)
        embed = self.res.fc(data)
        if self.mode == 'train':
            out = self.head(embed)
            return out
        else:
            return embed
