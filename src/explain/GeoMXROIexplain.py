from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.loader import DataLoader
from src.data.GeoMXData import GeoMXDataset
from src.models.GraphModel import ROIExpression
from src.utils.load import load
from typing import Any, Optional
from math import sqrt
import torch
from torch import Tensor
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = GeoMXDataset(root_dir='data/',
                        raw_subset_dir='TMA1_preprocessed',)
dataset.mode = 'explain'
model = ROIExpression(layers=1,
                        num_node_features=256,
                        num_edge_features=1,
                        num_embed_features=128,
                        embed_dropout=0.3,
                        conv_dropout=0.3,
                        num_out_features=dataset.get(0).y.shape[0],
                        heads=1).to(device, dtype=float)
model.load_state_dict(load('out/ROI.pt', 'model'))
model.eval()

explainer = Explainer(model=model,
                      algorithm=GNNExplainer(epochs=200),
                      explanation_type='model',
                      model_config=dict(mode='regression',
                                        task_level='graph',
                                        return_type='raw'),
                      node_mask_type='object',
                      edge_mask_type='object')

loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

def _visualize_graph_via_networkx(
    edge_index: Tensor,
    edge_weight: Tensor,
    pos: Tensor,
    path: Optional[str] = None,
) -> Any:
    import matplotlib.pyplot as plt
    import networkx as nx

    pos = pos.numpy()
    g = nx.DiGraph()
    node_size = 1

    for node in edge_index.view(-1).unique().tolist():
        g.add_node(node)

    for (src, dst), w in zip(edge_index.tolist(), edge_weight.tolist()):
        g.add_edge(src, dst, alpha=w)

    ax = plt.gca()
    new_pos = {}
    for i in list(range(pos.shape[0])):
        new_pos[i] = pos[i]
    new_pos = pos
    for src, dst, data in g.edges(data=True):
        ax.annotate(
            '',
            xy=pos[src],
            xytext=pos[dst],
            arrowprops=dict(
                arrowstyle="->",
                alpha=data['alpha'],
                shrinkA=sqrt(node_size) / 2.0,
                shrinkB=sqrt(node_size) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ),
        )

    nodes = nx.draw_networkx_nodes(g, pos, node_size=node_size,
                                   node_color='white', margins=0.1, )
    nodes.set_edgecolor('black')
    nx.draw_networkx_labels(g, pos, font_size=1)

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()

with tqdm(loader, total=len(loader), desc=f"Explaining haha") as train_loader:
    for idx, batch in enumerate(loader):
        if idx > 0:
            pass
        explanation = explainer(x=batch.x, edge_index=batch.edge_index, target=batch.y, edge_attr=batch.edge_attr, batch=batch.batch)
        print(f'Generated explanations in {explanation.available_explanations}')

        # path = 'feature_importance.png'
        # explanation.visualize_feature_importance(path, top_k=None)
        #print(f"Feature importance plot has been saved to '{path}'")

        path = 'subgraph.pdf'
        _visualize_graph_via_networkx(explanation.edge_index, explanation.edge_mask, batch.pos, path)
        #explanation.visualize_graph(path)
        print(f"Subgraph visualization plot has been saved to '{path}'")