from skimage import io
import numpy as np
import pandas as pd
import squidpy as sq
import scanpy as sc
from anndata import AnnData
from src.models.GraphModel import ROIExpression
from src.data.GeoMXData import GeoMXDataset
import matplotlib.pyplot as plt
import torchvision
import torch

df = pd.read_csv("data/raw/measurements.csv", header=0, sep=",")
df = df[["Image", "Centroid.X.px", "Centroid.Y.px"]] #'Class'
df = df[df["Image"] == "023-2B27.tiff"]
df = df.drop("Image", axis=1)
mask = ~df.duplicated(subset=['Centroid.X.px', 'Centroid.Y.px'], keep=False) | ~df.duplicated(subset=['Centroid.X.px', 'Centroid.Y.px'], keep='first')
df = df[mask]
# Replace commas with dots and convert to float
# df["Centroid X px"] = df["Centroid X px"].str.replace(',', '.').astype(float)
# df["Centroid Y px"] = df["Centroid Y px"].str.replace(',', '.').astype(float)


img = io.imread('data/023-2B27.tiff', plugin='tifffile')

image = np.expand_dims(img, axis=3)
test=sq.im.ImageContainer(image, dims=("y", "x", "z", "channels"))

seg = sq.im.ImageContainer(img, layer="img1", dims=("y", "x", "channels"))

x = df["Centroid.X.px"].round().astype(int).values
y = df["Centroid.Y.px"].round().astype(int).values

# for cell in list(range(x.shape[0])):
#     print(cell)
#     delta_x1 = 10 if x[cell] >= 10 else x[cell]
#     delta_y1 = 10 if y[cell] >= 10 else y[cell]
#     delta_x2 = 10 if img.shape[1]-x[cell] >= 10 else img.shape[1]-x[cell]
#     delta_y2 = 10 if img.shape[0]-y[cell] >= 10 else img.shape[0]-y[cell]
#     cell_img = img[y[cell]-delta_y1:y[cell]+delta_y2,x[cell]-delta_x1:x[cell]+delta_x2,:]

#     io.imsave(f"data/raw/cell_images/cell_{cell}.tiff", cell_img)

# m = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
# img1= cell_img.astype("float64")
# img1 = (img1/np.max(img))*255
# img10 = np.expand_dims(img1[:,:,0], axis=2)
# img10 = np.concatenate((img10, img10, img10), axis=2)
# img10=torchvision.transforms.ToTensor()(img10)
# img10=torch.unsqueeze(img10, dim=0)
# img10=torchvision.models.ResNet101_Weights.DEFAULT.transforms(antialias=True)(img10)
# m.avgpool(m.layer4(m.layer3(m.layer2(m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(img10)))))))))

# seg = sq.im.ImageContainer(img[:,:,0], layer="img1", dims=("y", "x"))
# seg.add_img(img[:,:,1], layer="img2", dims=("y", "x"))
# seg.add_img(img[:,:,2], layer="img3", dims=("y", "x"))

# print(seg)
# #img = sq.im.ImageContainer.concat(seg)

# seg.show("img1")

# model = ROIExpression(layers=3, num_node_features=256, num_embed_features=128, num_out_features=49).to(torch.device('cpu'), dtype=float)
# model.load_state_dict(torch.load('out/ROI.pt')['model'])
# model.eval()
# dataset = GeoMXDataset(raw_subset_dir='TMA1_preprocessed')
# dataset.mode = 'embed'
# g1 = dataset.get(0)
# counts = torch.abs(model.project(model.gnn(g1))).detach().numpy()

counts = np.random.default_rng(42).integers(0, 15, size=(df.shape[0], 100))

coordinates = np.column_stack((df["Centroid.X.px"].to_numpy(), df["Centroid.Y.px"].to_numpy()))

adata = AnnData(counts, obsm={"spatial": coordinates})

#adata.obs["CellType"] = df["Class"].values

sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True, radius=30.0)

# sq.pl.spatial_scatter(
#     adata,
#     color="CellType",
#     connectivity_key="spatial_connectivities",
#     edges_color="black",
#     shape=None,
#     edges_width=1,
#     size=50,
# )

spatial_key = "spatial"
library_id = "tissue42"
adata.uns[spatial_key] = {library_id: {}}
adata.uns[spatial_key][library_id]["images"] = {}
adata.uns[spatial_key][library_id]["images"] = {"hires": img}
adata.uns[spatial_key][library_id]["scalefactors"] = {
    "tissue_hires_scalef": 1,
    "spot_diameter_fullres": 0.5,
}

# print("normalize total")
# sc.pp.normalize_total(adata)
# print("log transform")
# sc.pp.log1p(adata)
# print("scale")
# sc.pp.scale(adata, max_value=10)

# resolution = 0.5
# print("PCA")
# sc.tl.pca(adata, svd_solver="arpack")
# print("neighbors")
# sc.pp.neighbors(adata, n_neighbors=10, n_pcs=5)
# print("UMAP")
# sc.tl.umap(adata)
# print("Leiden")
# sc.tl.leiden(adata, resolution=resolution)

cluster = np.load('out/p0_023-2B27.npy', allow_pickle=True)
adata.obs['cluster'] = cluster

sq.pl.spatial_scatter(adata, 
                      color="cluster",
                      #connectivity_key="spatial_connectivities",
                      #edges_color="grey",
                      #edges_width=1,
                      size=25,
                      img_channel=0
                    )

# print("test")