import torch
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils.convert import to_networkx


edge_list = torch.tensor([
    [0, 0, 0, 1, 2, 2, 3, 3],
    [1, 2, 3, 0, 0, 3, 2, 0]
], dtype=torch.long)

node_features = torch.tensor([
    [-8, 1, 5, 8, 2, -3],
    [-1, 0, 2, -3, 0, 1],
    [1, -1, 0, -1, 2, 1],
    [0, 1, 4, -2, 3, 4],
], dtype=torch.long)

edge_weight = torch.tensor([
    [35.], [48.], [12.],
    [10.], [70.], [5.],
    [15.], [8.]
], dtype=torch.long)

data = Data(x=node_features, edge_index=edge_list, edge_attr=edge_weight)

print(f"Number of Nodes: {data.num_nodes}")
print(f"Number of Edges: {data.num_edges}")
print(f"Number of Features per Node (Length of Feature Vector): {data.num_node_features}")
print(f"Number of weights per Edge (Edge-Features): {data.num_edge_features}")

G = to_networkx(data)
nx.draw_networkx(G)