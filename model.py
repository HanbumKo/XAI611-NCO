import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GINEConv


class SimpleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_mp_layers):
        super(SimpleGCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.mp_layers =  nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim, aggr="add") for _ in range(n_mp_layers)
        ])

    def forward(self, torch_graph):
        node_feat = torch_graph.x # (# nodes, 3)
        edge_index = torch_graph.edge_index # (2, # edges)
        edge_attr = torch_graph.edge_attr # (# edges, )

        y_hat = self.fc1(node_feat) # (# nodes, 3) -> (# nodes, 32)
        for mp_layer in self.mp_layers:
            y_hat = mp_layer(y_hat, edge_index, edge_weight=edge_attr) # (# nodes, 32) -> (# nodes, 32)
        
        y_hat = self.fc2(y_hat) # (# nodes, 32) -> (# nodes, 1)
        y_hat = torch.sigmoid(y_hat)
        return y_hat