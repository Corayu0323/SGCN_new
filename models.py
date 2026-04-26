import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class GNN_PyG(nn.Module):
    """Multi-layer GNN (PyG backend) supporting sage / gcn / graphsaint / sgcn."""

    def __init__(
        self,
        node_feats,
        n_classes,
        n_layers,
        n_hidden,
        activation,
        dropout,
        input_drop,
        edge_drop,
        mpnn='gcn',
        jk=False,
    ):
        super().__init__()
        self.n_layers  = n_layers
        self.n_hidden  = n_hidden
        self.n_classes = n_classes
        self.mpnn      = mpnn
        self.jk        = jk
        self.edge_drop = edge_drop

        self.node_encoder = nn.Linear(node_feats, n_hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            if mpnn == 'sage':
                self.convs.append(SAGEConv(n_hidden, n_hidden))
            else:  # gcn / graphsaint / sgcn
                # GraphSAINT and SGCN both use standard GCN convolution; their
                # key differences are in subgraph-level sampling and aggregation
                # strategies during training (handled in train.py).
                self.convs.append(GCNConv(n_hidden, n_hidden, add_self_loops=False))

            self.norms.append(nn.BatchNorm1d(n_hidden))

        self.pred_linear = nn.Linear(n_hidden, n_classes)
        self.input_drop  = nn.Dropout(input_drop)
        self.dropout     = nn.Dropout(dropout)
        self.activation  = activation

    def forward(self, x, edge_index, edge_attr=None):
        # Random edge drop during training
        if self.training and self.edge_drop > 0 and edge_index.shape[1] > 0:
            mask       = torch.rand(edge_index.shape[1], device=edge_index.device) >= self.edge_drop
            edge_index = edge_index[:, mask]

        h = self.node_encoder(x)
        h = F.relu(h)
        h = self.input_drop(h)

        h_local = []
        h_last  = None

        for i in range(self.n_layers):
            h = self.convs[i](h, edge_index)

            if h_last is not None:
                h = h + h_last[: h.shape[0], :]
            h_last = h

            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
            h_local.append(h)

        if self.jk:
            h_local = [t[: h.shape[0], :] for t in h_local]
            h = torch.sum(torch.stack(h_local), dim=0)

        return self.pred_linear(h)
