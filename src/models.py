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

    def forward_layerwise(self, x_B0, bipartite_edges, sizes):
        """Layer-wise forward pass for GraphSAGE mini-batch training.

        Implements the per-layer aggregation described in Algorithm 2 of
        Hamilton et al. (2017) "Inductive Representation Learning on Large
        Graphs".  At layer l, SAGEConv aggregates from B_l (source set)
        to B_{l+1} (target set), where B_{l+1} always occupies the FIRST
        ``sizes[l+1]`` positions in B_l.

        Parameters
        ----------
        x_B0 : Tensor, shape (n_B0, in_feats)
            Node features for all nodes in B_0 (the outermost hop set).
        bipartite_edges : list[Tensor]
            Per-layer bipartite edge indices.  bipartite_edges[l] has shape
            (2, n_edges_l) with source local indices in [0, sizes[l]) and
            target local indices in [0, sizes[l+1]).
        sizes : list[int]
            sizes[l] = |B_l|, length = n_layers + 1.

        Returns
        -------
        Tensor, shape (n_seeds, n_classes)  where n_seeds = sizes[-1].
        """
        h = self.node_encoder(x_B0)
        h = F.relu(h)
        h = self.input_drop(h)

        h_last = None
        h_local = []

        for i, (ei, n_tgt) in enumerate(zip(bipartite_edges, sizes[1:])):
            n_src = h.shape[0]  # |B_i|

            # Edge drop for regularisation (same convention as forward()).
            if self.training and self.edge_drop > 0 and ei.shape[1] > 0:
                mask = torch.rand(ei.shape[1], device=ei.device) >= self.edge_drop
                ei = ei[:, mask]

            # SAGEConv bipartite call: x = (source_feats, target_feats).
            # B_{i+1} occupies the first n_tgt rows of B_i, so target
            # features are simply h[:n_tgt].
            # Corresponds to the AGGREGATE + COMBINE steps in
            # Hamilton et al. 2017 Algorithm 2, line 4.
            h = self.convs[i]((h, h[:n_tgt]), ei, size=(n_src, n_tgt))

            # Skip connection from the previous layer's raw conv output
            # (mirrors the residual logic in forward()).
            if h_last is not None:
                h = h + h_last[:h.shape[0], :]
            h_last = h

            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
            h_local.append(h)

        if self.jk:
            # Jump Knowledge: sum layer contributions; trim each to seed size.
            n_seeds = sizes[-1]
            h_local = [t[:n_seeds, :] for t in h_local]
            h = torch.sum(torch.stack(h_local), dim=0)

        return self.pred_linear(h)

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
