import random

import numpy as np
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.utils import scatter

from .models import GNN_PyG


def set_seed(seed_val=0):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(dataset_name):
    dataset_obj = PygNodePropPredDataset(name=dataset_name, root='/mnt/SGCN/dataset')
    evaluator   = Evaluator(name=dataset_name)
    split_idx   = dataset_obj.get_idx_split()
    train_idx   = split_idx['train']
    val_idx     = split_idx['valid']
    test_idx    = split_idx['test']
    data        = dataset_obj[0]
    return data, train_idx, val_idx, test_idx, evaluator


def preprocess(data, train_idx, n_classes):
    # Aggregate edge features to node features via sum of incoming edges
    x = scatter(data.edge_attr, data.edge_index[1], dim=0,
                dim_size=data.num_nodes, reduce='sum')
    data.x = x

    # Training labels as additional input features (others stay zero)
    data.train_labels_onehot = torch.zeros(data.num_nodes, n_classes)
    data.train_labels_onehot[train_idx, data.y[train_idx, 0].long()] = 1
    return data


def gen_model(n_node_feats, n_classes, use_labels, n_layers, n_hidden,
              dropout, input_drop, edge_drop, mpnn, jk):
    n_feats = (n_node_feats + n_classes) if use_labels else n_node_feats
    return GNN_PyG(
        n_feats,
        n_classes,
        n_layers=n_layers,
        n_hidden=n_hidden,
        activation=F.relu,
        dropout=dropout,
        input_drop=input_drop,
        edge_drop=edge_drop,
        mpnn=mpnn,
        jk=jk,
    )


def add_labels(x, train_labels_onehot, idx, n_classes, device):
    """Concatenate one-hot training labels to node features for the given indices."""
    labels_onehot = torch.zeros([x.shape[0], n_classes], device=device)
    labels_onehot[idx] = train_labels_onehot[idx].to(device)
    return torch.cat([x, labels_onehot], dim=-1)
