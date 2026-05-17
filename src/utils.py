import random

import numpy as np
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.utils import scatter

from .models import GNN_PyG


def apply_ood_perturbation(data, node_ratio, rewire_ratio, seed):
    """Apply OOD graph perturbation by Bernoulli node sampling + edge rewiring."""
    if not (0.0 <= node_ratio <= 1.0):
        raise ValueError(f'node_ratio must be in [0, 1], got {node_ratio}')
    if not (0.0 <= rewire_ratio <= 1.0):
        raise ValueError(f'rewire_ratio must be in [0, 1], got {rewire_ratio}')

    edge_index = data.edge_index
    device = edge_index.device
    num_nodes = int(data.num_nodes)
    rng = random.Random(int(seed))

    selected_nodes_list = [i for i in range(num_nodes) if rng.random() < node_ratio]
    selected_nodes = torch.tensor(selected_nodes_list, dtype=torch.long)

    edge_index_cpu = edge_index.cpu()
    edge_attr_cpu = data.edge_attr.cpu() if getattr(data, 'edge_attr', None) is not None else None
    edge_attr_dim = int(edge_attr_cpu.shape[1]) if edge_attr_cpu is not None else 0

    adjacency = [set() for _ in range(num_nodes)]
    undirected_edges = set()
    self_loops = set()
    edge_attr_map = {}

    for e in range(edge_index_cpu.shape[1]):
        u = int(edge_index_cpu[0, e])
        v = int(edge_index_cpu[1, e])
        if u == v:
            self_loops.add(u)
            continue
        a, b = (u, v) if u < v else (v, u)
        undirected_edges.add((a, b))
        adjacency[a].add(b)
        adjacency[b].add(a)
        if edge_attr_cpu is not None and (a, b) not in edge_attr_map:
            edge_attr_map[(a, b)] = edge_attr_cpu[e].clone()

    degree_before = {n: len(adjacency[n]) for n in selected_nodes_list}
    rewired_per_node = {}
    total_rewired = 0

    for node in selected_nodes_list:
        neighbors = list(adjacency[node])
        degree = len(neighbors)
        k = int(rewire_ratio * degree)
        if k <= 0 or degree == 0:
            rewired_per_node[node] = 0
            continue

        k = min(k, degree)
        removed_edges = []
        for nbr in rng.sample(neighbors, k):
            if nbr not in adjacency[node]:
                continue
            adjacency[node].remove(nbr)
            adjacency[nbr].remove(node)
            a, b = (node, nbr) if node < nbr else (nbr, node)
            undirected_edges.discard((a, b))
            attr = edge_attr_map.pop((a, b), None) if edge_attr_cpu is not None else None
            removed_edges.append((nbr, attr))

        added = 0
        max_attempts = max(100, 20 * k)
        attempts = 0
        while added < len(removed_edges) and attempts < max_attempts:
            attempts += 1
            cand = rng.randrange(num_nodes)
            if cand == node or cand in adjacency[node]:
                continue

            adjacency[node].add(cand)
            adjacency[cand].add(node)
            a, b = (node, cand) if node < cand else (cand, node)
            undirected_edges.add((a, b))
            if edge_attr_cpu is not None:
                src_attr = removed_edges[added][1]
                if src_attr is None:
                    src_attr = torch.zeros(edge_attr_dim, dtype=edge_attr_cpu.dtype)
                edge_attr_map[(a, b)] = src_attr.clone()
            added += 1

        # If unable to add enough fresh edges, restore leftovers to keep degree stable.
        if added < len(removed_edges):
            for idx in range(added, len(removed_edges)):
                nbr, attr = removed_edges[idx]
                if nbr == node or nbr in adjacency[node]:
                    continue
                adjacency[node].add(nbr)
                adjacency[nbr].add(node)
                a, b = (node, nbr) if node < nbr else (nbr, node)
                undirected_edges.add((a, b))
                if edge_attr_cpu is not None:
                    if attr is None:
                        attr = torch.zeros(edge_attr_dim, dtype=edge_attr_cpu.dtype)
                    edge_attr_map[(a, b)] = attr.clone()

        rewired_per_node[node] = added
        total_rewired += added

    degree_after = {n: len(adjacency[n]) for n in selected_nodes_list}
    degree_change = {n: degree_after[n] - degree_before[n] for n in selected_nodes_list}

    directed_edges = []
    directed_attrs = []
    for u, v in sorted(undirected_edges):
        directed_edges.append([u, v])
        directed_edges.append([v, u])
        if edge_attr_cpu is not None:
            attr = edge_attr_map.get((u, v))
            if attr is None:
                attr = torch.zeros(edge_attr_dim, dtype=edge_attr_cpu.dtype)
            directed_attrs.append(attr.clone())
            directed_attrs.append(attr.clone())

    for u in sorted(self_loops):
        directed_edges.append([u, u])
        if edge_attr_cpu is not None:
            directed_attrs.append(torch.zeros(edge_attr_dim, dtype=edge_attr_cpu.dtype))

    edge_index_ood = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()

    data_ood = data.clone()
    data_ood.edge_index = edge_index_ood.to(device)
    if edge_attr_cpu is not None:
        data_ood.edge_attr = torch.stack(directed_attrs, dim=0).to(device)

    before_vals = list(degree_before.values())
    after_vals = list(degree_after.values())
    mean_before = float(np.mean(before_vals)) if before_vals else 0.0
    mean_after = float(np.mean(after_vals)) if after_vals else 0.0

    stats = {
        'selected_nodes': selected_nodes,
        'num_selected_nodes': int(len(selected_nodes_list)),
        'selected_ratio': (len(selected_nodes_list) / num_nodes) if num_nodes > 0 else 0.0,
        'rewired_edges': int(total_rewired),
        'max_rewired_per_node': int(max(rewired_per_node.values(), default=0)),
        'mean_degree_before': mean_before,
        'mean_degree_after': mean_after,
        'degree_change': degree_change,
    }
    return data_ood, stats


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
