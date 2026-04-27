import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import NeighborLoader, GraphSAINTRandomWalkSampler
from torch_geometric.utils import subgraph as pyg_subgraph

from .utils import add_labels, gen_model


def _cuda_sync(device):
    """Synchronize CUDA if available to get accurate timing boundaries."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


# ── SGCN helpers ─────────────────────────────────────────────────────────────

# Fraction of n_sample used as BFS seed nodes (1/20 = 5%)
_SGCN_SEED_RATIO = 20
# Hard upper bound on BFS hops for random_walk sampling
_SGCN_RANDOM_WALK_MAX_HOPS = 10
# Minimum training nodes to inject when a sampled subgraph contains none
_SGCN_MIN_TRAIN_NODES = 32
# Number of validation nodes sampled for the per-subgraph quality score
_SGCN_VAL_SAMPLE_SIZE = 512

def _sample_subgraph_nodes(edge_index, n_nodes, train_idx, method, n_sample,
                           subgraph_max_nodes=None, unsampled_nodes=None):
    """Return a 1-D sorted LongTensor of sampled node indices.

    All returned tensors live on the same device as *edge_index*, enabling
    fully GPU-resident sampling when called with CUDA tensors.

    Supported methods
    -----------------
    random_node  – uniformly sample *n_sample* nodes at random.
    random_edge  – sample random edges and collect their incident nodes.
    random_walk  – BFS expansion from random training-set seeds (no hop cap).
    snowball     – BFS expansion capped at 2 hops from random seeds.

    Parameters
    ----------
    subgraph_max_nodes : int or None
        When provided, overrides *n_sample* as the hard upper bound on the
        number of returned nodes.  Has priority over the ratio-based value.
    unsampled_nodes : 1-D LongTensor or None
        Nodes not yet covered in the current epoch.  When provided, each
        sampling method prioritises these nodes so that the sequence of
        subgraphs collectively covers the whole graph before revisiting
        already-sampled regions.
    """
    # All tensor ops run on the same device as edge_index (CPU or CUDA).
    device = edge_index.device

    # subgraph_max_nodes takes priority over the ratio-derived n_sample.
    if subgraph_max_nodes is not None and subgraph_max_nodes > 0:
        n_sample = subgraph_max_nodes
    n_sample = min(n_sample, n_nodes)

    # Determine whether there are meaningful unsampled nodes to prioritise.
    has_priority = (
        unsampled_nodes is not None and len(unsampled_nodes) > 0
    )

    if method == 'random_node':
        if has_priority:
            n_priority = len(unsampled_nodes)
            if n_priority >= n_sample:
                # Enough unsampled nodes to fill the quota entirely.
                perm = torch.randperm(n_priority, device=device)[:n_sample]
                return unsampled_nodes[perm].sort().values
            else:
                # Take all unsampled nodes, then fill remainder from the rest.
                remaining = n_sample - n_priority
                sampled_mask = torch.ones(n_nodes, dtype=torch.bool, device=device)
                sampled_mask[unsampled_nodes] = False
                sampled_pool = sampled_mask.nonzero(as_tuple=False).squeeze(1)
                perm = torch.randperm(len(sampled_pool), device=device)[:remaining]
                return torch.cat([unsampled_nodes,
                                  sampled_pool[perm]]).sort().values
        perm = torch.randperm(n_nodes, device=device)[:n_sample]
        return perm.sort().values

    elif method == 'random_edge':
        n_edges = edge_index.shape[1]
        if has_priority:
            # Build a boolean mask of priority nodes, then select edges that
            # touch at least one priority node first.
            prio_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
            prio_mask[unsampled_nodes] = True
            edge_has_prio = prio_mask[edge_index[0]] | prio_mask[edge_index[1]]
            prio_edges  = edge_has_prio.nonzero(as_tuple=False).squeeze(1)
            other_edges = (~edge_has_prio).nonzero(as_tuple=False).squeeze(1)

            n_prio_sample = min(n_sample * 2, len(prio_edges))
            perm_p  = torch.randperm(len(prio_edges), device=device)[:n_prio_sample]
            chosen  = edge_index[:, prio_edges[perm_p]].flatten().unique()

            if len(chosen) < n_sample and len(other_edges) > 0:
                extra_e = other_edges[
                    torch.randperm(len(other_edges), device=device)[:n_sample * 2]
                ]
                extra_n = edge_index[:, extra_e].flatten().unique()
                chosen  = torch.cat([chosen, extra_n]).unique()
        else:
            edge_perm = torch.randperm(n_edges, device=device)[:min(n_sample * 2, n_edges)]
            chosen    = edge_index[:, edge_perm].flatten().unique()

        if len(chosen) < n_sample:
            extra = torch.randperm(n_nodes, device=device)[:n_sample - len(chosen)]
            chosen = torch.cat([chosen, extra]).unique()
        return chosen[:n_sample].sort().values

    elif method in ('random_walk', 'snowball'):
        # Build the seed pool: prefer training nodes inside unsampled regions.
        if has_priority:
            prio_train = unsampled_nodes[
                torch.isin(unsampled_nodes, train_idx)
            ]
            if len(prio_train) > 0:
                seed_pool = prio_train
            else:
                # Fall back to any unsampled node as seeds.
                seed_pool = unsampled_nodes
        else:
            seed_pool = train_idx

        n_seeds   = min(max(n_sample // _SGCN_SEED_RATIO, 1), len(seed_pool))
        seed_perm = torch.randperm(len(seed_pool), device=device)[:n_seeds]
        seeds     = seed_pool[seed_perm]

        visited = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        visited[seeds] = True

        row, col  = edge_index
        max_hops  = 2 if method == 'snowball' else _SGCN_RANDOM_WALK_MAX_HOPS

        for _ in range(max_hops):
            if int(visited.sum()) >= n_sample:
                break
            mask      = visited[row]
            new_nodes = col[mask]
            visited[new_nodes] = True

        visited_nodes = visited.nonzero(as_tuple=False).squeeze(1)

        if len(visited_nodes) > n_sample:
            perm = torch.randperm(len(visited_nodes), device=device)[:n_sample]
            return visited_nodes[perm].sort().values
        elif len(visited_nodes) < n_sample:
            unvisited  = (~visited).nonzero(as_tuple=False).squeeze(1)
            extra_perm = torch.randperm(len(unvisited), device=device)[:n_sample - len(visited_nodes)]
            return torch.cat([visited_nodes, unvisited[extra_perm]]).sort().values

        return visited_nodes.sort().values

    else:
        raise ValueError(
            f"Unknown subsampling_method: {method!r}. "
            f"Choose from: 'random_node', 'random_edge', 'random_walk', 'snowball'."
        )


def train_epoch_sgcn(model, data, criterion, optimizer, device,
                     train_idx, val_idx,
                     n_subgraphs=5,
                     local_epochs=5,
                     subsampling_method='random_node',
                     subgraph_max_nodes=256,
                     max_subgraph_edges=300000,
                     subgraph_ratio=0.5,
                     truncation_ratio=0.2,
                     aggregation_method='sgcn',
                     use_labels=False, n_classes=112,
                     debug_subgraph_stats=False,
                     x_full_dev=None, y_full_dev=None,
                     edge_index_dev=None, edge_attr_dev=None,
                     min_subgraph_nodes=0,
                     min_train_nodes_in_subgraph=_SGCN_MIN_TRAIN_NODES):
    """SGCN training epoch with subgraph sampling, local multi-epoch training,
    and configurable aggregation.

    Algorithm
    ---------
    For each of *n_subgraphs* independent subgraphs:

    1. Sample a subgraph of at most *subgraph_max_nodes* nodes using the
       chosen *subsampling_method*.  *subgraph_ratio* is used as a fallback
       when *subgraph_max_nodes* is not set (i.e. <= 0).
       Nodes not yet covered in this epoch are prioritised so that the
       sequence of subgraphs collectively covers the whole graph.
    2. Enforce a hard edge-count limit of *max_subgraph_edges* by randomly
       dropping edges when the induced subgraph exceeds that threshold.
    3. Reset the model to the epoch-start parameters (every subgraph starts
       from the same *epoch_init_state* – subgraphs are fully independent).
       Clear optimizer momentum so subgraphs do not share gradient state.
    4. Run *local_epochs* gradient steps on the fixed subgraph data (no
       re-sampling between local steps).
    5. Evaluate the resulting local model on a mini-batch of validation nodes
       (via a forward pass over the subgraph augmented with those val nodes).
    6. Record the local state dict and validation loss (as quality score).

    After all subgraphs are processed:

    * Discard the bottom *truncation_ratio* fraction by validation score
      (truncation mechanism – suppresses noise-dominated subgraphs).
    * Aggregate the remaining local states according to *aggregation_method*.
    * Load the aggregated state into the model and clear stale optimizer
      momentum.

    Coverage guarantee
    ------------------
    When *n_subgraphs* <= 0 the count is derived automatically:
        n_subgraphs = ceil(n_nodes / nodes_per_subgraph)
    so that n_subgraphs × nodes_per_subgraph ≥ n_nodes (full graph coverage).
    Within each epoch an ``epoch_sampled_mask`` tracks which nodes have been
    covered; subsequent subgraphs receive the uncovered nodes as a priority
    pool and sample from them first.

    Timing
    ------
    This function records per-subgraph timings:
      subgraph_sampling_time_r : time to sample and build the subgraph.
      subgraph_train_time_r    : time to run *local_epochs* gradient steps.
      subgraph_eval_time_r     : time for the validation scoring forward pass.
      subgraph_total_time_r    = sampling + train + eval for subgraph r.

    The returned epoch time uses the **parallel-pipeline** convention:
      max_subgraph_pipeline_time = max_r(subgraph_total_time_r)
      sgcn_epoch_time_max        = max_subgraph_pipeline_time + aggregation_time

    This is NOT the serial sum of all subgraph times.  It models the
    wall-clock time of a future parallel implementation where all subgraphs
    run concurrently and the epoch completes when the slowest subgraph
    finishes, plus the final aggregation step.

    Parameters
    ----------
    n_subgraphs         : int   – number of independent subgraphs per epoch.
                                   Set <= 0 to auto-derive from coverage
                                   constraint (ceil(n_nodes / n_sample)).
    local_epochs        : int   – number of local gradient steps per subgraph
                                   (L in the paper).  Default: 5.
    subsampling_method  : str   – one of 'random_node', 'random_edge',
                                   'random_walk', 'snowball'.
    subgraph_max_nodes  : int   – hard upper bound on nodes per subgraph.
                                   Takes priority over *subgraph_ratio*.
                                   Set <= 0 to fall back to *subgraph_ratio*.
    max_subgraph_edges  : int   – hard upper bound on edges per subgraph.
                                   Excess edges are randomly dropped (with
                                   matching edge_attr rows when present).
                                   Set <= 0 to disable.
    subgraph_ratio      : float – fraction of graph nodes per subgraph; used
                                   only when *subgraph_max_nodes* <= 0.
    truncation_ratio    : float – fraction of worst-performing subgraphs
                                   to discard before aggregation.
    aggregation_method  : str   – aggregation strategy after truncation:
                                   'sgcn'     – softmax-weighted average over
                                                validation scores (default).
                                   'avg'      – uniform equal-weight average
                                                (SGCN-Avg).
                                   'weighted' – performance-based linear-
                                                normalized weighted average
                                                (SGCN-Weighted).
    debug_subgraph_stats : bool – when True, print per-subgraph shape and
                                   CUDA memory stats before each forward pass.
    min_subgraph_nodes  : int  – if > 0, pad sampled subgraph to this many
                                   nodes by drawing random graph nodes.
                                   0 (default) disables padding.
    min_train_nodes_in_subgraph : int – minimum training nodes that must be
                                   present in a subgraph.  When the subgraph
                                   contains fewer, random training nodes are
                                   added to reach this threshold.
                                   Default: _SGCN_MIN_TRAIN_NODES (32).

    Returns
    -------
    avg_loss                 : float – average training loss over valid subgraphs.
    sgcn_epoch_time_max      : float – parallel-pipeline epoch time (see above).
    total_sampling_time      : float – sum of all subgraph sampling times
                                        (kept for backward compatibility).
    extra_sgcn               : dict  – detailed timing fields:
        local_epochs, max_subgraph_pipeline_time, aggregation_time,
        subgraph_sampling_times, subgraph_train_times,
        subgraph_eval_times, subgraph_total_times.
    """
    model.train()

    # ── Ensure full-graph tensors are GPU-resident for SGCN subgraph ops.
    # When pre-built by run(), these are reused across all epochs with zero
    # extra host→device copy.  Fall back to on-demand transfer if not supplied.
    if edge_index_dev is None:
        edge_index_dev = data.edge_index.to(device)
    if x_full_dev is None:
        x_full_dev = data.x.to(device)
    if y_full_dev is None:
        y_full_dev = data.y.to(device)
    if edge_attr_dev is None and data.edge_attr is not None:
        edge_attr_dev = data.edge_attr.to(device)

    # Move split indices to device so all isin/randperm ops stay on GPU.
    train_idx_dev = train_idx.to(device)
    val_idx_dev   = val_idx.to(device)

    n_nodes       = data.num_nodes
    # subgraph_max_nodes takes priority; fall back to ratio-based size.
    if subgraph_max_nodes is not None and subgraph_max_nodes > 0:
        n_sample = subgraph_max_nodes
    else:
        n_sample = max(1, int(n_nodes * subgraph_ratio))

    # Auto-derive n_subgraphs when not explicitly set so that the product
    # n_subgraphs × n_sample >= n_nodes (full-graph coverage guarantee).
    # n_sample already reflects subgraph_max_nodes (set above), so this
    # formula uses the actual per-subgraph node budget.
    if n_subgraphs is None or n_subgraphs <= 0:
        n_subgraphs = math.ceil(n_nodes / n_sample)

    # Snapshot of model parameters at the start of this epoch.  Every local
    # subgraph model is initialised from this state so that aggregation is
    # well-defined and subgraphs are fully independent of each other.
    # Kept on GPU to avoid PCIe round-trips on every subgraph reset.
    epoch_init_state = {k: v.clone() for k, v in model.state_dict().items()}

    local_states  = []
    val_scores    = []
    loss_sum      = 0.0
    valid_batches = 0

    val_sample_size = min(_SGCN_VAL_SAMPLE_SIZE, len(val_idx_dev))

    # Per-subgraph timing lists (index r corresponds to subgraph r).
    subgraph_sampling_times = []
    subgraph_train_times    = []
    subgraph_eval_times     = []

    # Epoch-level coverage mask: tracks which nodes have been included in at
    # least one subgraph so far.  Unsampled nodes are fed as a priority pool
    # to each successive sampler, ensuring the sequence of subgraphs covers
    # the entire graph before revisiting already-sampled regions.
    # Kept on GPU so nonzero() and indexing stay device-local.
    epoch_sampled_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)

    for _sg in range(n_subgraphs):
        # ── 1. Sample subgraph node indices ─────────────────────────────────
        _cuda_sync(device)
        t_sample_start = time.time()

        # Pass nodes not yet visited this epoch as a priority pool.
        unsampled_nodes = epoch_sampled_mask.logical_not().nonzero(
            as_tuple=False
        ).squeeze(1)
        if len(unsampled_nodes) == 0:
            unsampled_nodes = None

        node_idx  = _sample_subgraph_nodes(
            edge_index_dev, n_nodes, train_idx_dev, subsampling_method, n_sample,
            subgraph_max_nodes=subgraph_max_nodes,
            unsampled_nodes=unsampled_nodes,
        )

        # Optional: pad subgraph to at least min_subgraph_nodes total nodes.
        if min_subgraph_nodes > 0 and len(node_idx) < min_subgraph_nodes:
            target_nodes   = min(min_subgraph_nodes, n_nodes)
            n_extra        = target_nodes - len(node_idx)
            all_nodes      = torch.arange(n_nodes, device=device)
            candidate_mask = torch.ones(n_nodes, dtype=torch.bool, device=device)
            candidate_mask[node_idx] = False
            candidates  = all_nodes[candidate_mask]
            if len(candidates) > 0:
                perm     = torch.randperm(len(candidates), device=device)[:n_extra]
                node_idx = torch.cat([node_idx, candidates[perm]]).unique().sort().values

        # Guarantee at least min_train_nodes_in_subgraph training nodes are included.
        train_in_sub = torch.isin(node_idx, train_idx_dev).sum().item()
        if train_in_sub < min_train_nodes_in_subgraph:
            n_need   = min_train_nodes_in_subgraph - train_in_sub
            extra    = train_idx_dev[
                torch.randperm(len(train_idx_dev), device=device)[:min(n_need, len(train_idx_dev))]
            ]
            node_idx = torch.cat([node_idx, extra]).unique().sort().values

        # Mark these nodes as covered for the remainder of this epoch so that
        # subsequent subgraphs are biased toward the still-uncovered region.
        epoch_sampled_mask[node_idx] = True

        # ── 2. Build induced subgraph on GPU ─────────────────────────────────
        # Constructing the subgraph directly on GPU avoids the CPU pyg_subgraph
        # round-trip (host↔device copy) that was the main CPU bottleneck.
        in_subgraph_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        in_subgraph_mask[node_idx] = True
        src = edge_index_dev[0]
        dst = edge_index_dev[1]
        edge_keep = in_subgraph_mask[src] & in_subgraph_mask[dst]
        del in_subgraph_mask
        edge_index_sub_global = edge_index_dev[:, edge_keep]
        edge_attr_sub = edge_attr_dev[edge_keep] if edge_attr_dev is not None else None

        # ── 2a. Enforce hard edge-count cap ─────────────────────────────────
        if max_subgraph_edges is not None and max_subgraph_edges > 0:
            n_edges_sub = edge_index_sub_global.size(1)
            if n_edges_sub > max_subgraph_edges:
                perm           = torch.randperm(n_edges_sub, device=device)[:max_subgraph_edges]
                edge_index_sub_global = edge_index_sub_global[:, perm]
                if edge_attr_sub is not None:
                    edge_attr_sub = edge_attr_sub[perm]

        # Relabel global node ids to contiguous local ids (all on GPU).
        global_to_local = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
        global_to_local[node_idx] = torch.arange(len(node_idx), device=device)
        edge_index_sub = global_to_local[edge_index_sub_global]
        del global_to_local
        del edge_index_sub_global

        x_sub  = x_full_dev[node_idx]
        y_sub  = y_full_dev[node_idx]
        ei_sub = edge_index_sub
        ea_sub = edge_attr_sub

        _cuda_sync(device)
        subgraph_sampling_times.append(time.time() - t_sample_start)

        train_mask = torch.isin(node_idx, train_idx_dev)
        if not train_mask.any():
            del x_sub, y_sub, ei_sub, ea_sub
            # Record zero train/eval times; sampling time is preserved.
            subgraph_train_times.append(0.0)
            subgraph_eval_times.append(0.0)
            continue

        if use_labels:
            non_train_local = torch.where(~train_mask)[0]
            x_sub = add_labels(
                # non_train_local is a GPU tensor; move to CPU to index the
                # CPU-resident train_labels_onehot without a device mismatch.
                x_sub, data.train_labels_onehot, non_train_local.cpu(), n_classes, device
            )

        # ── Debug: print subgraph stats before forward pass ─────────────────
        if debug_subgraph_stats:
            print(
                f'[SGCN debug] subgraph {_sg}: '
                f'num_sub_nodes={x_sub.shape[0]}, '
                f'num_sub_edges={ei_sub.shape[1]}, '
                f'x_sub.shape={tuple(x_sub.shape)}, '
                f'ei_sub.shape={tuple(ei_sub.shape)}'
                + (f', ea_sub.shape={tuple(ea_sub.shape)}' if ea_sub is not None else '')
                + (
                    f', cuda_allocated={torch.cuda.memory_allocated(device)}, '
                    f'cuda_reserved={torch.cuda.memory_reserved(device)}'
                    if device.type == 'cuda' else ''
                )
            )

        # ── 3. Reset to epoch-start state; clear optimizer momentum ─────────
        # Each subgraph starts from the same epoch_init_state so that
        # subgraphs are fully independent and their trained states are
        # comparable for aggregation.  Clearing the optimizer state ensures
        # accumulated momentum from previous subgraphs does not bleed through.
        # epoch_init_state is already on GPU, so no .to(device) transfer needed.
        model.load_state_dict(epoch_init_state)
        optimizer.state.clear()
        model.train()

        # ── 4. Train L local epochs on the fixed subgraph data ──────────────
        # All local_epochs steps use the same subgraph – no re-sampling.
        _cuda_sync(device)
        t_train_start  = time.time()
        # train_mask is already on device (computed from GPU node_idx/train_idx).
        train_mask_dev = train_mask
        last_loss      = 0.0

        for _le in range(local_epochs):
            pred = model(x_sub, ei_sub, ea_sub)
            loss = criterion(pred[train_mask_dev], y_sub[train_mask_dev].float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_loss = loss.item()
            del pred, loss

        _cuda_sync(device)
        subgraph_train_times.append(time.time() - t_train_start)

        loss_sum      += last_loss
        valid_batches += 1

        # Free training tensors before the validation forward pass.
        del train_mask_dev
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # ── 5. Quick validation score ────────────────────────────────────────
        _cuda_sync(device)
        t_eval_start = time.time()

        model.eval()
        with torch.no_grad():
            val_sample  = val_idx_dev[
                torch.randperm(len(val_idx_dev), device=device)[:val_sample_size]
            ]
            # Augment subgraph with val nodes so GCN can aggregate their
            # neighborhood context without leaking their labels.
            eval_node_idx = torch.cat([node_idx, val_sample]).unique()
            eval_node_idx = eval_node_idx.sort().values

            # Build eval subgraph on GPU – same approach as training subgraph.
            in_eval_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
            in_eval_mask[eval_node_idx] = True
            edge_keep_eval = in_eval_mask[edge_index_dev[0]] & in_eval_mask[edge_index_dev[1]]
            del in_eval_mask
            ei_eval_global = edge_index_dev[:, edge_keep_eval]
            ea_eval = edge_attr_dev[edge_keep_eval] if edge_attr_dev is not None else None

            # Apply the same edge cap to the eval subgraph.
            if max_subgraph_edges is not None and max_subgraph_edges > 0:
                n_edges_eval = ei_eval_global.size(1)
                if n_edges_eval > max_subgraph_edges:
                    perm_eval      = torch.randperm(n_edges_eval, device=device)[:max_subgraph_edges]
                    ei_eval_global = ei_eval_global[:, perm_eval]
                    if ea_eval is not None:
                        ea_eval = ea_eval[perm_eval]

            # Relabel to contiguous local ids (GPU).
            g2l_eval = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
            g2l_eval[eval_node_idx] = torch.arange(len(eval_node_idx), device=device)
            ei_eval = g2l_eval[ei_eval_global]
            del g2l_eval
            del ei_eval_global

            x_eval  = x_full_dev[eval_node_idx]
            y_eval  = y_full_dev[eval_node_idx]

            if use_labels:
                eval_train_mask = torch.isin(eval_node_idx, train_idx_dev)
                non_train_eval  = torch.where(~eval_train_mask)[0]
                x_eval = add_labels(
                    x_eval, data.train_labels_onehot,
                    non_train_eval.cpu(), n_classes, device
                )

            pred_eval      = model(x_eval, ei_eval, ea_eval)
            # val_sample and eval_node_idx are both on device; no .to(device) needed.
            val_local_mask = torch.isin(eval_node_idx, val_sample)
            val_loss = criterion(
                pred_eval[val_local_mask], y_eval[val_local_mask].float()
            )
            val_score = -val_loss.item()   # higher is better

            del pred_eval, val_loss, val_local_mask
            del x_eval, y_eval, ei_eval, ea_eval

        _cuda_sync(device)
        subgraph_eval_times.append(time.time() - t_eval_start)

        # Save local state dict on GPU; aggregation will run fully on device.
        local_states.append({k: v.clone() for k, v in model.state_dict().items()})
        val_scores.append(val_score)

        # Release per-subgraph GPU tensors.
        del x_sub, y_sub, ei_sub, ea_sub, train_mask
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ── Compute per-subgraph total times ─────────────────────────────────────
    # Pad lists to the same length in case some subgraphs were skipped.
    while len(subgraph_train_times) < len(subgraph_sampling_times):
        subgraph_train_times.append(0.0)
    while len(subgraph_eval_times) < len(subgraph_sampling_times):
        subgraph_eval_times.append(0.0)
    subgraph_total_times = [
        s + t + e
        for s, t, e in zip(
            subgraph_sampling_times, subgraph_train_times, subgraph_eval_times
        )
    ]

    # ── Fallback if no valid subgraph was processed ──────────────────────────
    if not local_states:
        # epoch_init_state is already on GPU; load directly.
        model.load_state_dict(epoch_init_state)
        extra_sgcn = {
            'local_epochs':               local_epochs,
            'max_subgraph_pipeline_time': 0.0,
            'aggregation_time':           0.0,
            'subgraph_sampling_times':    subgraph_sampling_times,
            'subgraph_train_times':       subgraph_train_times,
            'subgraph_eval_times':        subgraph_eval_times,
            'subgraph_total_times':       subgraph_total_times,
        }
        total_sampling_time = sum(subgraph_sampling_times)
        return 0.0, 0.0, total_sampling_time, extra_sgcn

    # ── 6. Truncation: keep top (1 − truncation_ratio) subgraphs ────────────
    _cuda_sync(device)
    t_agg_start = time.time()

    n_keep     = max(1, int(len(local_states) * (1.0 - truncation_ratio)))
    sorted_idx = sorted(range(len(val_scores)), key=lambda i: val_scores[i],
                        reverse=True)
    kept_idx   = sorted_idx[:n_keep]

    # ── 7. Aggregate local states according to aggregation_method ───────────
    # kept_scores and weights are created directly on device so the entire
    # stacking/weighted-sum stays on GPU without extra PCIe transfers.
    kept_scores = torch.tensor([val_scores[i] for i in kept_idx],
                               dtype=torch.float, device=device)

    if aggregation_method == 'avg':
        # SGCN-Avg: uniform equal-weight average
        weights = torch.ones(len(kept_idx), dtype=torch.float, device=device) / len(kept_idx)
    elif aggregation_method == 'weighted':
        # SGCN-Weighted: performance-based linear normalization.
        # Shift scores so the minimum becomes a small positive value, then
        # normalize so weights sum to 1.
        shifted = kept_scores - kept_scores.min() + 1e-8
        weights = shifted / shifted.sum()
    else:
        # Default 'sgcn': softmax-weighted average over validation scores
        if aggregation_method != 'sgcn':
            raise ValueError(
                f"Unknown aggregation_method: {aggregation_method!r}. "
                f"Choose from: 'sgcn', 'avg', 'weighted'."
            )
        weights = torch.softmax(kept_scores, dim=0)

    agg_state = {}
    for key in epoch_init_state:
        # local_states entries are GPU tensors; stacking stays on device.
        stacked = torch.stack(
            [local_states[i][key].float() for i in kept_idx], dim=0
        )
        w           = weights.view([-1] + [1] * (stacked.dim() - 1))
        agg_state[key] = (stacked * w).sum(dim=0).to(epoch_init_state[key].dtype)

    # ── 8. Load aggregated state; clear stale optimiser momentum ────────────
    # agg_state tensors are already on device; no .to(device) map needed.
    model.load_state_dict(agg_state)
    optimizer.state.clear()

    _cuda_sync(device)
    aggregation_time = time.time() - t_agg_start

    # ── Compute parallel-pipeline epoch time ────────────────────────────────
    # max_subgraph_pipeline_time: the slowest subgraph's end-to-end time.
    # sgcn_epoch_time_max models the wall-clock time when all R subgraphs
    # run in parallel, finishing when the slowest one completes, followed by
    # the aggregation step.
    #   sgcn_epoch_time_max = max_r(sample_r + train_r + eval_r)
    #                         + aggregation_time
    max_subgraph_pipeline_time = max(subgraph_total_times) if subgraph_total_times else 0.0
    sgcn_epoch_time_max        = max_subgraph_pipeline_time + aggregation_time

    avg_loss            = loss_sum / valid_batches if valid_batches > 0 else 0.0
    total_sampling_time = sum(subgraph_sampling_times)

    extra_sgcn = {
        'local_epochs':               local_epochs,
        'max_subgraph_pipeline_time': max_subgraph_pipeline_time,
        'aggregation_time':           aggregation_time,
        'subgraph_sampling_times':    subgraph_sampling_times,
        'subgraph_train_times':       subgraph_train_times,
        'subgraph_eval_times':        subgraph_eval_times,
        'subgraph_total_times':       subgraph_total_times,
    }
    return avg_loss, sgcn_epoch_time_max, total_sampling_time, extra_sgcn


def train_epoch(model, dataloader, criterion, optimizer, device,
                use_labels=False, n_classes=112):
    model.train()
    loss_sum, total = 0, 0
    sampling_time   = 0.0

    _cuda_sync(device)
    epoch_start = time.time()

    # Manual iterator is used to time each batch fetch (sampling_time) separately
    # from the forward/backward pass without restructuring the training loop.
    loader_iter = iter(dataloader)
    for _ in range(len(dataloader)):
        t_sample = time.time()
        batch = next(loader_iter)
        sampling_time += time.time() - t_sample

        batch      = batch.to(device)
        batch_size = batch.batch_size

        if use_labels:
            non_seed_idx = torch.arange(batch_size, batch.x.shape[0], device=device)
            x = add_labels(batch.x, batch.train_labels_onehot, non_seed_idx, n_classes, device)
        else:
            x = batch.x

        pred = model(x, batch.edge_index, batch.edge_attr)
        loss = criterion(pred[:batch_size], batch.y[:batch_size].float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * batch_size
        total    += batch_size

    _cuda_sync(device)
    epoch_time = time.time() - epoch_start

    return loss_sum / total, epoch_time, sampling_time


def train_epoch_fullbatch(model, x, edge_index, y, edge_attr, train_idx,
                          criterion, optimizer, device,
                          use_labels=False, n_classes=112,
                          train_labels_onehot=None):
    """Full-batch GCN training epoch.

    A single forward/backward pass is performed over the entire graph.
    Loss is computed only on training nodes.
    """
    model.train()

    _cuda_sync(device)
    epoch_start = time.time()

    if use_labels:
        n_nodes = x.shape[0]
        train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        train_mask[train_idx] = True
        non_train_idx_cpu = (~train_mask).nonzero(as_tuple=False).squeeze(1).cpu()
        x = add_labels(x, train_labels_onehot, non_train_idx_cpu, n_classes, device)

    pred = model(x, edge_index, edge_attr)
    loss = criterion(pred[train_idx], y[train_idx].float())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _cuda_sync(device)
    epoch_time = time.time() - epoch_start

    return loss.item(), epoch_time, 0.0


@torch.no_grad()
def evaluate_fullbatch(model, x, edge_index, labels, edge_attr,
                       train_idx, val_idx, test_idx,
                       criterion, evaluator, device,
                       use_labels=False, n_classes=112,
                       train_labels_onehot=None):
    """Full-batch evaluation over the entire graph."""
    model.eval()

    _cuda_sync(device)
    eval_start = time.time()

    if use_labels:
        all_idx_cpu = torch.arange(x.shape[0])  # CPU tensor; train_labels_onehot is CPU-resident
        x = add_labels(x, train_labels_onehot, all_idx_cpu, n_classes, device)

    pred = model(x, edge_index, edge_attr)

    _cuda_sync(device)
    eval_time = time.time() - eval_start

    train_loss = criterion(pred[train_idx], labels[train_idx].float()).item()
    val_loss   = criterion(pred[val_idx],   labels[val_idx].float()).item()
    test_loss  = criterion(pred[test_idx],  labels[test_idx].float()).item()

    return (
        evaluator(pred[train_idx], labels[train_idx]),
        evaluator(pred[val_idx],   labels[val_idx]),
        evaluator(pred[test_idx],  labels[test_idx]),
        train_loss, val_loss, test_loss,
        pred,
        eval_time,
    )


def train_epoch_saint(model, dataloader, criterion, optimizer, device,
                      train_idx, use_labels=False, n_classes=112):
    """Training epoch using GraphSAINT subgraph-sampling batches.

    GraphSAINT batches are induced subgraphs where every node may be a
    training node.  Training nodes are identified via ``batch.train_mask``.
    When the batch carries ``batch.node_norm`` (sampling normalisation
    weights produced by the GraphSAINT sampler), each per-node loss term is
    scaled by the corresponding weight before summing.
    """
    model.train()
    loss_sum, total = 0, 0
    sampling_time   = 0.0

    _cuda_sync(device)
    epoch_start = time.time()

    loader_iter = iter(dataloader)
    for _ in range(len(dataloader)):
        t_sample = time.time()
        batch = next(loader_iter)
        sampling_time += time.time() - t_sample

        batch = batch.to(device)

        train_mask = batch.train_mask
        if train_mask.sum() == 0:
            continue

        if use_labels:
            # Reveal labels for non-training nodes only (same convention as
            # train_epoch, where seed-node labels are withheld).
            non_train_local_idx = torch.where(~train_mask)[0]
            x = add_labels(batch.x, batch.train_labels_onehot,
                           non_train_local_idx, n_classes, device)
        else:
            x = batch.x

        pred = model(x, batch.edge_index, batch.edge_attr)

        # Apply GraphSAINT sampling normalisation weights when available.
        if hasattr(batch, 'node_norm'):
            loss_per_node = F.binary_cross_entropy_with_logits(
                pred[train_mask], batch.y[train_mask].float(), reduction='none'
            )
            loss = (loss_per_node * batch.node_norm[train_mask]).sum()
        else:
            loss = criterion(pred[train_mask], batch.y[train_mask].float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_train_in_batch = train_mask.sum().item()
        # node_norm path: loss is already a weighted sum; add directly.
        # fallback path: criterion returns a mean; scale back to a sum for
        # consistent per-node averaging across the epoch.
        if hasattr(batch, 'node_norm'):
            loss_sum += loss.item()
        else:
            loss_sum += loss.item() * n_train_in_batch
        total    += n_train_in_batch

    _cuda_sync(device)
    epoch_time = time.time() - epoch_start

    return (loss_sum / total if total > 0 else 0.0), epoch_time, sampling_time


@torch.no_grad()
def evaluate(model, dataloader, labels, train_idx, val_idx, test_idx,
             criterion, evaluator, device, use_labels=False, n_classes=112):
    model.eval()
    preds      = torch.zeros(labels.shape, device=device)
    eval_times = 1

    _cuda_sync(device)
    eval_start = time.time()

    for _ in range(eval_times):
        for batch in dataloader:
            batch      = batch.to(device)
            batch_size = batch.batch_size

            if use_labels:
                all_idx = torch.arange(batch.x.shape[0], device=device)
                x = add_labels(batch.x, batch.train_labels_onehot, all_idx, n_classes, device)
            else:
                x = batch.x

            pred = model(x, batch.edge_index, batch.edge_attr)
            preds[batch.n_id[:batch_size]] += pred[:batch_size]

    preds /= eval_times

    _cuda_sync(device)
    eval_time = time.time() - eval_start

    train_loss = criterion(preds[train_idx], labels[train_idx].float()).item()
    val_loss   = criterion(preds[val_idx],   labels[val_idx].float()).item()
    test_loss  = criterion(preds[test_idx],  labels[test_idx].float()).item()

    return (
        evaluator(preds[train_idx], labels[train_idx]),
        evaluator(preds[val_idx],   labels[val_idx]),
        evaluator(preds[test_idx],  labels[test_idx]),
        train_loss, val_loss, test_loss,
        preds,
        eval_time,
    )


def run(data, labels, train_idx, val_idx, test_idx, evaluator, n_running,
        gen_model_fn, device, n_layers, lr, weight_decay, n_epochs,
        eval_every, log_every, save_pred, use_labels=False, n_classes=112,
        mpnn='gcn',
        subsampling_method='random_node',
        truncation_ratio=0.2,
        aggregation_method='sgcn',
        n_subgraphs=5,
        local_epochs=5,
        subgraph_max_nodes=256,
        max_subgraph_edges=300000,
        debug_subgraph_stats=False,
        min_subgraph_nodes=0,
        min_train_nodes_in_subgraph=_SGCN_MIN_TRAIN_NODES):
    evaluator_wrapper = lambda pred, lbls: evaluator.eval(
        {'y_pred': pred, 'y_true': lbls}
    )['rocauc']

    train_batch_size = (len(train_idx) + 9) // 10

    if mpnn == 'graphsaint':
        # Attach boolean split masks to data so GraphSAINT batches inherit them.
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask[val_idx] = True
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[test_idx] = True

        # GraphSAINT: sample random-walk-induced subgraphs instead of
        # per-node neighborhoods.  num_steps mirrors the ~10 batches/epoch
        # produced by NeighborLoader, and walk_length provides a 2-hop reach.
        saint_num_steps = max(len(train_idx) // train_batch_size, 1)
        train_loader = GraphSAINTRandomWalkSampler(
            data,
            batch_size=train_batch_size,
            walk_length=2,
            num_steps=saint_num_steps,
            num_workers=4,
        )
    elif mpnn == 'sgcn':
        # SGCN handles its own subgraph sampling inside train_epoch_sgcn;
        # no external DataLoader is required.
        train_loader = None
    elif mpnn == 'gcn':
        # GCN uses full-batch training; no DataLoader is required.
        train_loader = None
    else:
        train_loader = NeighborLoader(
            data,
            num_neighbors=[16] * n_layers,
            batch_size=train_batch_size,
            input_nodes=train_idx.cpu(),
            shuffle=True,
            num_workers=4,
        )

    if mpnn != 'gcn':
        eval_loader = NeighborLoader(
            data,
            num_neighbors=[32] * n_layers,
            batch_size=32768,
            input_nodes=torch.cat([train_idx.cpu(), val_idx.cpu(), test_idx.cpu()]),
            shuffle=False,
            num_workers=4,
        )
    else:
        eval_loader = None

    criterion    = nn.BCEWithLogitsLoss()
    model        = gen_model_fn().to(device)
    optimizer    = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.75, patience=50, verbose=True
    )

    best_val_score, final_test_score = 0, 0
    val_score, test_score = 0, 0
    final_pred  = None
    epoch_records = []

    # ── SGCN: pre-load full graph to GPU and keep resident for all epochs. ───
    # Doing this once here avoids repeated host→device copies inside the inner
    # subgraph loop, which was the primary cause of CPU-bound behaviour.
    if mpnn == 'sgcn':
        _sgcn_x_dev         = data.x.to(device)
        _sgcn_y_dev         = data.y.to(device)
        _sgcn_edge_index_dev = data.edge_index.to(device)
        _sgcn_edge_attr_dev  = data.edge_attr.to(device) if data.edge_attr is not None else None
        _sgcn_train_idx_dev  = train_idx.to(device)
        _sgcn_val_idx_dev    = val_idx.to(device)
    else:
        _sgcn_x_dev = _sgcn_y_dev = _sgcn_edge_index_dev = _sgcn_edge_attr_dev = None
        _sgcn_train_idx_dev = _sgcn_val_idx_dev = None

    # ── GCN: pre-load full graph to GPU for full-batch training/evaluation. ──
    if mpnn == 'gcn':
        _gcn_x_dev          = data.x.to(device)
        _gcn_edge_index_dev = data.edge_index.to(device)
        _gcn_edge_attr_dev  = data.edge_attr.to(device) if data.edge_attr is not None else None
        _gcn_train_idx_dev  = train_idx.to(device)
        _gcn_val_idx_dev    = val_idx.to(device)
        _gcn_test_idx_dev   = test_idx.to(device)
    else:
        _gcn_x_dev = _gcn_edge_index_dev = _gcn_edge_attr_dev = None
        _gcn_train_idx_dev = _gcn_val_idx_dev = _gcn_test_idx_dev = None

    _cuda_sync(device)
    run_start = time.time()

    for epoch in range(1, n_epochs + 1):
        if mpnn == 'graphsaint':
            loss, epoch_time, sampling_time = train_epoch_saint(
                model, train_loader, criterion, optimizer, device,
                train_idx, use_labels, n_classes
            )
        elif mpnn == 'sgcn':
            loss, epoch_time, sampling_time, extra_sgcn = train_epoch_sgcn(
                model, data, criterion, optimizer, device,
                _sgcn_train_idx_dev, _sgcn_val_idx_dev,
                n_subgraphs=n_subgraphs,
                local_epochs=local_epochs,
                subsampling_method=subsampling_method,
                subgraph_max_nodes=subgraph_max_nodes,
                max_subgraph_edges=max_subgraph_edges,
                truncation_ratio=truncation_ratio,
                aggregation_method=aggregation_method,
                use_labels=use_labels, n_classes=n_classes,
                debug_subgraph_stats=debug_subgraph_stats,
                x_full_dev=_sgcn_x_dev,
                y_full_dev=_sgcn_y_dev,
                edge_index_dev=_sgcn_edge_index_dev,
                edge_attr_dev=_sgcn_edge_attr_dev,
                min_subgraph_nodes=min_subgraph_nodes,
                min_train_nodes_in_subgraph=min_train_nodes_in_subgraph,
            )
        elif mpnn == 'gcn':
            loss, epoch_time, sampling_time = train_epoch_fullbatch(
                model, _gcn_x_dev, _gcn_edge_index_dev, labels, _gcn_edge_attr_dev,
                _gcn_train_idx_dev, criterion, optimizer, device,
                use_labels=use_labels, n_classes=n_classes,
                train_labels_onehot=data.train_labels_onehot,
            )
        else:
            loss, epoch_time, sampling_time = train_epoch(
                model, train_loader, criterion, optimizer, device, use_labels, n_classes
            )

        record = {
            'epoch':               epoch,
            'train_loss':          loss,
            'val_auc':             float('nan'),
            'test_auc':            float('nan'),
            'train_sampling_time': sampling_time,
            'train_epoch_time':    epoch_time,
            'eval_time':           float('nan'),
        }

        # For SGCN, add detailed timing fields from the parallel-pipeline model.
        # epoch_time already holds sgcn_epoch_time_max for SGCN.
        if mpnn == 'sgcn':
            record.update({
                'local_epochs':               extra_sgcn['local_epochs'],
                'max_subgraph_pipeline_time': extra_sgcn['max_subgraph_pipeline_time'],
                'aggregation_time':           extra_sgcn['aggregation_time'],
                'sgcn_epoch_time_max':        epoch_time,
                'subgraph_sampling_times':    extra_sgcn['subgraph_sampling_times'],
                'subgraph_train_times':       extra_sgcn['subgraph_train_times'],
                'subgraph_eval_times':        extra_sgcn['subgraph_eval_times'],
                'subgraph_total_times':       extra_sgcn['subgraph_total_times'],
            })

        if epoch == n_epochs or epoch % eval_every == 0 or epoch % log_every == 0:
            if mpnn == 'gcn':
                (train_score, val_score, test_score,
                 train_loss, val_loss, test_loss,
                 pred, eval_time) = evaluate_fullbatch(
                    model, _gcn_x_dev, _gcn_edge_index_dev, labels, _gcn_edge_attr_dev,
                    _gcn_train_idx_dev, _gcn_val_idx_dev, _gcn_test_idx_dev,
                    criterion, evaluator_wrapper, device,
                    use_labels=use_labels, n_classes=n_classes,
                    train_labels_onehot=data.train_labels_onehot,
                )
            else:
                (train_score, val_score, test_score,
                 train_loss, val_loss, test_loss,
                 pred, eval_time) = evaluate(
                    model, eval_loader, labels, train_idx, val_idx, test_idx,
                    criterion, evaluator_wrapper, device, use_labels, n_classes
                )

            record['val_auc']   = val_score
            record['test_auc']  = test_score
            record['eval_time'] = eval_time

            if val_score > best_val_score:
                best_val_score   = val_score
                final_test_score = test_score
                final_pred       = pred

            if epoch % log_every == 0:
                print(
                    f'Epoch: {epoch:04d} | '
                    f'Loss: {loss:.4f} | '
                    f'Train: {100 * train_score:.2f}% | '
                    f'Valid: {100 * val_score:.2f}% | '
                    f'Test: {100 * test_score:.2f}% | '
                    f'Best Valid: {100 * best_val_score:.2f}% | '
                    f'Best Test: {100 * final_test_score:.2f}%'
                )

        epoch_records.append(record)
        lr_scheduler.step(val_score)

    _cuda_sync(device)
    total_run_time = time.time() - run_start

    if save_pred and final_pred is not None:
        os.makedirs('./output', exist_ok=True)
        torch.save(torch.sigmoid(final_pred), f'./output/{n_running}.pt')

    return {
        'best_val_auc':   best_val_score,
        'best_test_auc':  final_test_score,
        'final_val_auc':  val_score,
        'final_test_auc': test_score,
        'total_run_time': total_run_time,
        'epoch_records':  epoch_records,
    }
