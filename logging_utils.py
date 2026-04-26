import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import yaml


def setup_dirs(results_dir='results'):
    """Create results/csv and results/figures directories."""
    csv_dir     = os.path.join(results_dir, 'csv')
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(csv_dir,     exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    return csv_dir, figures_dir


def setup_experiment_dir(results_dir='results', timestamp=None):
    """Create a unique experiment folder and a figures subfolder inside it.

    Returns
    -------
    exp_dir     : str – path to the new experiment folder.
    figures_dir : str – path to the figures subfolder inside exp_dir.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir     = os.path.join(results_dir, f'exp_{timestamp}')
    figures_dir = os.path.join(exp_dir, 'figures')
    os.makedirs(exp_dir,     exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    return exp_dir, figures_dir


def _get_git_commit():
    """Return the current HEAD commit SHA, or 'N/A' if not in a git repo."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else 'N/A'
    except Exception as exc:
        print(f'[logging_utils] Could not retrieve git commit: {exc}')
        return 'N/A'


def save_config(config, exp_dir, device=None):
    """Save experiment configuration and reproducibility metadata to config.yaml.

    Parameters
    ----------
    config  : dict – hyperparameters and settings for the experiment.
    exp_dir : str  – path to the experiment folder.
    device  : optional torch.device or str – device used for training.
    """
    import torch

    meta = {
        'timestamp':      datetime.now().strftime('%Y%m%d_%H%M%S'),
        'git_commit':     _get_git_commit(),
        'python_version': sys.version,
        'torch_version':  torch.__version__,
        'device':         str(device) if device is not None else 'N/A',
    }
    full_config = {'config': config, 'meta': meta}
    path = os.path.join(exp_dir, 'config.yaml')
    with open(path, 'w') as f:
        yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)
    print(f'Saved: {path}')
    return full_config


def update_experiment_index(exp_dir, config, agg_stats, results_dir='results'):
    """Append one row to results/experiment_index.csv for this experiment.

    Parameters
    ----------
    exp_dir     : str  – path to the experiment folder (used as exp_id).
    config      : dict – hyperparameters (must contain method, dataset, etc.).
    agg_stats   : dict – aggregated statistics dict (from compute_aggregate).
    results_dir : str  – root results directory.
    """
    exp_id    = os.path.basename(exp_dir)
    timestamp = exp_id.replace('exp_', '', 1)

    row = {
        'exp_id':             exp_id,
        'timestamp':          timestamp,
        'method':             config.get('method', 'N/A'),
        'dataset':            config.get('dataset', 'N/A'),
        'sampling_method':    config.get('sampling_method', 'N/A'),
        'trunc_ratio':        config.get('trunc_ratio', float('nan')),
        'local_epochs':       config.get('local_epochs', float('nan')),
        'mean_best_test_auc': agg_stats.get('mean_best_test_auc', float('nan')),
        'std_best_test_auc':  agg_stats.get('std_best_test_auc', float('nan')),
        'n_runs':             agg_stats.get('n_runs', 0),
    }

    index_path = os.path.join(results_dir, 'experiment_index.csv')
    new_row_df = pd.DataFrame([row])

    if os.path.exists(index_path):
        existing = pd.read_csv(index_path)
        updated  = pd.concat([existing, new_row_df], ignore_index=True)
    else:
        updated = new_row_df

    updated.to_csv(index_path, index=False)
    print(f'Updated: {index_path}')
    return updated


def build_epoch_df(method, run_id, seed, epoch_records):
    """Return a DataFrame with per-epoch metrics for a single run."""
    df = pd.DataFrame(epoch_records)
    df.insert(0, 'seed',   seed)
    df.insert(0, 'run_id', run_id)
    df.insert(0, 'method', method)
    return df


def build_run_record(method, run_id, seed, result):
    """Return a dict summarising a single run."""
    epoch_records  = result['epoch_records']
    eval_times     = [r['eval_time']           for r in epoch_records
                      if not np.isnan(r['eval_time'])]
    epoch_times    = [r['train_epoch_time']    for r in epoch_records]
    sampling_times = [r['train_sampling_time'] for r in epoch_records]

    record = {
        'method':                  method,
        'run_id':                  run_id,
        'seed':                    seed,
        'best_val_auc':            result['best_val_auc'],
        'best_test_auc':           result['best_test_auc'],
        'final_val_auc':           result['final_val_auc'],
        'final_test_auc':          result['final_test_auc'],
        'mean_train_sampling_time': np.mean(sampling_times),
        'mean_train_epoch_time':   np.mean(epoch_times),
        'mean_eval_time':          np.mean(eval_times) if eval_times else float('nan'),
        'total_run_time':          result['total_run_time'],
    }

    # For SGCN, include parallel-pipeline timing summary fields.
    if method == 'sgcn' and epoch_records and 'sgcn_epoch_time_max' in epoch_records[0]:
        sgcn_epoch_times = [r['sgcn_epoch_time_max']        for r in epoch_records]
        max_sg_times     = [r['max_subgraph_pipeline_time'] for r in epoch_records]
        agg_times        = [r['aggregation_time']           for r in epoch_records]
        local_epochs_val = epoch_records[0].get('local_epochs', float('nan'))
        record.update({
            'local_epochs':                  local_epochs_val,
            'mean_sgcn_epoch_time_max':      np.mean(sgcn_epoch_times),
            'mean_max_subgraph_pipeline_time': np.mean(max_sg_times),
            'mean_aggregation_time':         np.mean(agg_times),
        })

    return record


def save_epoch_metrics(epoch_dfs, csv_dir):
    """Concatenate per-run epoch DataFrames and write epoch_metrics.csv."""
    df   = pd.concat(epoch_dfs, ignore_index=True)
    path = os.path.join(csv_dir, 'epoch_metrics.csv')
    df.to_csv(path, index=False)
    print(f'Saved: {path}')
    return df


def save_run_summary(run_records, csv_dir):
    """Write one row per run to run_summary.csv."""
    df   = pd.DataFrame(run_records)
    path = os.path.join(csv_dir, 'run_summary.csv')
    df.to_csv(path, index=False)
    print(f'Saved: {path}')
    return df


def compute_aggregate(run_records):
    """Return a dict of aggregated statistics across runs."""
    df     = pd.DataFrame(run_records)
    method = df['method'].iloc[0] if len(df) > 0 else 'unknown'
    n_runs = len(df)
    agg = {
        'method':                 method,
        'n_runs':                 n_runs,
        'mean_best_val_auc':      df['best_val_auc'].mean(),
        'std_best_val_auc':       df['best_val_auc'].std(ddof=1),
        'mean_best_test_auc':     df['best_test_auc'].mean(),
        'std_best_test_auc':      df['best_test_auc'].std(ddof=1),
        'mean_train_epoch_time':  df['mean_train_epoch_time'].mean(),
        'std_train_epoch_time':   df['mean_train_epoch_time'].std(ddof=1),
        'mean_eval_time':         df['mean_eval_time'].mean(),
        'std_eval_time':          df['mean_eval_time'].std(ddof=1),
        'mean_total_run_time':    df['total_run_time'].mean(),
        'std_total_run_time':     df['total_run_time'].std(ddof=1),
    }
    # For SGCN runs include parallel-pipeline timing aggregates.
    if 'mean_sgcn_epoch_time_max' in df.columns:
        agg.update({
            'local_epochs':                       df['local_epochs'].iloc[0],
            'mean_sgcn_epoch_time_max':           df['mean_sgcn_epoch_time_max'].mean(),
            'std_sgcn_epoch_time_max':            df['mean_sgcn_epoch_time_max'].std(ddof=1),
            'mean_max_subgraph_pipeline_time':    df['mean_max_subgraph_pipeline_time'].mean(),
            'std_max_subgraph_pipeline_time':     df['mean_max_subgraph_pipeline_time'].std(ddof=1),
            'mean_aggregation_time':              df['mean_aggregation_time'].mean(),
            'std_aggregation_time':               df['mean_aggregation_time'].std(ddof=1),
        })
    return agg


def save_aggregate_summary(run_records, csv_dir):
    """Write one row of aggregated statistics to aggregate_summary.csv."""
    agg  = compute_aggregate(run_records)
    df   = pd.DataFrame([agg])
    path = os.path.join(csv_dir, 'aggregate_summary.csv')
    df.to_csv(path, index=False)
    print(f'Saved: {path}')
    return df
