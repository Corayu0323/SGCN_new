import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_loss_curve(epoch_df, figures_dir, method=None):
    """Plot training loss vs epoch for each run and save as loss_curve.png."""
    fig, ax = plt.subplots()
    for run_id, grp in epoch_df.groupby('run_id'):
        ax.plot(grp['epoch'], grp['train_loss'], label=f'Run {run_id}')
    title = 'Training Loss vs Epoch'
    if method is not None:
        title = f'[{method}] {title}'
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss')
    ax.legend()
    path = os.path.join(figures_dir, 'loss_curve.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


def plot_auc_curve(epoch_df, figures_dir, method=None):
    """Plot validation and test AUC vs epoch for each run and save as auc_curve.png."""
    fig, ax = plt.subplots()
    eval_df = epoch_df.dropna(subset=['val_auc', 'test_auc'])
    for run_id, grp in eval_df.groupby('run_id'):
        ax.plot(grp['epoch'], grp['val_auc'],  label=f'Val (Run {run_id})')
        ax.plot(grp['epoch'], grp['test_auc'], linestyle='--',
                label=f'Test (Run {run_id})')
    title = 'Validation and Test AUC vs Epoch'
    if method is not None:
        title = f'[{method}] {title}'
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ROC-AUC')
    ax.legend()
    path = os.path.join(figures_dir, 'auc_curve.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


def plot_auc_boxplot(run_df, figures_dir, method=None):
    """Boxplot of best_test_auc distribution across runs per method."""
    methods = run_df['method'].unique().tolist()
    data    = [run_df.loc[run_df['method'] == m, 'best_test_auc'].values
               for m in methods]
    fig, ax = plt.subplots()
    ax.boxplot(data, labels=methods)
    title = 'Best Test AUC Distribution Across Runs'
    if method is not None:
        title = f'[{method}] {title}'
    ax.set_title(title)
    ax.set_xlabel('Method')
    ax.set_ylabel('Best Test ROC-AUC')
    path = os.path.join(figures_dir, 'auc_boxplot.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


def plot_time_bar(agg_dict, figures_dir, method=None):
    """Bar chart of mean timing statistics and save as time_bar.png."""
    keys   = ['mean_train_epoch_time', 'mean_eval_time', 'mean_total_run_time']
    labels = ['Train Epoch', 'Eval', 'Total Run']
    values = [agg_dict[k] for k in keys]
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    title = 'Mean Timing Statistics'
    if method is not None:
        title = f'[{method}] {title}'
    ax.set_title(title)
    ax.set_xlabel('Phase')
    ax.set_ylabel('Time (seconds)')
    path = os.path.join(figures_dir, 'time_bar.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')
