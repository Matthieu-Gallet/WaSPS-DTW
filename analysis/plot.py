"""
Visualization functions for barycenter, classification, and geographic results.

All functions are pure: they take numpy arrays and produce matplotlib figures.
No file paths, no I/O inside plot functions.

Module-level flag:
  SAVE_FIGURES = False   (set to True in batch contexts to save instead of display)
"""

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix

SAVE_FIGURES = False


# =============================================================================
# Style
# =============================================================================

def setup_ieee_style():
    """Configure matplotlib for IEEE publication style."""
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'STIXGeneral']
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.size'] = 8
    rcParams['axes.labelsize'] = 9
    rcParams['axes.titlesize'] = 9
    rcParams['xtick.labelsize'] = 7
    rcParams['ytick.labelsize'] = 7
    rcParams['legend.fontsize'] = 7
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.format'] = 'pdf'
    rcParams['axes.linewidth'] = 0.5
    rcParams['grid.linewidth'] = 0.5
    rcParams['lines.linewidth'] = 1.0


# =============================================================================
# Barycenter / series plots
# =============================================================================

def plot_with_correspondences(ax, lambda1: np.ndarray, lambda2: np.ndarray,
                               lambda_bary: np.ndarray, title: str,
                               gamma: float, threshold: float = 0.1):
    """
    Plot two series with their barycenter and Soft-DTW alignment correspondences.

    Args:
        ax: Matplotlib axis
        lambda1, lambda2: Input series
        lambda_bary: Computed barycenter series
        title: Plot title
        gamma: Soft-DTW gamma used for visualization
        threshold: Alignment weight threshold for drawing correspondence lines
    """
    from sdtw.distance import SquaredEuclidean
    from sdtw.soft_dtw import SoftDTW

    ax.set_yscale('log')
    coords1 = np.arange(len(lambda1))
    coords2 = np.arange(len(lambda2))
    coords_b = np.arange(len(lambda_bary))

    ax.plot(coords1, lambda1, 'o-', color='blue', linewidth=2.5, markersize=10,
            label='Series 1',
            path_effects=[pe.Stroke(linewidth=4, foreground='white'), pe.Normal()])
    ax.plot(coords2, lambda2, 's-', color='red', linewidth=2.5, markersize=10,
            label='Series 2',
            path_effects=[pe.Stroke(linewidth=4, foreground='white'), pe.Normal()])
    ax.plot(coords_b, lambda_bary, 'd-', color='green', linewidth=3, markersize=12,
            label='Barycenter',
            path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])

    for lam, coords in [(lambda1, coords1), (lambda2, coords2)]:
        D = SquaredEuclidean(lam[:, np.newaxis], lambda_bary[:, np.newaxis])
        sdtw = SoftDTW(D, gamma=gamma)
        sdtw.compute()
        E = sdtw.grad()
        E_norm = E / (np.max(E) + 1e-8)
        for i in range(len(lam)):
            for j in range(len(lambda_bary)):
                w = E_norm[i, j]
                if w > threshold:
                    ax.plot([coords[i], coords_b[j]], [lam[i], lambda_bary[j]],
                            'k-', alpha=w * 0.5, linewidth=w * 2)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Temporal index', fontsize=12)
    ax.set_ylabel('λ (log scale)', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, which='both')


def plot_multiple_series(ax, lambda_list: List[np.ndarray], lambda_bary: np.ndarray,
                          title: str, gamma: float, threshold: float = 0.15):
    """
    Plot multiple series with their barycenter and alignment correspondences.
    """
    from sdtw.distance import SquaredEuclidean
    from sdtw.soft_dtw import SoftDTW

    ax.set_yscale('log')
    colors = ['blue', 'red', 'orange', 'purple']
    markers = ['o', 's', '^', 'v']
    coords_b = np.arange(len(lambda_bary))

    for idx, (lam, color, marker) in enumerate(zip(lambda_list, colors, markers)):
        coords = np.arange(len(lam))
        ax.plot(coords, lam, f'{marker}-', color=color, linewidth=2, markersize=8,
                label=f'Series {idx + 1}', alpha=0.7,
                path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])

    ax.plot(coords_b, lambda_bary, 'd-', color='green', linewidth=3, markersize=12,
            label='Barycenter',
            path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])

    for lam in lambda_list:
        D = SquaredEuclidean(lam[:, np.newaxis], lambda_bary[:, np.newaxis])
        sdtw = SoftDTW(D, gamma=gamma)
        sdtw.compute()
        E = sdtw.grad()
        E_norm = E / (np.max(E) + 1e-8)
        coords = np.arange(len(lam))
        for i in range(len(lam)):
            for j in range(len(lambda_bary)):
                w = E_norm[i, j]
                if w > threshold:
                    ax.plot([coords[i], coords_b[j]], [lam[i], lambda_bary[j]],
                            'k-', alpha=w * 0.3, linewidth=w * 1.5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Temporal index', fontsize=12)
    ax.set_ylabel('λ (log scale)', fontsize=12)
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.grid(True, alpha=0.3, which='both')


def create_barycenter_comparison_figure(lambda_est1: np.ndarray, lambda_est2: np.ndarray,
                                         lambda_bary_raw: np.ndarray,
                                         lambda_bary_est: np.ndarray,
                                         lambda_bary_wass: np.ndarray,
                                         gamma: float,
                                         output_path: Optional[str] = None):
    """
    Three-panel figure comparing Euclidean-raw, Euclidean-params, and Wasserstein barycenters.
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    plot_with_correspondences(axes[0], lambda_est1, lambda_est2, lambda_bary_raw,
                               f'Euclidean on raw data\n(gamma={gamma})', gamma)
    plot_with_correspondences(axes[1], lambda_est1, lambda_est2, lambda_bary_est,
                               f'Euclidean on estimated parameters\n(gamma={gamma})', gamma)
    plot_with_correspondences(axes[2], lambda_est1, lambda_est2, lambda_bary_wass,
                               f'Wasserstein SGD\n(gamma={gamma})', gamma)
    plt.tight_layout()
    if output_path and SAVE_FIGURES:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


# =============================================================================
# Geographic zone plots
# =============================================================================

def plot_zone(ax, lambda_series: List[np.ndarray], bary: np.ndarray,
              title_suffix: str,
              markers: Optional[List[str]] = None,
              colors: Optional[List[str]] = None):
    """Plot a geographic zone: individual series + barycenter."""
    if colors is None:
        colors = ['blue', 'red', 'orange', 'purple']
    if markers is None:
        markers = ['o', 's', '^', 'v']

    ax.set_yscale('log')
    for i, lam in enumerate(lambda_series):
        ax.plot(np.arange(len(lam)), lam,
                marker=markers[i % len(markers)], color=colors[i % len(colors)],
                linewidth=1, markersize=3, alpha=0.25, label=f'Series {i + 1}')

    ax.plot(np.arange(len(bary)), bary, 'd-', color='green', linewidth=3, markersize=4,
            label='Barycenter',
            path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
    ax.set_title(title_suffix)
    ax.set_xlabel('Temporal index')
    ax.set_ylabel('λ (log)')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)


def create_groups_barycenter_figures(results: Dict, gamma: float, figures_dir: str):
    """
    Create 3 figures (one per barycenter method) each with one subplot per group.

    Args:
        results: {group_key: {'lambda_series': List[ndarray],
                              'bary_euclidean_raw': ndarray,
                              'bary_euclidean_params': ndarray,
                              'bary_wasserstein_sgd': ndarray}}
        gamma:       Gamma used in the experiment.
        figures_dir: Output directory.
    """
    colors = ['blue', 'red', 'orange', 'purple']
    markers = ['o', 's', '^', 'v']
    n_groups = len(results)

    figs = {m: plt.subplots(n_groups, 1, figsize=(12, 5 * n_groups), sharex=True)
            for m in ('euclidean_raw', 'euclidean_params', 'wasserstein_sgd')}
    for fig, axes in figs.values():
        if n_groups == 1:
            axes = [axes]

    for i, (group_key, group) in enumerate(results.items()):
        subtitle = group.get('name', group_key)
        lambda_series = group['lambda_series']
        for method in ('euclidean_raw', 'euclidean_params', 'wasserstein_sgd'):
            fig, axes = figs[method]
            ax = axes[i] if n_groups > 1 else axes[0]
            plot_zone(ax, lambda_series, group[f'bary_{method}'],
                      subtitle, markers, colors)

    method_labels = {
        'euclidean_raw':    'Euclidean on raw data',
        'euclidean_params': 'Euclidean on parameters',
        'wasserstein_sgd':  'Wasserstein SGD',
    }
    for method, (fig, _) in figs.items():
        fig.suptitle(f'Group barycenters — {method_labels[method]} (γ={gamma})',
                     fontsize=16)
        plt.figure(fig.number)
        plt.tight_layout()
        if SAVE_FIGURES:
            out = Path(figures_dir) / f'group_barycenters_{method}_gamma_{gamma}.png'
            plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)


# =============================================================================
# Prediction plots
# =============================================================================

def plot_predictions(X_test: np.ndarray, Y_test: np.ndarray, Y_pred_dict: Dict,
                     train_length: int, predict_length: int,
                     n_plots: int = 4, output_path: Optional[str] = None):
    """Plot prediction results comparing multiple models."""
    fig = plt.figure(figsize=(20, 6))
    pos = 220
    colors = {'Soft-DTW': 'blue', 'Wasserstein': 'green', 'Euclidean': 'orange'}
    n_plots = min(n_plots, X_test.shape[0])

    for i in range(n_plots):
        pos += 1
        ax = fig.add_subplot(pos)
        gt = np.concatenate([X_test[i], Y_test[i]])[-train_length - predict_length:]
        n_in = len(X_test[i])
        for name, Y_pred in Y_pred_dict.items():
            ax.plot(range(n_in, n_in + len(Y_pred[i])), Y_pred[i],
                    alpha=0.75, lw=3, label=name, color=colors.get(name, 'blue'), zorder=10)
        ax.semilogy(gt, c='k', alpha=0.3, lw=3, label='Ground truth', zorder=5)
        y_min, y_max = ax.get_ylim()
        ax.plot([n_in, n_in], [y_min, y_max], lw=3, ls='--', c='red', alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value (log scale)')
        ax.set_ylim(max(np.min(gt), 1e-1), y_max)
        ax.legend()

    fig.set_tight_layout(True)
    if output_path and SAVE_FIGURES:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_losses(losses_dict: Dict, output_path: Optional[str] = None, log_scale: bool = True):
    """Plot training loss curves for multiple models."""
    colors = {'Soft-DTW': 'blue', 'Wasserstein': 'green', 'Euclidean': 'orange'}
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, losses in losses_dict.items():
        ax.plot(losses, label=name, color=colors.get(name, 'blue'))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    if log_scale:
        ax.set_yscale('log')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True)
    if output_path and SAVE_FIGURES:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


# =============================================================================
# Classification plots
# =============================================================================

def plot_confusion_matrix(Y_true: np.ndarray, Y_pred: np.ndarray,
                           idx_to_label: Dict[int, str],
                           title: str = '',
                           output_path: Optional[str] = None) -> plt.Figure:
    """
    Plot a single normalised confusion matrix.

    Args:
        Y_true:       Ground-truth integer labels.
        Y_pred:       Predicted integer labels.
        idx_to_label: {int: class_name} mapping.
        title:        Figure title (e.g. method name + F1).
        output_path:  If set and SAVE_FIGURES, save as PDF to this path.
    """
    setup_ieee_style()
    labels = sorted(idx_to_label.keys())
    target_names = [idx_to_label[i] for i in labels]

    cm = confusion_matrix(Y_true, Y_pred, labels=labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color=color, fontsize=8)
    ax.set_xticks(np.arange(len(target_names)))
    ax.set_yticks(np.arange(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha='right')
    ax.set_yticklabels(target_names)
    ax.set_ylabel('True Class')
    ax.set_xlabel('Predicted Class')
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if output_path and SAVE_FIGURES:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
    return fig


def plot_confusion_matrices_all_methods(
        results_by_method: Dict[str, dict],
        idx_to_label: Dict[int, str],
        output_dir: Optional[str] = None) -> List[plt.Figure]:
    """
    Plot one confusion matrix per method.

    Args:
        results_by_method: {'euclidean_raw': {'predictions': ..., 'Y_test': ...,
                                               'f1_weighted': ...}, ...}
        idx_to_label:      {int: class_name}
        output_dir:        If set and SAVE_FIGURES, save each figure.
    """
    figs = []
    for method, r in results_by_method.items():
        title = f'{method}  F1={r["f1_weighted"]:.3f}'
        out = (str(Path(output_dir) / f'confusion_{method}.pdf')
               if output_dir else None)
        figs.append(plot_confusion_matrix(
            r['Y_test'], r['predictions'], idx_to_label, title, out
        ))
    return figs


def plot_gamma_sensitivity(aggregated: Dict[str, Dict],
                           methods: Optional[List[str]] = None,
                           output_dir: Optional[str] = None) -> plt.Figure:
    """
    Boxplot of weighted F1 across gamma values, one series per method.

    Args:
        aggregated: {method: {gamma: [f1_fold_0, f1_fold_1, ...]}}
                    Built from aggregate_folds(results, group_by=['gamma', 'method'])
                    and reorganised by the caller.
        methods:    Methods to plot (default: all keys in aggregated).
        output_dir: If set and SAVE_FIGURES, save as PDF.
    """
    setup_ieee_style()
    if methods is None:
        methods = list(aggregated.keys())

    all_gammas = sorted({g for m in methods for g in aggregated[m]})
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    fig, ax = plt.subplots(figsize=(7, 2.5))
    width = 0.8 / len(methods)

    for mi, (method, color) in enumerate(zip(methods, colors)):
        positions = [i + (mi - len(methods) / 2 + 0.5) * width
                     for i in range(len(all_gammas))]
        data = [aggregated[method].get(g, []) for g in all_gammas]
        data = [d for d in data if d]
        if not data:
            continue
        bp = ax.boxplot(data, positions=positions[:len(data)], widths=width * 0.9,
                        patch_artist=True, showfliers=False,
                        medianprops={'linewidth': 1.5, 'color': 'black'})
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        bp['boxes'][0].set_label(method)

    ax.set_xticks(range(len(all_gammas)))
    ax.set_xticklabels([f'{g:.0e}' if (g < 0.1 or g > 10) else f'{g:.1f}'
                        for g in all_gammas])
    ax.set_xlabel('Gamma (γ)')
    ax.set_ylabel('F1 (weighted)')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if output_dir and SAVE_FIGURES:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(output_dir) / 'gamma_sensitivity.pdf', bbox_inches='tight')
    return fig


def plot_sample_size_sensitivity(aggregated: Dict[str, Dict],
                                 methods: Optional[List[str]] = None,
                                 output_dir: Optional[str] = None) -> plt.Figure:
    """
    Boxplot of weighted F1 across training sample fractions, one series per method.

    Args:
        aggregated: {method: {fraction: [f1_fold_0, f1_fold_1, ...]}}
        methods:    Methods to plot (default: all keys in aggregated).
        output_dir: If set and SAVE_FIGURES, save as PDF.
    """
    setup_ieee_style()
    if methods is None:
        methods = list(aggregated.keys())

    all_fracs = sorted({f for m in methods for f in aggregated[m]})
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    fig, ax = plt.subplots(figsize=(7, 2.5))
    width = 0.8 / len(methods)

    for mi, (method, color) in enumerate(zip(methods, colors)):
        positions = [i + (mi - len(methods) / 2 + 0.5) * width
                     for i in range(len(all_fracs))]
        data = [aggregated[method].get(f, []) for f in all_fracs]
        data = [d for d in data if d]
        if not data:
            continue
        bp = ax.boxplot(data, positions=positions[:len(data)], widths=width * 0.9,
                        patch_artist=True, showfliers=False,
                        medianprops={'linewidth': 1.5, 'color': 'black'})
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        bp['boxes'][0].set_label(method)

    ax.set_xticks(range(len(all_fracs)))
    ax.set_xticklabels([f'{f:.2f}' if f < 1 else '1.0' for f in all_fracs])
    ax.set_xlabel('Training Sample Fraction')
    ax.set_ylabel('F1 (weighted)')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if output_dir and SAVE_FIGURES:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(output_dir) / 'sample_size_sensitivity.pdf', bbox_inches='tight')
    return fig


def plot_class_barycenters(barycenters: Dict[int, np.ndarray],
                            X_train: List[np.ndarray],
                            Y_train: np.ndarray,
                            idx_to_regime: Dict[int, str],
                            method_name: str,
                            n_samples: int = 10,
                            output_dir: Optional[str] = None):
    """
    One figure per class: training samples (faded) + barycenter (bold).

    Works for both raw data (shape T×D) and parameter data (shape T×1).
    Uses the mean across the feature dimension for plotting.

    Args:
        barycenters: {class_label: barycenter_array}
        X_train: List of training sample arrays
        Y_train: Training labels
        idx_to_regime: {label_int: regime_code}
        method_name: Used for file naming
        n_samples: Samples per class to overlay
        output_dir: If set and SAVE_FIGURES, figures are saved as PDF + tex
    """
    setup_ieee_style()

    fig_width = 8.27   # A4 width inches
    fig_height = 2.33  # 1/5 A4 height inches

    T = X_train[0].shape[0]
    time_axis = pd.date_range(start='2019-01-01', periods=T, freq='D')
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    method_suffix = (method_name.replace(' ', '_')
                                .replace('(', '').replace(')', '').replace('-', '_'))

    if output_dir:
        out_path = Path(output_dir) / 'class_barycenters'
        out_path.mkdir(parents=True, exist_ok=True)

    for class_label in sorted(barycenters.keys()):
        class_name = idx_to_regime[class_label]
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

        class_idx = [i for i, y in enumerate(Y_train) if y == class_label]
        rng = np.random.RandomState(42 + class_label)
        rng.shuffle(class_idx)

        barycenter = barycenters[class_label]
        bary_mean = np.mean(barycenter, axis=tuple(range(1, barycenter.ndim))) \
            if barycenter.ndim > 1 else barycenter

        all_vals = np.concatenate([
            (np.mean(X_train[i], axis=tuple(range(1, X_train[i].ndim)))
             if X_train[i].ndim > 1 else X_train[i])
            for i in class_idx[:n_samples]
        ])
        use_log = bool((all_vals > 0).all() and all_vals.min() > 0 and bary_mean.min() > 0)

        for i, idx in enumerate(class_idx[:n_samples]):
            sample = X_train[idx]
            sample_mean = np.mean(sample, axis=tuple(range(1, sample.ndim))) \
                if sample.ndim > 1 else sample
            if use_log:
                ax.semilogy(time_axis, sample_mean,
                            color=colors[i % len(colors)], alpha=0.4, linewidth=0.8)
            else:
                ax.plot(time_axis, sample_mean,
                        color=colors[i % len(colors)], alpha=0.4, linewidth=0.8)

        plot_fn = ax.semilogy if use_log else ax.plot
        plot_fn(time_axis, bary_mean, color='black', linewidth=2, label='Barycenter')
        ax.set_title(class_name, fontsize=8, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('λ' if use_log else 'normalized value')
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
        ax.tick_params(axis='x', rotation=0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=7)
        plt.tight_layout()

        if output_dir and SAVE_FIGURES:
            fname = out_path / f'barycenter_{class_name}_{method_suffix}'
            plt.savefig(str(fname) + '.pdf', bbox_inches='tight', dpi=300)
            try:
                import tikzplotlib
                tikzplotlib.save(str(fname) + '.tex')
            except ImportError:
                pass

        plt.close()


# =============================================================================
# RMSE study plots
# =============================================================================

def plot_rmse_vs_samples(aggregated: Dict,
                         output_dir: Optional[str] = None) -> plt.Figure:
    """
    Line plot with error bands: RMSE vs n_samples for each (estimator, method) pair.

    Args:
        aggregated: {(n_samples, estimator): {'mean_rmse_euclidean': float,
                                               'std_rmse_euclidean':  float,
                                               'mean_rmse_wasserstein': float,
                                               'std_rmse_wasserstein':  float}}
                    Built from to_dataframe(load_results(study_dir)) and groupby.
        output_dir: If set and SAVE_FIGURES, save as PDF.
    """
    setup_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), sharey=False)
    method_labels = {'euclidean': 'Euclidean params', 'wasserstein': 'Wasserstein SGD'}
    colors = {'mle': 'tab:blue', 'log_cumulant': 'tab:orange'}
    markers = {'mle': 'o', 'log_cumulant': 's'}

    estimators = sorted({est for (_, est) in aggregated})
    sample_sizes_all = sorted({ns for (ns, _) in aggregated})

    for ax, bary_method in zip(axes, ['euclidean', 'wasserstein']):
        for est in estimators:
            xs = [ns for ns in sample_sizes_all if (ns, est) in aggregated]
            ys = [aggregated[(ns, est)][f'mean_rmse_{bary_method}'] for ns in xs]
            errs = [aggregated[(ns, est)][f'std_rmse_{bary_method}'] for ns in xs]
            ax.errorbar(xs, ys, yerr=errs, label=est,
                        color=colors[est], marker=markers[est],
                        linewidth=1.2, markersize=4, capsize=3)
        ax.set_xscale('log')
        ax.set_xlabel('n_samples')
        ax.set_ylabel('RMSE')
        ax.set_title(method_labels[bary_method])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir and SAVE_FIGURES:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(output_dir) / 'rmse_vs_samples.pdf', bbox_inches='tight')
    return fig


# =============================================================================
# Interpolation plots
# =============================================================================

def plot_interpolation_sequence(interpolants: np.ndarray,
                                t_values: np.ndarray,
                                label_a: str,
                                label_b: str,
                                method: str,
                                output_path: Optional[str] = None) -> plt.Figure:
    """
    One subplot per interpolation step showing the geodesic from class A to class B.

    Args:
        interpolants: (K, T, D) array — stacked interpolant arrays.
        t_values:     (K,) array of t positions in [0, 1].
        label_a:      Name of the source class (t=0).
        label_b:      Name of the target class (t=1).
        method:       Method name for the title.
        output_path:  If set and SAVE_FIGURES, save as PDF.
    """
    setup_ieee_style()
    K = len(t_values)
    fig, axes = plt.subplots(1, K, figsize=(2.5 * K, 3), sharey=True)
    if K == 1:
        axes = [axes]

    cmap = plt.cm.RdYlGn
    for i, (ax, t, interp) in enumerate(zip(axes, t_values, interpolants)):
        mean_series = np.mean(interp, axis=-1) if interp.ndim > 1 else interp
        color = cmap(1.0 - t)
        ax.semilogy(np.arange(len(mean_series)), mean_series, color=color, linewidth=1.5)
        ax.set_title(f't={t:.2f}', fontsize=8)
        ax.set_xlabel('Time')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel('λ (log)')

    fig.suptitle(f'Geodesic interpolation: {label_a} → {label_b}  [{method}]',
                 fontsize=9, fontweight='bold')
    plt.tight_layout()

    if output_path and SAVE_FIGURES:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
    return fig
