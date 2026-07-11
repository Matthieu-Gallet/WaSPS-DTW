"""
Classification visualization functions.

This module provides functions for:
- Confusion matrix plots
- Barycenter visualization with samples
- Gamma sensitivity plots
- Sample size sensitivity plots
- K-fold results boxplots

All plots are designed for IEEE publications with Times New Roman fonts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from typing import Dict, List, Optional, Union, Sequence, Mapping
from pathlib import Path
from sklearn.metrics import confusion_matrix
import warnings

# Try to import seaborn for enhanced visualizations
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    warnings.warn("Seaborn not installed. Some plots may have reduced quality.")


# =============================================================================
# Global settings for IEEE-compatible plots
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
# Hydro-regime name mapping and plot constants
# =============================================================================

REGIMES = {
    "Pluvial modérément contrasté": {"code": "PM"},
    "Pluvial contrasté":             {"code": "PC"},
    "Pluvio-nival":                  {"code": "PN"},
    "Nivo-pluvial":                  {"code": "NP"},
    "Nival & nivo-glaciaire":        {"code": "NN"},
    "Nival":                         {"code": "NG"},
    "nivo-glaciaire":                {"code": "NG"},
}

# First-match reverse map: short code → full French name
_CODE_TO_NAME: Dict[str, str] = {}
for _regime_name, _regime_info in REGIMES.items():
    _code = _regime_info["code"]
    if _code not in _CODE_TO_NAME:
        _CODE_TO_NAME[_code] = _regime_name

# English names used in publication figures
_CODE_TO_NAME_EN: Dict[str, str] = {
    "PM": "Moderately contrasted pluvial",
    "PC": "Contrasted pluvial",
    "PN": "Pluvio-nival",
    "NP": "Nivo-pluvial",
    "NN": "Nival & nivo-glacial",
    "NG": "Nival",
}

# Methods displayed in class-pair plots (in order)
_PAIR_METHODS = ['euclidean_raw', 'euclidean_params', 'wasserstein_params']

_METHOD_COLORS = {
    'euclidean_raw':      '#1f77b4',  # blue
    'euclidean_params':   '#ff7f0e',  # orange
    'wasserstein_params': '#2ca02c',  # green
}

_METHOD_LABELS = {
    'euclidean_raw':      'Euclidean (Raw Data)',
    'euclidean_params':   'Euclidean (Parameters)',
    'wasserstein_params': 'Wasserstein (Parameters)',
}

# Up to 20 distinct class colours
_CLASS_PALETTE = list(plt.cm.tab20.colors)


# =============================================================================
# Confusion matrix plots
# =============================================================================

def plot_confusion_matrices(results: Dict, Y_test: np.ndarray, 
                            idx_to_regime: Dict[int, str],
                            output_dir: str = None, save_pdf: bool = True):
    """
    Plot confusion matrices for available classification methods.
    
    Args:
        results: Dictionary with classification results
        Y_test: True test labels
        idx_to_regime: Mapping from label index to regime code
        output_dir: Output directory for saving
        save_pdf: Whether to save as PDF
    """
    setup_ieee_style()
    
    labels = sorted(idx_to_regime.keys())
    target_names = [idx_to_regime[i] for i in labels]
    method_candidates = [
        ('euclidean_raw', 'Soft-DTW Euclidean\n(Raw Data)'),
        ('euclidean_params', 'Soft-DTW Euclidean\n(Parameters)'),
        ('wasserstein_params', 'Soft-DTW Wasserstein\n(Parameters)'),
        ('wasserstein_weibull', 'WaSPS-DTW Weibull\n(Wasserstein)'),
        ('lstm_raw', 'LSTM Barycenter\n(Raw Data)'),
        ('ot_regul_raw', 'Regularized OT STA\n(Raw Data)'),
        ('shapelets_euclidean_raw', 'Learning Shapelets\nEuclidean Raw'),
        ('shapelets_euclidean_params', 'Learning Shapelets\nEuclidean Params'),
        ('shapelets_wasserstein_params', 'Learning Shapelets\nWasserstein Params')
    ]
    methods = [(k, n) for k, n in method_candidates if k in results]
    if not methods:
        return
    
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(2.15 * n_methods + 0.8, 2.3))
    if n_methods == 1:
        axes = [axes]
    
    for idx, (method_key, method_name) in enumerate(methods):
        Y_pred = results[method_key]['predictions']
        cm = confusion_matrix(Y_test, Y_pred, labels=labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.where(row_sums > 0,
                                  cm.astype('float') / np.maximum(row_sums, 1),
                                  0.0)
        
        ax = axes[idx]
        
        # Plot heatmap
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        
        # Add text annotations
        thresh = 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'white' if cm_normalized[i, j] > thresh else 'black'
                ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', color=color, fontsize=8)
        
        ax.set_xticks(np.arange(len(target_names)))
        ax.set_yticks(np.arange(len(target_names)))
        ax.set_xticklabels(target_names, rotation=45, ha='right')
        ax.set_yticklabels(target_names)
        
        ax.set_title(method_name, fontsize=9)
        
        if idx == 0:
            ax.set_ylabel('True Class')
        ax.set_xlabel('Predicted Class')
        
        # Add F1 score below title
        f1 = results[method_key]['f1_weighted']
        ax.text(0.5, -0.35, f'F1={f1:.3f}', transform=ax.transAxes, 
                ha='center', fontsize=8)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if save_pdf:
            plt.savefig(output_path / "confusion_matrices.pdf", bbox_inches='tight', dpi=300)
    
    plt.close()


# =============================================================================
# Barycenter visualization
# =============================================================================

def plot_barycenter_with_samples(results: Dict, X_train_raw: List[np.ndarray],
                                  X_train_params: List[np.ndarray],
                                  Y_train: np.ndarray, idx_to_regime: Dict[int, str],
                                  output_dir: str = None, save_pdf: bool = True,
                                  param_names: List[str] = None,
                                  log_scale: bool = True,
                                  time_axis=None):
    """
    Plot barycenters with training samples for each class.
    
    Args:
        results: Dictionary with classification results (containing barycenters)
        X_train_raw: Training samples (raw data)
        X_train_params: Training samples (parameters)
        Y_train: Training labels
        idx_to_regime: Mapping from label index to regime code
        output_dir: Output directory for saving
        save_pdf: Whether to save as PDF
        param_names: Names of parameters (default: ['λ', 'β', 'γ', 'δ'])
    """
    setup_ieee_style()
    
    if param_names is None:
        param_names = ['λ', 'β', 'γ', 'δ']
    
    unique_classes = np.unique(Y_train)
    n_classes = len(unique_classes)
    n_params = X_train_params[0].shape[1] if len(X_train_params[0].shape) > 1 else 1
    
    # Create figure with temporal axis
    fig, axes = plt.subplots(n_classes, n_params, figsize=(7, 1.8 * n_classes))
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    
    T = X_train_params[0].shape[0]
    if time_axis is None:
        time_axis = np.arange(T)
    use_date_fmt = isinstance(time_axis, pd.DatetimeIndex)
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))
    
    # Fix seed for consistent sample selection
    np.random.seed(42)
    
    for class_idx, class_label in enumerate(unique_classes):
        class_name = idx_to_regime[class_label]
        
        # Get samples for this class
        class_indices = [i for i in range(len(X_train_params)) if Y_train[i] == class_label]
        class_params = [X_train_params[i] for i in class_indices]
        
        # Get barycenter — prefer Wasserstein (Weibull or exponential), fall back to Euclidean
        bary_key = None
        for key in ('wasserstein_weibull', 'wasserstein_params', 'euclidean_params'):
            if (key in results and 'barycenters' in results.get(key, {})
                    and class_label in results[key]['barycenters']):
                bary_key = key
                break
        if bary_key is None:
            continue
        barycenter = results[bary_key]['barycenters'][class_label]
        
        for param_idx in range(n_params):
            ax = axes[class_idx, param_idx]
            
            plot_fn = ax.semilogy if log_scale else ax.plot
            for sample_params in class_params[:20]:
                plot_fn(time_axis, sample_params[:, param_idx],
                        color=colors[class_idx], alpha=0.2, linewidth=0.5)

            plot_fn(time_axis, barycenter[:, param_idx],
                    color='black', linewidth=1.5, label='Barycenter')

            if class_idx == 0:
                ax.set_title(param_names[param_idx], fontsize=10)

            if param_idx == 0:
                ax.set_ylabel(class_name, fontsize=10)

            if class_idx == n_classes - 1:
                if use_date_fmt:
                    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
                ax.tick_params(axis='x', rotation=0)
            else:
                ax.set_xticklabels([])
            
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if save_pdf:
            plt.savefig(output_path / "barycenters_with_samples.pdf", bbox_inches='tight', dpi=300)
    
    plt.close()


# =============================================================================
# Sensitivity analysis plots
# =============================================================================

def plot_gamma_sensitivity(results: Dict, output_dir: str = None, save_pdf: bool = True):
    """
    Plot gamma sensitivity analysis results with boxplots.
    
    Args:
        results: Dictionary with gamma sensitivity results (keyed by gamma values)
        output_dir: Output directory for saving
        save_pdf: Whether to save as PDF
    """
    setup_ieee_style()
    
    # Extract gamma values and metrics from results
    gamma_values = sorted(results.keys())
    n_gammas = len(gamma_values)
    
    # Method names and colors
    methods = [
        ('euclidean_raw', 'Soft-DTW Euclidean (Raw)', 'tab:blue'),
        ('euclidean_params', 'Soft-DTW Euclidean (Params)', 'tab:orange'),
        ('wasserstein_params', 'Soft-DTW Wasserstein (Params)', 'tab:green')
    ]
    
    fig, ax = plt.subplots(figsize=(7, 2.5))
    
    # Prepare data for boxplots - organize by gamma first, then method
    positions = []
    data_to_plot = []
    colors_list = []
    
    for gamma_idx, gamma in enumerate(gamma_values):
        for method_idx, (method_key, method_name, color) in enumerate(methods):
            # Get all F1 scores for this gamma and method
            if 'all_f1_weighted' in results[gamma][method_key]:
                f1_scores = results[gamma][method_key]['all_f1_weighted']
            else:
                # Fallback if all scores not available
                f1_scores = [results[gamma][method_key]['f1_weighted_mean']]
            
            pos = gamma_idx * (len(methods) + 1) + method_idx
            positions.append(pos)
            data_to_plot.append(f1_scores)
            colors_list.append(color)
    
    # Create boxplots
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=False, medianprops={'linewidth': 1.5, 'color': 'black'})
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Set x-axis
    center_positions = [i * (len(methods) + 1) + 1 for i in range(n_gammas)]
    ax.set_xticks(center_positions)
    ax.set_xticklabels([f'{g:.0e}' if g < 0.1 or g > 10 else f'{g:.1f}' for g in gamma_values])
    ax.set_xlabel('Gamma (γ)')
    ax.set_ylabel('F1 Score (weighted)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Calculate dynamic ylim based on data
    all_f1_values = []
    for data in data_to_plot:
        all_f1_values.extend(data)
    f1_min = min(all_f1_values)
    f1_max = max(all_f1_values)
    ax.set_ylim(0.75 * f1_min, 1.25 * f1_max)
    
    # Add legend horizontally below the figure
    legend_handles = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7) 
                     for _, _, color in methods]
    legend_labels = [name for _, name, _ in methods]
    ax.legend(legend_handles, legend_labels, loc='upper center', 
             bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=7, frameon=False)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if save_pdf:
            plt.savefig(output_path / "gamma_sensitivity.pdf", bbox_inches='tight', dpi=300)
    
    plt.close()


def plot_sample_size_sensitivity(results: Dict, output_dir: str = None, save_pdf: bool = True):
    """
    Plot sample size sensitivity analysis results with boxplots.
    
    Args:
        results: Dictionary with sample size sensitivity results (keyed by sample sizes)
        output_dir: Output directory for saving
        save_pdf: Whether to save as PDF
    """
    setup_ieee_style()
    
    # Extract sample sizes and metrics from results
    sample_sizes = sorted(results.keys())
    n_sizes = len(sample_sizes)
    
    # Method names and colors
    methods = [
        ('euclidean_raw', 'Soft-DTW Euclidean (Raw)', 'tab:blue'),
        ('euclidean_params', 'Soft-DTW Euclidean (Params)', 'tab:orange'),
        ('wasserstein_params', 'Soft-DTW Wasserstein (Params)', 'tab:green')
    ]
    
    fig, ax = plt.subplots(figsize=(7, 2.5))
    
    # Prepare data for boxplots - organize by sample size first, then method
    positions = []
    data_to_plot = []
    colors_list = []
    
    for size_idx, size in enumerate(sample_sizes):
        for method_idx, (method_key, method_name, color) in enumerate(methods):
            # Get all F1 scores for this size and method
            if 'all_f1_weighted' in results[size][method_key]:
                f1_scores = results[size][method_key]['all_f1_weighted']
            else:
                # Fallback if all scores not available
                f1_scores = [results[size][method_key]['f1_weighted_mean']]
            
            pos = size_idx * (len(methods) + 1) + method_idx
            positions.append(pos)
            data_to_plot.append(f1_scores)
            colors_list.append(color)
    
    # Create boxplots
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=False, medianprops={'linewidth': 1.5, 'color': 'black'})
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Set x-axis
    center_positions = [i * (len(methods) + 1) + 1 for i in range(n_sizes)]
    ax.set_xticks(center_positions)
    ax.set_xticklabels([f'{s:.2f}' if s < 1 else f'{int(s)}' for s in sample_sizes])
    ax.set_xlabel('Training Sample Fraction')
    ax.set_ylabel('F1 Score (weighted)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Calculate dynamic ylim based on data
    all_f1_values = []
    for data in data_to_plot:
        all_f1_values.extend(data)
    f1_min = min(all_f1_values)
    f1_max = max(all_f1_values)
    ax.set_ylim(0.75 * f1_min, 1.25 * f1_max)
    
    # Add legend horizontally below the figure
    legend_handles = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7) 
                     for _, _, color in methods]
    legend_labels = [name for _, name, _ in methods]
    ax.legend(legend_handles, legend_labels, loc='upper center', 
             bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=7, frameon=False)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if save_pdf:
            plt.savefig(output_path / "sample_size_sensitivity.pdf", bbox_inches='tight', dpi=300)
    
    plt.close()


# =============================================================================
# K-fold boxplot visualization
# =============================================================================

def plot_kfold_boxplots(aggregated_results: Dict, idx_to_regime: Dict[int, str],
                        output_dir: str = None, save_pdf: bool = True):
    """
    Plot boxplots of F1 scores across k-fold cross-validation.
    
    Args:
        aggregated_results: Aggregated results from k-fold CV
        idx_to_regime: Mapping from label index to regime code
        output_dir: Output directory for saving
        save_pdf: Whether to save as PDF
    """
    setup_ieee_style()
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # Prepare data for boxplot
    method_names = [
        'Soft-DTW\nEuclidean\n(Raw)',
        'Soft-DTW\nEuclidean\n(Params)',
        'Soft-DTW\nWasserstein\n(Params)'
    ]
    method_keys = ['euclidean_raw', 'euclidean_params', 'wasserstein_params']
    
    data = []
    for method_key in method_keys:
        if method_key in aggregated_results:
            data.append(aggregated_results[method_key]['all_f1_weighted'])
        else:
            data.append([])
    
    # Create boxplot
    bp = ax.boxplot(data, labels=method_names, patch_artist=True)
    
    # Color the boxes
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('F1 Score (weighted)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    # Add mean values as text
    for i, method_key in enumerate(method_keys):
        if method_key in aggregated_results:
            mean_val = aggregated_results[method_key]['f1_weighted_mean']
            ax.text(i + 1, mean_val + 0.02, f'{mean_val:.3f}', ha='center', fontsize=7)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if save_pdf:
            plt.savefig(output_path / "kfold_boxplot.pdf", bbox_inches='tight', dpi=300)
    
    plt.close()


# =============================================================================
# Combined summary figure
# =============================================================================

def plot_summary_figure(results: Dict, Y_test: np.ndarray, idx_to_regime: Dict[int, str],
                        gamma_results: Dict = None, sample_results: Dict = None,
                        output_dir: str = None, save_pdf: bool = True):
    """
    Create a combined summary figure with confusion matrices and sensitivity plots.
    
    Args:
        results: Dictionary with classification results
        Y_test: True test labels
        idx_to_regime: Mapping from label index to regime code
        gamma_results: Optional gamma sensitivity results
        sample_results: Optional sample size sensitivity results
        output_dir: Output directory for saving
        save_pdf: Whether to save as PDF
    """
    setup_ieee_style()
    
    # Determine layout based on available results
    n_rows = 1
    if gamma_results:
        n_rows += 1
    if sample_results:
        n_rows += 1
    
    fig = plt.figure(figsize=(7, 2.5 * n_rows))
    
    # Row 1: Confusion matrices
    for idx, (method_key, method_name) in enumerate([
        ('euclidean_raw', 'Soft-DTW Euclidean (Raw)'),
        ('euclidean_params', 'Soft-DTW Euclidean (Params)'),
        ('wasserstein_params', 'Soft-DTW Wasserstein (Params)')
    ]):
        if method_key not in results:
            continue
        
        ax = fig.add_subplot(n_rows, 3, idx + 1)
        
        Y_pred = results[method_key]['predictions']
        cm = confusion_matrix(Y_test, Y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        target_names = [idx_to_regime[i] for i in sorted(idx_to_regime.keys())]
        
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        
        thresh = 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'white' if cm_normalized[i, j] > thresh else 'black'
                ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', color=color, fontsize=7)
        
        ax.set_xticks(np.arange(len(target_names)))
        ax.set_yticks(np.arange(len(target_names)))
        ax.set_xticklabels(target_names, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(target_names, fontsize=7)
        
        f1 = results[method_key]['f1_weighted']
        ax.set_title(f'{method_name}\nF1={f1:.3f}', fontsize=8)
        
        if idx == 0:
            ax.set_ylabel('True Class')
    
    # Row 2: Gamma sensitivity (if available)
    current_row = 2
    if gamma_results:
        ax = fig.add_subplot(n_rows, 2, 2 * current_row - 1)
        gamma_values = gamma_results['gamma_values']
        
        for method_key, method_name, color in [
            ('euclidean_raw', 'Euclidean (Raw)', 'tab:blue'),
            ('euclidean_params', 'Euclidean (Params)', 'tab:orange'),
            ('wasserstein_params', 'Wasserstein (Params)', 'tab:green')
        ]:
            if method_key in gamma_results:
                ax.plot(gamma_values, gamma_results[method_key]['f1_weighted'], 
                       label=method_name, color=color, marker='o', markersize=3)
        
        ax.set_xscale('log')
        ax.set_xlabel('Gamma (γ)')
        ax.set_ylabel('F1 Score')
        ax.set_title('Gamma Sensitivity')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        current_row += 1
    
    # Row 3: Sample size sensitivity (if available)
    if sample_results:
        ax = fig.add_subplot(n_rows, 2, 2 * current_row - 1)
        sample_sizes = sample_results['sample_sizes']
        
        for method_key, method_name, color in [
            ('euclidean_raw', 'Euclidean (Raw)', 'tab:blue'),
            ('euclidean_params', 'Euclidean (Params)', 'tab:orange'),
            ('wasserstein_params', 'Wasserstein (Params)', 'tab:green')
        ]:
            if method_key in sample_results:
                ax.errorbar(sample_sizes, sample_results[method_key]['f1_weighted_mean'],
                           yerr=sample_results[method_key]['f1_weighted_std'],
                           label=method_name, color=color, marker='o', markersize=3, capsize=2)
        
        ax.set_xlabel('Training Samples per Class')
        ax.set_ylabel('F1 Score')
        ax.set_title('Sample Size Sensitivity')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if save_pdf:
            plt.savefig(output_path / "summary_figure.pdf", bbox_inches='tight', dpi=300)
    
    plt.close()


# =============================================================================
# Class pair barycenter plots (A4 format)
# =============================================================================

def plot_class_pair_barycenters(barycenters_by_method: Dict[str, Dict[int, np.ndarray]],
                                 X_train_raw: List[np.ndarray],
                                 X_train_params: List[np.ndarray],
                                 Y_train: np.ndarray,
                                 idx_to_regime: Dict[int, str],
                                 output_dir: str = None,
                                 save_pdf: bool = True,
                                 n_samples: int = 10,
                                 show_legend: Union[bool, Sequence[bool], Mapping[Union[int, str], bool]] = True):
    """
    Plot one figure per class (all methods overlaid) with training sample traces.

    This function keeps its historical name for backward compatibility, but now
    creates individual class figures instead of class-pair figures.

    Figures are saved under ``<output_dir>/class_pairs/`` and exported in PDF
    only when ``save_pdf`` is True.

    Args:
        barycenters_by_method: Dict {method_key: {class_label: barycenter}}.
        X_train_raw: Unused (kept for API compatibility).
        X_train_params: List of parameter training arrays, each shape (T, 1).
        Y_train: Integer class labels for every training sample.
        idx_to_regime: {class_label: regime_code}, e.g. {0: 'PM', 1: 'PC'}.
        output_dir: Base output directory; figures saved under ``<output_dir>/class_pairs/``.
        save_pdf: Whether to save figures as PDF.
        n_samples: Number of sample traces to draw per class.
        show_legend: Legend visibility control.
            - bool: same value for all classes.
            - Sequence[bool]: one flag per class (ordered by sorted class labels).
            - Mapping[int|str, bool]: keyed by class label or regime code.
    """
    setup_ieee_style()

    def _sanitize_filename(text: str) -> str:
        safe = ''.join(ch if (ch.isalnum() or ch in ['_', '-']) else '_' for ch in text.strip())
        while '__' in safe:
            safe = safe.replace('__', '_')
        return safe.strip('_') or 'class'

    all_classes = sorted(next(iter(barycenters_by_method.values())).keys())
    T = X_train_params[0].shape[0]
    time_axis = pd.date_range(start='2019-01-01', periods=T, freq='D')

    legend_map = None
    legend_seq = None
    if isinstance(show_legend, Mapping):
        legend_map = dict(show_legend)
    elif isinstance(show_legend, (list, tuple, np.ndarray)) and not isinstance(show_legend, (str, bytes)):
        legend_seq = [bool(v) for v in show_legend]

    if output_dir:
        output_path = Path(output_dir) / "class_pairs"
        output_path.mkdir(parents=True, exist_ok=True)

    fig_width = 8./3.
    fig_height = 4./3.

    for class_label in all_classes:
        code = idx_to_regime.get(class_label, str(class_label))
        class_name = _CODE_TO_NAME_EN.get(code, code)

        if legend_map is not None:
            class_show_legend = bool(
                legend_map.get(class_label, legend_map.get(str(class_label), legend_map.get(code, True)))
            )
        elif legend_seq is not None:
            class_pos = all_classes.index(class_label)
            class_show_legend = legend_seq[class_pos] if class_pos < len(legend_seq) else True
        else:
            class_show_legend = bool(show_legend)

        class_indices = [i for i in range(len(X_train_params)) if Y_train[i] == class_label]
        rng_pre = np.random.RandomState(42 + class_label)
        rng_pre.shuffle(class_indices)

        y_sample_vals = []
        for idx in class_indices[:n_samples]:
            s = X_train_params[idx]
            vals = 1.0 / np.maximum(s[:, 0], 1e-10)
            y_sample_vals.append(vals)

        if y_sample_vals:
            all_sv = np.concatenate(y_sample_vals)
            pos = all_sv[all_sv > 0]
            y_lo = float(np.nanpercentile(pos, 1)) * 0.85
            y_hi = float(np.nanpercentile(pos, 99)) * 1.15
            y_lo = max(y_lo, 1e-3)
        else:
            y_lo, y_hi = 0.1, 1000.0

        _lw_sp = min(0.2, 0.7 * (5.0 / max(n_samples, 5)) ** 0.23)
        _al_sp = min(0.10, 0.5 * (5.0 / max(n_samples, 5)) ** 0.65)

        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        fig.subplots_adjust(bottom=0.28 if class_show_legend else 0.16, top=0.95, left=0.10, right=0.99)

        for idx in class_indices[:n_samples]:
            s = X_train_params[idx]
            ax.plot(time_axis, 1.0 / np.maximum(s[:, 0], 1e-10),
                    color='black', alpha=_al_sp, linewidth=_lw_sp)

        for method_key in _PAIR_METHODS:
            if method_key not in barycenters_by_method:
                continue
            bary = barycenters_by_method[method_key].get(class_label)
            if bary is None:
                continue

            if method_key == 'euclidean_raw':
                if bary.ndim > 1:
                    b = np.mean(bary, axis=tuple(range(1, bary.ndim)))
                else:
                    b = bary.ravel()
                b_plot = np.maximum(b, 1e-10)
            else:
                b_plot = 1.0 / np.maximum(bary[:, 0], 1e-10)

            ax.plot(time_axis, b_plot,
                    color=_METHOD_COLORS[method_key],
                    linewidth=1.2,
                    zorder=5)

        ax.set_yscale('log')
        ax.set_ylim(0.25 * y_lo, 10 * y_hi)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
        ax.tick_params(axis='x', rotation=0, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.set_ylabel(r'$\lambda\,(m^3/s)$', fontsize=8)
        ax.set_xlabel('Time', fontsize=8)
        ax.grid(True, alpha=0.3)

        if class_show_legend:
            legend_handles = [
                plt.Line2D([0], [0],
                           color=_METHOD_COLORS[m],
                           linewidth=1.2,
                           label=_METHOD_LABELS[m])
                for m in _PAIR_METHODS
                if m in barycenters_by_method
            ]
            legend_handles.append(
                plt.Line2D([0], [0], color='black', lw=0.8, alpha=0.5, label='Samples')
            )
            fig.legend(
                handles=legend_handles,
                loc='lower center',
                ncol=len(legend_handles),
                fontsize=6.5,
                bbox_to_anchor=(0.5, 0.03),
                frameon=True,
            )

        if output_dir and save_pdf:
            filename = _sanitize_filename(class_name)
            plt.savefig(output_path / f"{filename}.pdf", bbox_inches='tight', dpi=300)

        plt.close()


# =============================================================================
# CPAZMaL sensitivity line plots
# =============================================================================

def plot_cpazmal_sensitivity(df: pd.DataFrame, x_col: str, xlabel: str,
                              output_dir: str, filename: str,
                              x_log: bool = False):
    """
    Generic line plot for CPAZMaL sensitivity sub-experiments.

    One line per method (euclidean_raw, euclidean_params, wasserstein_weibull),
    with shaded ±1-std band when 'f1_weighted_std' / 'f1_macro_std' columns exist.

    Parameters
    ----------
    df : DataFrame with columns [x_col, method, f1_weighted_mean, f1_macro_mean,
         optionally f1_weighted_std, f1_macro_std].
    x_col : column to use as the x-axis.
    xlabel : human-readable x-axis label.
    output_dir : directory where .png and .pdf are saved.
    filename : base file name without extension.
    x_log : if True, use log-scale x-axis.
    """
    setup_ieee_style()

    method_styles = {
        'euclidean_raw':       {'color': '#1f77b4', 'ls': '--',  'label': 'SDTW Euclidean (Raw)'},
        'euclidean_params':    {'color': '#ff7f0e', 'ls': '-.',  'label': 'SDTW Euclidean (Weibull params)'},
        'wasserstein_weibull': {'color': '#2ca02c', 'ls': '-',   'label': 'WaSPS-DTW Weibull'},
    }

    has_std_w = 'f1_weighted_std' in df.columns
    has_std_m = 'f1_macro_std' in df.columns

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    for method, style in method_styles.items():
        sub = df[df['method'] == method].sort_values(x_col)
        if sub.empty:
            continue
        for ax, (metric, std_col, has_std) in zip(
            axes,
            [('f1_weighted_mean', 'f1_weighted_std', has_std_w),
             ('f1_macro_mean',    'f1_macro_std',    has_std_m)],
        ):
            ax.plot(sub[x_col], sub[metric],
                    color=style['color'], ls=style['ls'],
                    marker='o', ms=5, label=style['label'])
            if has_std and std_col in sub.columns:
                ax.fill_between(sub[x_col],
                                sub[metric] - sub[std_col],
                                sub[metric] + sub[std_col],
                                color=style['color'], alpha=0.15)

    for ax, title in zip(axes, ['F1 (weighted)', 'F1 (macro)']):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if x_log:
            ax.set_xscale('log')

    plt.tight_layout()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{filename}.png", dpi=150, bbox_inches='tight')
    fig.savefig(out / f"{filename}.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {out / filename}.png")


# =============================================================================
# Multi-gamma barycenter comparison plots
# =============================================================================

def plot_barycenters_gamma_comparison(barycenters_by_gamma: dict,
                                       idx_to_class: dict,
                                       output_dir: str,
                                       param_names: list = None):
    """
    For each class: one figure comparing barycenters from 3 methods at multiple γ values.

    Layout (3 rows per class figure):
      Row 0 — Euclidean (Raw): mean pixel amplitude vs time, one line per γ.
      Row 1 — k (shape):  euclidean_params + wasserstein_weibull, one line per method×γ.
      Row 2 — λ (scale):  same.

    Line style encodes γ: first gamma → dashed, second gamma → solid.
    Color encodes method: orange = euclidean_params, green = wasserstein_weibull,
                          blue = euclidean_raw.

    Parameters
    ----------
    barycenters_by_gamma : {gamma: {class_label: {method_key: array}}}
    idx_to_class : {int label → str class name}
    output_dir : directory where per-class PDF files are saved.
    param_names : list of 2 strings for the 2 Weibull params (default: ['k', 'λ']).
    """
    setup_ieee_style()

    if param_names is None:
        param_names = ['k (shape)', 'λ (scale)']

    gammas   = sorted(barycenters_by_gamma.keys())
    ls_map   = {gammas[i]: ls for i, ls in enumerate(['--', '-', ':', '-.'][:len(gammas)])}

    method_colors = {
        'euclidean_raw':       '#1f77b4',  # blue
        'euclidean_params':    '#ff7f0e',  # orange
        'wasserstein_weibull': '#2ca02c',  # green
    }
    method_labels = {
        'euclidean_raw':       'Euclidean (Raw)',
        'euclidean_params':    'Euclidean (params)',
        'wasserstein_weibull': 'WaSPS-DTW Weibull',
    }

    # Determine class set from the first gamma entry
    first_gamma_data = barycenters_by_gamma[gammas[0]]
    classes = sorted(first_gamma_data.keys())

    out = Path(output_dir) / 'barycenter_gamma_cmp'
    out.mkdir(parents=True, exist_ok=True)

    for cls in classes:
        cls_name = idx_to_class.get(cls, str(cls))

        # Infer T from the first available barycenter
        T = None
        for g in gammas:
            for method in ('wasserstein_weibull', 'euclidean_params', 'euclidean_raw'):
                bary = barycenters_by_gamma[g].get(cls, {}).get(method)
                if bary is not None:
                    T = bary.shape[0]
                    break
            if T is not None:
                break
        if T is None:
            continue
        t_axis = np.arange(T)

        fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
        fig.suptitle(f'Barycenters — {cls_name}', fontsize=10, fontweight='bold')

        # ── Row 0: euclidean_raw mean amplitude ──────────────────────────────
        ax0 = axes[0]
        for g in gammas:
            bary = barycenters_by_gamma[g].get(cls, {}).get('euclidean_raw')
            if bary is None:
                continue
            mean_amp = np.nanmean(bary, axis=1)  # (T,) mean over W² pixels
            ax0.plot(t_axis, mean_amp,
                     color=method_colors['euclidean_raw'],
                     ls=ls_map[g], lw=1.2,
                     label=f'γ={g:.0e}')
        ax0.set_ylabel('Mean amplitude\n(Euclidean raw)', fontsize=8)
        ax0.legend(fontsize=7, loc='upper right')
        ax0.grid(True, alpha=0.3)

        # ── Rows 1-2: Weibull params from param-based methods ────────────────
        for param_idx, (ax, pname) in enumerate(zip(axes[1:], param_names)):
            for method in ('euclidean_params', 'wasserstein_weibull'):
                for g in gammas:
                    bary = barycenters_by_gamma[g].get(cls, {}).get(method)
                    if bary is None:
                        continue
                    ax.plot(t_axis, bary[:, param_idx],
                            color=method_colors[method],
                            ls=ls_map[g], lw=1.2,
                            label=f'{method_labels[method]}  γ={g:.0e}')
            ax.set_ylabel(pname, fontsize=8)
            ax.legend(fontsize=6, loc='upper right', ncol=2)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time index (date)', fontsize=8)
        plt.tight_layout()

        fname = ''.join(c if c.isalnum() or c in '_-' else '_' for c in cls_name)
        plt.savefig(out / f'barycenter_gamma_{fname}.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(out / f'barycenter_gamma_{fname}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  Multi-gamma barycenter plots saved to {out}/")


# =============================================================================
# Debug: samples histogram + fitted-PDF overlay (new in JAX branch)
# =============================================================================

def plot_samples_with_fitted_pdf(
    raw_series: List[np.ndarray],
    family: str,
    class_label: int,
    class_name: str,
    output_dir: str = None,
    n_timesteps: int = 3,
    methods: tuple = ('mle', 'log_cumulant'),
    save_pdf: bool = True,
) -> None:
    """Histogram + fitted-PDF overlay for one class at selected timesteps.

    Pools raw pixel/sample values from all training series of the class at
    ``n_timesteps`` evenly-spaced timesteps, draws a normalised histogram, and
    overlays fitted PDFs for each requested estimation method (MLE and/or
    log-cumulant).  Useful for diagnosing:
    - Whether the parametric family (exponential / Weibull) fits the data.
    - Whether MLE and log-cumulant agree (divergence → data poorly modelled).
    - Timesteps with very few valid samples (degenerate fit).

    Args:
        raw_series:  List of (T, N) raw sample arrays for one class.
        family:      'exponential' or 'weibull'.
        class_label: Integer class label (used for seed only).
        class_name:  Human-readable name (title and filename).
        output_dir:  Directory to save PDF; None = display only.
        n_timesteps: Number of timesteps to inspect (evenly spaced over T).
        methods:     Estimation methods to overlay ('mle', 'log_cumulant').
        save_pdf:    Save as PDF when ``output_dir`` is set.
    """
    from scipy.stats import expon, weibull_min
    try:
        import distributions as _dists
        from data.preprocess import clean_series as _clean
    except ImportError:
        warnings.warn(
            "plot_samples_with_fitted_pdf: cannot import 'distributions' / "
            "'data.preprocess' — ensure src/ is on sys.path."
        )
        return

    setup_ieee_style()

    if not raw_series:
        return

    T = raw_series[0].shape[0]
    dist = _dists.get(family)

    # Evenly spaced timesteps (0-based), deduplicated
    if n_timesteps >= T:
        t_indices = list(range(T))
    else:
        t_indices = [int(round((T - 1) * i / max(n_timesteps - 1, 1)))
                     for i in range(n_timesteps)]
        t_indices = sorted(set(t_indices))[:n_timesteps]

    _meth_colors = {'mle': '#e74c3c', 'log_cumulant': '#2980b9'}
    _meth_labels = {'mle': 'MLE', 'log_cumulant': 'Log-cumulant'}
    _meth_ls     = {'mle': '-',   'log_cumulant': '--'}

    n = len(t_indices)
    fig, axes = plt.subplots(n, 1, figsize=(5.0, 2.5 * n), squeeze=False)
    fig.suptitle(f'{class_name}  —  samples + fitted PDF', fontsize=9, fontweight='bold')

    for row_idx, t in enumerate(t_indices):
        ax = axes[row_idx, 0]

        # Pool and clean samples from all training series at timestep t
        pooled = np.concatenate([_clean(s[t]) for s in raw_series])
        if len(pooled) < 5:
            ax.set_title(f't={t}  (too few valid samples: {len(pooled)})', fontsize=8)
            ax.axis('off')
            continue

        ax.hist(pooled, bins=20,
                density=True, color='#95a5a6', alpha=0.65,
                label=f'samples (n={len(pooled)})', edgecolor='none')

        x_lo = max(np.percentile(pooled, 1) * 0.5, 1e-6)
        x_hi = np.percentile(pooled, 99) * 1.5
        x_grid = np.linspace(x_lo, x_hi, 300)

        for meth in methods:
            try:
                params = dist.estimate(pooled, method=meth)
                if family == 'exponential':
                    beta = float(params)
                    pdf_vals = expon(scale=1.0 / beta).pdf(x_grid)
                else:  # weibull
                    k_val, lam_val = float(params[0]), float(params[1])
                    pdf_vals = weibull_min(c=k_val, scale=lam_val, loc=0).pdf(x_grid)
                ax.plot(x_grid, pdf_vals,
                        color=_meth_colors.get(meth, 'k'),
                        ls=_meth_ls.get(meth, '-'),
                        lw=1.3,
                        label=_meth_labels.get(meth, meth))
            except Exception as exc:
                warnings.warn(
                    f"plot_samples_with_fitted_pdf: {meth} failed at t={t}: {exc}"
                )

        ax.set_xlabel('Value', fontsize=7)
        ax.set_ylabel('Density', fontsize=7)
        ax.set_title(f't={t}  (n={len(pooled)})', fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    plt.tight_layout()

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        safe = ''.join(c if (c.isalnum() or c in '_-') else '_'
                       for c in class_name.strip())
        while '__' in safe:
            safe = safe.replace('__', '_')
        safe = safe.strip('_') or f'class{class_label}'
        if save_pdf:
            plt.savefig(out / f'samples_pdf_{safe}.pdf', bbox_inches='tight', dpi=300)

    plt.close()


# =============================================================================
# Debug: barycenter + sample traces per method (new in JAX branch)
# =============================================================================

def plot_barycenter_debug(
    barycenter: np.ndarray,
    class_series: List[np.ndarray],
    family: str,
    class_label: int,
    class_name: str,
    method: str,
    output_dir: str = None,
    param_names: List[str] = None,
    n_samples: int = 10,
    save_pdf: bool = True,
) -> None:
    """Barycenter + sample traces for one (class, method) pair.

    Produces one subplot per parameter (or one subplot for the spatial mean
    when the representation is raw).  Sample traces are drawn in thin
    semi-transparent black; the fitted barycenter is drawn in a bold colour.

    Args:
        barycenter:   Fitted barycenter, shape (T, n_params) for params-repr
                      or (T, N) for raw-repr.
        class_series: List of (T, n_params or N) training series for the class.
        family:       'exponential' or 'weibull'.
        class_label:  Integer class label (seed and filename suffix).
        class_name:   Human-readable class name (title and filename).
        method:       Method key ('wasps', 'eucl_params', 'eucl_raw', 'sta').
        output_dir:   Directory to save PDF; None = display only.
        param_names:  Parameter names for subplot titles.  Default: family-based.
        n_samples:    Max number of sample traces to overlay.
        save_pdf:     Save as PDF when ``output_dir`` is set.
    """
    setup_ieee_style()

    is_raw = method in ('eucl_raw', 'eucl_raw_nodiv', 'sta')
    T = barycenter.shape[0]
    t_axis = np.arange(T)

    # For exponential wasps/eucl_params, invert rate β → mean 1/β so the curve is in
    # the same λ (discharge, m³/s) units/trend as eucl_raw's raw amplitude, mirroring
    # the existing pattern in plot_class_pair_barycenters (this module, column 0 only —
    # exponential has a single parameter, so "column 0" and "all columns" coincide).
    invert_to_lambda = (family == 'exponential'
                        and method in ('wasps', 'wasps_nodiv', 'eucl_params', 'eucl_params_nodiv'))

    # Default param names
    if param_names is None:
        if is_raw:
            param_names = ['spatial mean (raw samples)']
        elif invert_to_lambda:
            param_names = ['λ  (1/β, mean discharge)']
        elif family == 'exponential':
            param_names = ['rate β  (1/β = mean)']
        else:  # weibull
            param_names = ['k (shape)', 'λ (scale)']

    if is_raw:
        n_plots = 1
        # Collapse spatial dimension → mean amplitude for each series / barycenter
        def _to_plot(arr):
            if arr.ndim == 1:
                return arr
            return np.nanmean(arr, axis=1)
        series_vals = [_to_plot(s) for s in class_series]
        bary_vals   = [_to_plot(barycenter)]
        y_labels    = param_names[:1]
    else:
        n_params = barycenter.shape[1]
        n_plots  = n_params
        # (sample traces are read directly from class_series inside the plotting loop
        # below, with the same inversion applied there — not from a precomputed list)
        bary_vals   = [barycenter[:, p] for p in range(n_plots)]
        if invert_to_lambda:
            bary_vals = [1.0 / np.maximum(v, 1e-10) for v in bary_vals]
        y_labels    = (param_names + ['param'])[:n_plots]

    # Barcenter colour — method-specific
    _bary_colors = {
        'wasps':       '#2ca02c',   # green
        'eucl_params': '#ff7f0e',   # orange
        'eucl_raw':    '#1f77b4',   # blue
        'sta':         '#9467bd',   # purple
    }
    bary_color = _bary_colors.get(method, '#d62728')

    rng = np.random.RandomState(42 + class_label)
    class_idx = list(range(len(class_series)))
    rng.shuffle(class_idx)
    sel_idx = class_idx[:n_samples]

    _lw_s = max(0.2, 0.7 * (5.0 / max(n_samples, 5)) ** 0.3)
    _al_s = max(0.10, 0.5 * (5.0 / max(n_samples, 5)) ** 0.5)

    fig, axes = plt.subplots(n_plots, 1,
                              figsize=(5.5, 2.2 * n_plots),
                              squeeze=False, sharex=True)
    fig.suptitle(f'{class_name}  [{method}]  —  barycenter + samples',
                 fontsize=9, fontweight='bold')

    for p_idx in range(n_plots):
        ax = axes[p_idx, 0]

        # Sample traces
        for i in sel_idx:
            if is_raw:
                vals = series_vals[i]
            else:
                vals = class_series[i][:, p_idx]
                if invert_to_lambda:
                    vals = 1.0 / np.maximum(vals, 1e-10)
            ax.plot(t_axis, vals,
                    color='black', alpha=_al_s, linewidth=_lw_s)

        # Barycenter trace
        bv = bary_vals[p_idx]
        ax.plot(t_axis, bv,
                color=bary_color, linewidth=2.0, zorder=5,
                label='Barycenter')

        ax.set_yscale('log')
        ax.set_ylabel(y_labels[p_idx], fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if p_idx == 0:
            ax.legend(fontsize=7, loc='upper right')

    axes[-1, 0].set_xlabel('Time index', fontsize=8)
    plt.tight_layout()

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        safe_cls = ''.join(c if (c.isalnum() or c in '_-') else '_'
                           for c in class_name.strip()).strip('_') or f'class{class_label}'
        safe_mth = method.replace(' ', '_')
        if save_pdf:
            plt.savefig(out / f'bary_{safe_cls}_{safe_mth}.pdf',
                        bbox_inches='tight', dpi=300)

    plt.close()


def plot_all_class_barycenters_grid(barycenters: Dict[int, np.ndarray],
                                     X_train: List[np.ndarray],
                                     Y_train: np.ndarray,
                                     idx_to_regime: Dict[int, str],
                                     method_name: str,
                                     output_dir: str = None,
                                     save_pdf: bool = True,
                                     n_samples: int = 10,
                                     param_names: List[str] = None,
                                     is_raw: bool = False):
    """
    Plot all class barycenters in a grid layout with training samples.
    
    Creates one figure per parameter with all classes in 2-column layout.
    Each figure is A4 width and 1/5 A4 height per row of classes.
    
    Args:
        barycenters: Dictionary mapping class labels to barycenters.
        X_train: Training samples (list of arrays), shape (T, n_features) each.
        Y_train: Training labels.
        idx_to_regime: Mapping from label index to regime code.
        method_name: Name of the method for the title.
        output_dir: Output directory for saving.
        save_pdf: Whether to save as PDF.
        n_samples: Number of samples to plot per class (default: 10).
        param_names: Names of parameters (default: ['λ', 'β', 'γ', 'δ']).
        is_raw: If True, X_train contains raw spatial data (T × W²); the spatial
            mean is computed and inverted (1/mean) to convert to discharge units.
            Only one figure is produced in this case.
    """
    setup_ieee_style()
    
    unique_classes = sorted(barycenters.keys())
    n_classes = len(unique_classes)
    
    if is_raw:
        # For raw data: collapse spatial dimension → 1 virtual "parameter"
        n_params = 1
        effective_param_names = [r'$1/\lambda\,(m^3/s)$']
    else:
        if param_names is None:
            param_names = ['λ', 'β', 'γ', 'δ']
        n_params = X_train[0].shape[1] if len(X_train[0].shape) > 1 else 1
        effective_param_names = param_names
    
    # A4 dimensions — 25% shorter rows than original
    fig_width = 8.27
    row_height = 1.75  # was 2.33
    
    # Create time axis for 2019
    T = X_train[0].shape[0]
    time_axis = pd.date_range(start='2019-01-01', periods=T, freq='D')
    
    # Colors for samples
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    if output_dir:
        output_path = Path(output_dir) / "barycenter_grids"
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Number of rows needed (2 classes per row)
    n_rows = (n_classes + 1) // 2
    
    # Adaptive sample style based on count
    _lw_s = max(0.2, 0.7 * (5.0 / max(n_samples, 5)) ** 0.3)
    _al_s = max(0.10, 0.5 * (5.0 / max(n_samples, 5)) ** 0.5)

    # Pre-compute global y limits per parameter from all sample series
    # All values are converted to 1/λ = discharge space
    ylim_per_param = {}
    for param_idx in range(n_params):
        all_sample_vals = []
        for sample in X_train:
            if is_raw:
                y = np.maximum(np.mean(sample, axis=1), 1e-10)   # amplitude ≈ 1/λ
            else:
                y = 1.0 / np.maximum(sample[:, param_idx], 1e-10)  # λ → 1/λ
            all_sample_vals.append(y)
        sv = np.concatenate(all_sample_vals)
        pos = sv[sv > 0]
        ylim_per_param[param_idx] = (max(float(np.nanpercentile(pos, 1)) * 0.85, 1e-3),
                                     float(np.nanpercentile(pos, 99)) * 1.15)

    # Create one figure per parameter
    for param_idx in range(n_params):
        y_lo, y_hi = ylim_per_param[param_idx]
        fig, axes = plt.subplots(n_rows, 2, figsize=(fig_width, row_height * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for class_idx, class_label in enumerate(unique_classes):
            row = class_idx // 2
            col = class_idx % 2
            ax = axes[row, col]
            
            class_name = idx_to_regime[class_label]
            
            # Get samples for this class
            class_indices = [i for i in range(len(X_train)) if Y_train[i] == class_label]
            rng = np.random.RandomState(42 + class_label)
            rng.shuffle(class_indices)
            selected_indices = class_indices[:n_samples]
            
            # Plot training samples — all in 1/λ = discharge space
            for i, idx in enumerate(selected_indices):
                sample = X_train[idx]
                if is_raw:
                    y = np.maximum(np.mean(sample, axis=1), 1e-10)       # amplitude ≈ 1/λ
                else:
                    y = 1.0 / np.maximum(sample[:, param_idx], 1e-10)    # λ → 1/λ
                ax.semilogy(time_axis, y,
                            color='black', alpha=_al_s, linewidth=_lw_s)
            
            # Plot barycenter — all in 1/λ = discharge space
            barycenter = barycenters[class_label]
            if is_raw:
                b_vals = barycenter if barycenter.ndim == 1 else np.mean(barycenter, axis=1)
                b_plot = np.maximum(b_vals, 1e-10)                        # amplitude ≈ 1/λ
            else:
                b_plot = 1.0 / np.maximum(barycenter[:, param_idx], 1e-10)  # λ → 1/λ
            ax.semilogy(time_axis, b_plot, color='black', linewidth=2, label='Barycenter')
            
            ax.set_ylim(y_lo, y_hi)
            ax.set_title(f'{class_name}', fontsize=8, fontweight='bold')
            if col == 0:
                ax.set_ylabel(effective_param_names[param_idx], fontsize=9)
            if row == n_rows - 1:
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
                ax.tick_params(axis='x', rotation=0)
            else:
                ax.set_xticklabels([])
            ax.grid(True, alpha=0.3)
            if class_idx == 0:
                ax.legend(loc='upper right', fontsize=7)
        
        # Hide empty subplot if odd number of classes
        if n_classes % 2 == 1:
            axes[n_rows - 1, 1].set_visible(False)
        
        plt.tight_layout()
        
        if output_dir:
            method_suffix = method_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            if is_raw:
                pname = 'discharge'
            else:
                pname = effective_param_names[param_idx].replace('$', '').replace('/', '_').replace(' ', '')
            filename = f"grid_param_{param_idx}_{pname}_{method_suffix}"
            if save_pdf:
                plt.savefig(output_path / f"{filename}.pdf", bbox_inches='tight', dpi=300)
        
        plt.close()
