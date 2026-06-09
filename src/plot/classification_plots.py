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
    
    target_names = [idx_to_regime[i] for i in sorted(idx_to_regime.keys())]
    method_candidates = [
        ('euclidean_raw', 'Soft-DTW Euclidean\n(Raw Data)'),
        ('euclidean_params', 'Soft-DTW Euclidean\n(Parameters)'),
        ('wasserstein_params', 'Soft-DTW Wasserstein\n(Parameters)'),
        ('lstm_raw', 'LSTM Barycenter\n(Raw Data)'),
        ('ot_regul_raw', 'Regularized OT STA\n(Raw Data)')
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
        cm = confusion_matrix(Y_test, Y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
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
                                  param_names: List[str] = None):
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
    
    # Create time axis for 2019
    T = X_train_params[0].shape[0]
    time_axis = pd.date_range(start='2019-01-01', periods=T, freq='D')
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))
    
    # Fix seed for consistent sample selection
    np.random.seed(42)
    
    for class_idx, class_label in enumerate(unique_classes):
        class_name = idx_to_regime[class_label]
        
        # Get samples for this class
        class_indices = [i for i in range(len(X_train_params)) if Y_train[i] == class_label]
        class_params = [X_train_params[i] for i in class_indices]
        
        # Get barycenter (use Wasserstein barycenter if available)
        if 'wasserstein_params' in results and 'barycenters' in results['wasserstein_params']:
            barycenter = results['wasserstein_params']['barycenters'][class_label]
        elif 'euclidean_params' in results and 'barycenters' in results['euclidean_params']:
            barycenter = results['euclidean_params']['barycenters'][class_label]
        else:
            continue
        
        for param_idx in range(n_params):
            ax = axes[class_idx, param_idx]
            
            # Plot individual samples (light)
            for sample_params in class_params[:20]:  # Limit to 20 samples for clarity
                ax.semilogy(time_axis, sample_params[:, param_idx], 
                       color=colors[class_idx], alpha=0.2, linewidth=0.5)
            
            # Plot barycenter (bold)
            ax.semilogy(time_axis, barycenter[:, param_idx], 
                   color='black', linewidth=1.5, label='Barycenter')
            
            if class_idx == 0:
                ax.set_title(param_names[param_idx], fontsize=10)
            
            if param_idx == 0:
                ax.set_ylabel(class_name, fontsize=10)
            
            if class_idx == n_classes - 1:
                # Format x-axis with months
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
