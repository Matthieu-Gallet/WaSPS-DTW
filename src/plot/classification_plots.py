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
from typing import Dict, List, Optional
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
# Confusion matrix plots
# =============================================================================

def plot_confusion_matrices(results: Dict, Y_test: np.ndarray, 
                            idx_to_regime: Dict[int, str],
                            output_dir: str = None, save_pdf: bool = True):
    """
    Plot confusion matrices for all three methods.
    
    Args:
        results: Dictionary with classification results
        Y_test: True test labels
        idx_to_regime: Mapping from label index to regime code
        output_dir: Output directory for saving
        save_pdf: Whether to save as PDF
    """
    setup_ieee_style()
    
    target_names = [idx_to_regime[i] for i in sorted(idx_to_regime.keys())]
    methods = [
        ('euclidean_raw', 'Soft-DTW Euclidean\n(Raw Data)'),
        ('euclidean_params', 'Soft-DTW Euclidean\n(Parameters)'),
        ('wasserstein_params', 'Soft-DTW Wasserstein\n(Parameters)')
    ]
    
    # IEEE double-column width ≈ 7 inches
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
    
    for idx, (method_key, method_name) in enumerate(methods):
        if method_key not in results:
            continue
            
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

def plot_class_pair_barycenters(barycenters: Dict[int, np.ndarray],
                                 X_train: List[np.ndarray],
                                 Y_train: np.ndarray,
                                 idx_to_regime: Dict[int, str],
                                 method_name: str,
                                 output_dir: str = None,
                                 save_pdf: bool = True,
                                 n_samples: int = 10,
                                 param_names: List[str] = None,
                                 is_raw_data: bool = False):
    """
    Plot barycenters with training samples for each pair of classes.
    
    Each figure shows 2 classes side by side (2x1 horizontal layout) with
    10 training samples and the barycenter. Figures are sized for A4 width
    and 1/5 A4 height.
    
    Args:
        barycenters: Dictionary mapping class labels to barycenters
        X_train: Training samples (list of arrays)
        Y_train: Training labels
        idx_to_regime: Mapping from label index to regime code
        method_name: Name of the method for the title
        output_dir: Output directory for saving
        save_pdf: Whether to save as PDF
        n_samples: Number of samples to plot per class (default: 10)
        param_names: Names of parameters (default: ['λ', 'β', 'γ', 'δ'])
        is_raw_data: Whether data is raw (True) or parameters (False)
    """
    setup_ieee_style()
    
    if param_names is None:
        param_names = ['λ', 'β', 'γ', 'δ']
    
    unique_classes = sorted(barycenters.keys())
    n_classes = len(unique_classes)
    
    # Determine number of parameters/features
    sample_shape = X_train[0].shape
    if is_raw_data:
        # Raw data: shape is (T, D, W, W) - flatten spatial dims
        n_features = 1  # We'll plot mean across spatial dimensions
    else:
        # Parameters: shape is (T, n_params)
        n_features = sample_shape[1] if len(sample_shape) > 1 else 1
    
    # A4 dimensions: 210mm x 297mm
    # Width: 210mm ≈ 8.27 inches
    # Height: 1/5 of 297mm ≈ 59mm ≈ 2.33 inches
    fig_width = 8.27  # A4 width in inches
    fig_height = 2.33  # 1/5 A4 height in inches
    
    # Create time axis for 2019
    T = X_train[0].shape[0]
    time_axis = pd.date_range(start='2019-01-01', periods=T, freq='D')
    
    # Generate all pairs of classes
    from itertools import combinations
    class_pairs = list(combinations(unique_classes, 2))
    
    # Also add each class paired with itself for completeness
    # Actually, user asked for pairs, so we'll do combinations
    
    # Colors for samples and barycenter
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    if output_dir:
        output_path = Path(output_dir) / "class_pairs"
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Fix seed for consistent sample selection across methods
    np.random.seed(42)
    
    for pair_idx, (class1, class2) in enumerate(class_pairs):
        class1_name = idx_to_regime[class1]
        class2_name = idx_to_regime[class2]
        
        # Create figure with 2 subplots side by side
        if is_raw_data:
            fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        else:
            # For parameters, we need n_features rows
            fig, axes = plt.subplots(n_features, 2, figsize=(fig_width, fig_height * n_features))
            if n_features == 1:
                axes = axes.reshape(1, 2)
        
        for col_idx, (class_label, class_name) in enumerate([(class1, class1_name), (class2, class2_name)]):
            # Get samples for this class
            class_indices = [i for i in range(len(X_train)) if Y_train[i] == class_label]
            # Use seed based on class label for reproducibility
            rng = np.random.RandomState(42 + class_label)
            rng.shuffle(class_indices)
            selected_indices = class_indices[:n_samples]
            
            barycenter = barycenters[class_label]
            
            if is_raw_data:
                # Raw data: plot mean across spatial dimensions
                ax = axes[col_idx]
                
                # Plot training samples
                for i, idx in enumerate(selected_indices):
                    sample = X_train[idx]
                    # Mean across spatial dimensions
                    # If shape is (T, D, W, W), mean over (1,2,3)
                    # If shape is (T, D), mean over (1,)
                    if sample.ndim == 4:
                        sample_mean = np.mean(sample, axis=(1, 2, 3))
                    elif sample.ndim == 2:
                        sample_mean = np.mean(sample, axis=1)
                    else:
                        sample_mean = sample.flatten()
                    ax.semilogy(time_axis, sample_mean, color=colors[i % len(colors)], 
                           alpha=0.4, linewidth=0.8)
                
                # Plot barycenter
                if barycenter.ndim > 1:
                    barycenter_mean = np.mean(barycenter, axis=tuple(range(1, barycenter.ndim)))
                else:
                    barycenter_mean = barycenter
                ax.semilogy(time_axis, barycenter_mean, color='black', linewidth=2, 
                       label='Barycenter')
                
                ax.set_title(f'{class_name}', fontsize=8, fontweight='bold')
                ax.set_xlabel('Date')
                if col_idx == 0:
                    ax.set_ylabel('Discharge (mean)')
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
                ax.tick_params(axis='x', rotation=0)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize=7)
                
            else:
                # Parameters: plot each parameter in a separate row
                for param_idx in range(n_features):
                    ax = axes[param_idx, col_idx]
                    
                    # Plot training samples
                    for i, idx in enumerate(selected_indices):
                        sample = X_train[idx]
                        ax.semilogy(time_axis, sample[:, param_idx], 
                               color=colors[i % len(colors)], alpha=0.4, linewidth=0.8)
                    
                    # Plot barycenter
                    ax.semilogy(time_axis, barycenter[:, param_idx], color='black', 
                           linewidth=2, label='Barycenter')
                    
                    # Labels
                    if param_idx == 0:
                        ax.set_title(f'{class_name}', fontsize=8, fontweight='bold')
                    if col_idx == 0:
                        ax.set_ylabel(param_names[param_idx], fontsize=9)
                    if param_idx == n_features - 1:
                        ax.set_xlabel('Date')
                        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
                        ax.tick_params(axis='x', rotation=0)
                    else:
                        ax.set_xticklabels([])
                    
                    ax.grid(True, alpha=0.3)
                    if param_idx == 0 and col_idx == 1:
                        ax.legend(loc='upper right', fontsize=7)
        
        plt.tight_layout()
        
        if output_dir:
            # Clean method name for filename
            method_suffix = method_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            filename = f"pair_{class1_name}_{class2_name}_{method_suffix}"
            if save_pdf:
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
                                     param_names: List[str] = None):
    """
    Plot all class barycenters in a grid layout with training samples.
    
    Creates one figure per parameter with all classes in 2-column layout.
    Each figure is A4 width and 1/5 A4 height per row of classes.
    
    Args:
        barycenters: Dictionary mapping class labels to barycenters
        X_train: Training samples (list of parameter arrays)
        Y_train: Training labels
        idx_to_regime: Mapping from label index to regime code
        method_name: Name of the method for the title
        output_dir: Output directory for saving
        save_pdf: Whether to save as PDF
        n_samples: Number of samples to plot per class (default: 10)
        param_names: Names of parameters (default: ['λ', 'β', 'γ', 'δ'])
    """
    setup_ieee_style()
    
    if param_names is None:
        param_names = ['λ', 'β', 'γ', 'δ']
    
    unique_classes = sorted(barycenters.keys())
    n_classes = len(unique_classes)
    n_params = X_train[0].shape[1] if len(X_train[0].shape) > 1 else 1
    
    # A4 dimensions
    fig_width = 8.27  # A4 width in inches
    row_height = 2.33  # 1/5 A4 height per row
    
    # Create time axis for 2019
    T = X_train[0].shape[0]
    time_axis = pd.date_range(start='2019-01-01', periods=T, freq='D')
    
    # Colors for samples
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    if output_dir:
        output_path = Path(output_dir) / "barycenter_grids"
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Fix seed for consistent sample selection across methods
    np.random.seed(42)
    
    # Number of rows needed (2 classes per row)
    n_rows = (n_classes + 1) // 2
    
    # Create one figure per parameter
    for param_idx in range(n_params):
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
            # Use seed based on class label for reproducibility
            rng = np.random.RandomState(42 + class_label)
            rng.shuffle(class_indices)
            selected_indices = class_indices[:n_samples]
            
            # Plot training samples
            for i, idx in enumerate(selected_indices):
                sample = X_train[idx]
                ax.semilogy(time_axis, sample[:, param_idx], 
                       color=colors[i % len(colors)], alpha=0.4, linewidth=0.8)
            
            # Plot barycenter
            barycenter = barycenters[class_label]
            ax.semilogy(time_axis, barycenter[:, param_idx], color='black', 
                   linewidth=2, label='Barycenter')
            
            ax.set_title(f'{class_name}', fontsize=8, fontweight='bold')
            if col == 0:
                ax.set_ylabel(param_names[param_idx], fontsize=9)
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
            # Clean method name for filename
            method_suffix = method_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            filename = f"grid_param_{param_idx}_{param_names[param_idx]}_{method_suffix}"
            if save_pdf:
                plt.savefig(output_path / f"{filename}.pdf", bbox_inches='tight', dpi=300)
        
        plt.close()
