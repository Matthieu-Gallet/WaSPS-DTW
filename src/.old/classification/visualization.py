"""
Visualization functions for classification results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional
from pathlib import Path
from sklearn.metrics import confusion_matrix
import pandas as pd
import sys

# Add parent directory for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from estimator import LogCumulant

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# IEEE-compatible font settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10


def plot_confusion_matrices(results: Dict, Y_test: np.ndarray, idx_to_regime: Dict[int, str],
                            output_dir: str, fold_id: str = ""):
    """
    Plot and save confusion matrices for all methods.
    
    Args:
        results: Dictionary with classification results
        Y_test: True test labels
        idx_to_regime: Mapping from label index to regime code
        output_dir: Output directory
        fold_id: Identifier for the fold (for k-fold CV)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    target_names = [idx_to_regime[i] for i in sorted(idx_to_regime.keys())]
    
    methods = [
        ('euclidean_raw', 'Soft-DTW Euclidean (Raw Data)'),
        ('euclidean_params', 'Soft-DTW Euclidean (Parameters)'),
        ('wasserstein_params', 'Soft-DTW Wasserstein (Parameters)')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (method_key, method_name) in enumerate(methods):
        if method_key in results:
            Y_pred = results[method_key]['predictions']
            cm = confusion_matrix(Y_test, Y_pred)
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot
            ax = axes[idx]
            if HAS_SEABORN:
                import seaborn as sns
                sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues', 
                           xticklabels=target_names, yticklabels=target_names,
                           ax=ax, cbar_kws={'label': 'Proportion'})
            else:
                # Fallback to matplotlib imshow
                im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto')
                ax.set_xticks(np.arange(len(target_names)))
                ax.set_yticks(np.arange(len(target_names)))
                ax.set_xticklabels(target_names)
                ax.set_yticklabels(target_names)
                
                # Add text annotations
                for i in range(len(target_names)):
                    for j in range(len(target_names)):
                        text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
                
                plt.colorbar(im, ax=ax, label='Proportion')
            
            ax.set_title(method_name, fontweight='bold')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
    
    plt.tight_layout()
    
    filename = f"confusion_matrices{fold_id}.pdf"
    plt.savefig(output_path / filename, format='pdf', bbox_inches='tight')
    plt.close()


def plot_barycenter_with_samples(barycenters: Dict, train_samples: List[np.ndarray],
                                 train_labels: np.ndarray, idx_to_regime: Dict[int, str],
                                 method_name: str, output_dir: str, 
                                 n_samples_per_class: int = 10, alpha: float = 0.2,
                                 time_coords: Optional[np.ndarray] = None,
                                 is_raw_data: bool = False):
    """
    Plot barycenters with training samples for each class.
    
    Args:
        barycenters: Dictionary mapping class labels to barycenters
        train_samples: List of training samples
        train_labels: Training labels
        idx_to_regime: Mapping from label index to regime code
        method_name: Name of the method (for title and filename)
        output_dir: Output directory
        n_samples_per_class: Number of samples to plot per class
        alpha: Alpha value for sample lines
        time_coords: Optional time coordinates for x-axis (in decimal years)
        is_raw_data: If True, estimate lambda parameters for raw data before plotting
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    n_classes = len(barycenters)
    
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4), sharex=True)
    if n_classes == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    # Fix random seed for consistent sample selection across methods
    np.random.seed(42)
    
    # If raw data, estimate lambda parameters
    if is_raw_data:
        estimator = LogCumulant(distribution='exponential')
        
        # Convert barycenters to lambda parameters
        barycenters_lambda = {}
        for class_label, barycenter in barycenters.items():
            # barycenter shape: (T, D*W*W) for raw data
            # We want to estimate one lambda per timestep using all spatial values
            lambda_bary = []
            for t in range(barycenter.shape[0]):
                # Get all spatial values at this timestep
                spatial_vals = barycenter[t, :]  # Shape: (D*W*W,)
                valid_vals = spatial_vals[spatial_vals > 0]
                if len(valid_vals) > 0:
                    estimator.fit(valid_vals)
                    lambda_bary.append(estimator.get_params())
                else:
                    lambda_bary.append(np.nan)
            barycenters_lambda[class_label] = np.array(lambda_bary)
        
        # Convert samples to lambda parameters
        train_samples_lambda = []
        for sample in train_samples:
            # sample shape: (T, D*W*W)
            # We want to estimate one lambda per timestep using all spatial values
            lambda_sample = []
            for t in range(sample.shape[0]):
                # Get all spatial values at this timestep
                spatial_vals = sample[t, :]  # Shape: (D*W*W,)
                valid_vals = spatial_vals[spatial_vals > 0]
                if len(valid_vals) > 0:
                    estimator.fit(valid_vals)
                    lambda_sample.append(estimator.get_params())
                else:
                    lambda_sample.append(np.nan)
            train_samples_lambda.append(np.array(lambda_sample))
        
        barycenters = barycenters_lambda
        train_samples = train_samples_lambda
    
    for idx, (class_label, barycenter) in enumerate(sorted(barycenters.items())):
        ax = axes[idx]
        regime_name = idx_to_regime[class_label]
        
        # Get samples for this class (using fixed seed for consistency)
        class_indices = np.where(train_labels == class_label)[0]
        n_plot = min(n_samples_per_class, len(class_indices))
        plot_indices = np.random.choice(class_indices, size=n_plot, replace=False)
        
        # Determine x-axis values
        if time_coords is not None:
            import pandas as pd
            if isinstance(time_coords, pd.DatetimeIndex):
                # Convert to numeric for plotting, then format labels
                x_vals = np.arange(len(time_coords))
            else:
                x_vals = time_coords
        else:
            x_vals = np.arange(len(barycenter) if barycenter.ndim == 1 else barycenter.shape[0])
        
        # Plot samples with low alpha
        for i in plot_indices:
            sample = train_samples[i]
            # If sample is 2D (parameter), flatten to 1D
            if sample.ndim == 2 and sample.shape[1] == 1:
                sample = sample.flatten()
            ax.plot(x_vals[:len(sample)], sample, color=colors[idx], alpha=alpha, linewidth=1)
        
        # Plot barycenter
        barycenter_plot = barycenter
        if barycenter.ndim == 2 and barycenter.shape[1] == 1:
            barycenter_plot = barycenter.flatten()
        ax.plot(x_vals[:len(barycenter_plot)], barycenter_plot, color=colors[idx], linewidth=3, 
               label=f'Barycenter {regime_name}')
        
        ax.set_title(f'Class: {regime_name}', fontweight='bold')
        
        # Set x-axis labels for datetime
        if time_coords is not None and isinstance(time_coords, pd.DatetimeIndex):
            # Select 4 evenly spaced positions for month labels
            n_ticks = 4
            tick_positions = np.linspace(0, len(x_vals)-1, n_ticks, dtype=int)
            tick_labels = [time_coords[i].strftime('%b') for i in tick_positions]  # 3-letter month names
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.set_xlabel('Month (2019)')
        else:
            if time_coords is not None:
                ax.set_xlabel('Year')
            else:
                ax.set_xlabel('Time Step')
        
        ax.set_ylabel('Lambda Parameter' if is_raw_data or 'Parameter' in method_name else 'Value')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Create safe filename
    safe_method_name = method_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
    filename = f"barycenter_samples_{safe_method_name}.pdf"
    plt.savefig(output_path / filename, format='pdf', bbox_inches='tight')
    plt.close()


def plot_gamma_sensitivity(results: Dict, output_dir: str):
    """
    Plot F1 scores as a function of gamma for all methods using boxplots.
    
    Args:
        results: Dictionary with results for each gamma value and fold
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    gamma_values = sorted(results.keys())
    
    methods = {
        'euclidean_raw': 'Euclidean (Raw)',
        'euclidean_params': 'Euclidean (Params)',
        'wasserstein_params': 'Wasserstein (Params)'
    }
    
    # Prepare data for boxplot: collect F1 scores across folds for each gamma and method
    # Results structure: results[gamma][method_key] contains fold results
    boxplot_data = {method_key: [] for method_key in methods.keys()}
    positions = {method_key: [] for method_key in methods.keys()}
    
    for i, gamma in enumerate(gamma_values):
        for method_key in methods.keys():
            # Get all fold scores for this gamma and method
            fold_scores = results[gamma][method_key].get('fold_scores', [])
            if not fold_scores:  # If no fold data, use mean/std
                mean = results[gamma][method_key]['f1_weighted_mean']
                boxplot_data[method_key].append([mean])
            else:
                boxplot_data[method_key].append(fold_scores)
            positions[method_key].append(i)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.8))
    
    # Tab colors
    colors_list = ['tab:blue', 'tab:green', 'tab:red']
    color_map = {k: colors_list[i] for i, k in enumerate(methods.keys())}
    
    # Plot boxplots for each method
    box_width = 0.25
    for method_idx, (method_key, method_name) in enumerate(methods.items()):
        # Offset positions for side-by-side boxplots
        offset = (method_idx - 1) * box_width
        pos = [p + offset for p in positions[method_key]]
        
        bp = ax.boxplot(boxplot_data[method_key], positions=pos, widths=box_width*0.8,
                       patch_artist=True, showfliers=True,
                       boxprops=dict(facecolor=color_map[method_key], alpha=0.6),
                       medianprops=dict(color='black', linewidth=1.5),
                       whiskerprops=dict(color=color_map[method_key]),
                       capprops=dict(color=color_map[method_key]),
                       flierprops=dict(marker='o', markerfacecolor=color_map[method_key], 
                                     markersize=4, alpha=0.5))
    
    # Configure axes
    ax.set_xlabel('Gamma', fontsize=11, fontweight='bold')
    ax.set_ylabel('F1 Score (Weighted)', fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(gamma_values)))
    ax.set_xticklabels([f'{g:.1f}' for g in gamma_values])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Create legend manually
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[k], alpha=0.6, label=methods[k]) 
                      for k in methods.keys()]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12),
             ncol=3, frameon=False, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / "gamma_sensitivity.pdf", format='pdf', bbox_inches='tight')
    plt.close()


def plot_sample_size_sensitivity(results: Dict, output_dir: str):
    """
    Plot F1 scores as a function of training sample size for all methods using boxplots.
    
    Args:
        results: Dictionary with results for each sample size and fold
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    sample_sizes = sorted(results.keys())
    
    methods = {
        'euclidean_raw': 'Euclidean (Raw)',
        'euclidean_params': 'Euclidean (Params)',
        'wasserstein_params': 'Wasserstein (Params)'
    }
    
    # Prepare data for boxplot
    boxplot_data = {method_key: [] for method_key in methods.keys()}
    positions = {method_key: [] for method_key in methods.keys()}
    
    for i, size in enumerate(sample_sizes):
        for method_key in methods.keys():
            # Get all fold scores for this size and method
            fold_scores = results[size][method_key].get('fold_scores', [])
            if not fold_scores:  # If no fold data, use mean
                mean = results[size][method_key]['f1_weighted_mean']
                boxplot_data[method_key].append([mean])
            else:
                boxplot_data[method_key].append(fold_scores)
            positions[method_key].append(i)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.8))
    
    # Tab colors
    colors_list = ['tab:blue', 'tab:green', 'tab:red']
    color_map = {k: colors_list[i] for i, k in enumerate(methods.keys())}
    
    # Plot boxplots for each method
    box_width = 0.25
    for method_idx, (method_key, method_name) in enumerate(methods.items()):
        # Offset positions for side-by-side boxplots
        offset = (method_idx - 1) * box_width
        pos = [p + offset for p in positions[method_key]]
        
        bp = ax.boxplot(boxplot_data[method_key], positions=pos, widths=box_width*0.8,
                       patch_artist=True, showfliers=True,
                       boxprops=dict(facecolor=color_map[method_key], alpha=0.6),
                       medianprops=dict(color='black', linewidth=1.5),
                       whiskerprops=dict(color=color_map[method_key]),
                       capprops=dict(color=color_map[method_key]),
                       flierprops=dict(marker='o', markerfacecolor=color_map[method_key], 
                                     markersize=4, alpha=0.5))
    
    # Configure axes
    ax.set_xlabel('Training Sample Size (fraction)', fontsize=11, fontweight='bold')
    ax.set_ylabel('F1 Score (Weighted)', fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(sample_sizes)))
    ax.set_xticklabels([f'{s:.1f}' for s in sample_sizes])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Create legend manually
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[k], alpha=0.6, label=methods[k]) 
                      for k in methods.keys()]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12),
             ncol=3, frameon=False, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / "sample_size_sensitivity.pdf", format='pdf', bbox_inches='tight')
    plt.close()
