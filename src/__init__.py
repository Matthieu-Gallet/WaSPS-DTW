"""
Source package for Soft-DTW analysis tools.

This package provides tools for analyzing time series data with Soft-DTW
and Wasserstein distance metrics, including:

Modules:
- dataloader: Data loading and preprocessing for NetCDF files
- data_generator: Synthetic data generation for experiments
- estimator: Parameter estimation (MLE, log-cumulant)
- optimizer: SGD-based optimization for barycenters and MLP predictors
- plot: Visualization utilities
- utils: General utility functions
- experiments: Experiment scripts (simulation, geographic_barycenter)

Example usage:
    from src.dataloader import load_data
    from src.optimizer import sgd_barycenter
    from src.plot import plot_with_correspondences
"""

__version__ = "2.0.0"
__author__ = "Soft-DTW Analysis Tools"

# Main imports for convenience
from .utils import print_timing, get_optimal_bins, setup_paths
from .dataloader import load_data, extract_lambda_series, split_train_test, create_sliding_windows
from .data_generator import generate_exponential_series, generate_shifted_series
from .estimator import (
    MLE, LogCumulant
)
from .optimizer import sgd_barycenter, MLP, sgd_predictor, predict
from .plot import (
    plot_with_correspondences, plot_multiple_series, create_barycenter_comparison_figure,
    plot_zone, create_geographic_barycenter_figures,
    plot_predictions, plot_losses
)

__all__ = [
    # Utils
    'print_timing', 'get_optimal_bins', 'setup_paths',
    # Dataloader
    'load_data', 'extract_lambda_series', 'split_train_test', 'create_sliding_windows',
    # Data generator
    'generate_exponential_series', 'generate_shifted_series',
    # Estimator
    'ExponentialMLE', 'estimate_exponential_mle',
    'ExponentialLogCumulant', 'estimate_exponential_log_cumulant',
    # Optimizer
    'sgd_barycenter', 'MLP', 'sgd_predictor', 'predict',
    # Plot
    'plot_with_correspondences', 'plot_multiple_series', 'create_barycenter_comparison_figure',
    'plot_zone', 'create_geographic_barycenter_figures',
    'plot_predictions', 'plot_losses',
]