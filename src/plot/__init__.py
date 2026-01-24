"""
Plotting module for visualization of barycenter results and time series.

This module provides visualization functions for:
- Barycenter comparison plots
- Time series with DTW alignments
- Training loss curves
- Classification results visualization
"""

from .barycenter_plots import (
    plot_with_correspondences,
    plot_multiple_series,
    create_barycenter_comparison_figure
)
from .series_plots import plot_zone, create_geographic_barycenter_figures
from .prediction_plots import plot_predictions, plot_losses
from .classification_plots import (
    setup_ieee_style,
    plot_confusion_matrices,
    plot_barycenter_with_samples,
    plot_gamma_sensitivity,
    plot_sample_size_sensitivity,
    plot_kfold_boxplots,
    plot_summary_figure,
    plot_class_pair_barycenters,
    plot_all_class_barycenters_grid
)

__all__ = [
    'plot_with_correspondences',
    'plot_multiple_series',
    'create_barycenter_comparison_figure',
    'plot_zone',
    'create_geographic_barycenter_figures',
    'plot_predictions',
    'plot_losses',
    # Classification plots
    'setup_ieee_style',
    'plot_confusion_matrices',
    'plot_barycenter_with_samples',
    'plot_gamma_sensitivity',
    'plot_sample_size_sensitivity',
    'plot_kfold_boxplots',
    'plot_summary_figure',
    'plot_class_pair_barycenters',
    'plot_all_class_barycenters_grid'
]
