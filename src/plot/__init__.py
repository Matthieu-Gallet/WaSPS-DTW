"""Plotting module — classification results, barycenters, sensitivity analysis."""

from .classification_plots import (
    setup_ieee_style,
    plot_confusion_matrices,
    plot_barycenter_with_samples,
    plot_gamma_sensitivity,
    plot_sample_size_sensitivity,
    plot_kfold_boxplots,
    plot_summary_figure,
    plot_class_pair_barycenters,
    plot_all_class_barycenters_grid,
)

__all__ = [
    'setup_ieee_style',
    'plot_confusion_matrices',
    'plot_barycenter_with_samples',
    'plot_gamma_sensitivity',
    'plot_sample_size_sensitivity',
    'plot_kfold_boxplots',
    'plot_summary_figure',
    'plot_class_pair_barycenters',
    'plot_all_class_barycenters_grid',
]
