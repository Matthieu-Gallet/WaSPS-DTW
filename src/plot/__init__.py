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
    # Previously defined but not exported
    plot_cpazmal_sensitivity,
    plot_barycenters_gamma_comparison,
    # Debug helpers (new in JAX branch)
    plot_samples_with_fitted_pdf,
    plot_barycenter_debug,
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
    'plot_cpazmal_sensitivity',
    'plot_barycenters_gamma_comparison',
    'plot_samples_with_fitted_pdf',
    'plot_barycenter_debug',
]
