"""Plotting module — confusion matrices, per-class barycenter overlays."""

from .classification_plots import (
    setup_ieee_style,
    plot_confusion_matrices,
    plot_class_pair_barycenters,
)

__all__ = [
    'setup_ieee_style',
    'plot_confusion_matrices',
    'plot_class_pair_barycenters',
]
