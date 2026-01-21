"""
Plotting module for visualization of barycenter results and time series.

This module provides visualization functions for:
- Barycenter comparison plots
- Time series with DTW alignments
- Training loss curves
"""

from .barycenter_plots import (
    plot_with_correspondences,
    plot_multiple_series,
    create_barycenter_comparison_figure
)
from .series_plots import plot_zone, create_geographic_barycenter_figures
from .prediction_plots import plot_predictions, plot_losses

__all__ = [
    'plot_with_correspondences',
    'plot_multiple_series',
    'create_barycenter_comparison_figure',
    'plot_zone',
    'create_geographic_barycenter_figures',
    'plot_predictions',
    'plot_losses'
]
