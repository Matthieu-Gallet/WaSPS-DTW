"""
Data loading module for time series analysis.

This module provides functions for loading and preprocessing NetCDF data
for temporal prediction tasks.
"""

from .netcdf_loader import load_data
from .series_extraction import extract_lambda_series, extract_multiple_windows_around_position
from .preprocessing import split_train_test, create_sliding_windows

__all__ = [
    'load_data',
    'extract_lambda_series',
    'extract_multiple_windows_around_position',
    'split_train_test',
    'create_sliding_windows'
]
