"""
Data loading module for time series analysis.

This module provides functions for loading and preprocessing NetCDF data
for temporal prediction tasks and classification experiments.
"""

from .netcdf_loader import load_data
from .series_extraction import extract_lambda_series, extract_multiple_windows_around_position
from .preprocessing import split_train_test, create_sliding_windows
from .classification_loader import (
    load_classification_dataset,
    preprocess_samples,
    estimate_parameters_for_samples,
    load_class_thresholds,
)
from .cpazmal_loader import (
    MLDatasetLoader,
    download_cpazmal,
    windows_to_time_series,
    extract_time_series,
)

__all__ = [
    'load_data',
    'extract_lambda_series',
    'extract_multiple_windows_around_position',
    'split_train_test',
    'create_sliding_windows',
    # Classification (UCR-style)
    'load_classification_dataset',
    'preprocess_samples',
    'estimate_parameters_for_samples',
    'load_class_thresholds',
    # CPAZMaL SAR dataset
    'MLDatasetLoader',
    'download_cpazmal',
    'windows_to_time_series',
    'extract_time_series',
]
