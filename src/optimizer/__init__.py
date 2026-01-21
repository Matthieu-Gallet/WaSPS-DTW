"""
Optimizer module for barycenter computation and prediction.

This module provides optimization algorithms for:
- SGD barycenter computation with Soft-DTW
- MLP predictor training with various loss functions
"""

from .sgd_barycenter import sgd_barycenter
from .mlp_predictor import MLP, sgd_predictor, predict

__all__ = ['sgd_barycenter', 'MLP', 'sgd_predictor', 'predict']
