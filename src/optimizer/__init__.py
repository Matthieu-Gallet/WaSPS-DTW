"""
Optimizer module for barycenter computation and prediction.

This module provides optimization algorithms for:
- SGD barycenter computation with Soft-DTW Wasserstein distance
- MLP predictor training with various loss functions
"""

from .wasserstein_barycenter_sgd import sgd_barycenter
from .mlp_predictor import MLP, sgd_predictor, predict

__all__ = ['sgd_barycenter', 'MLP', 'sgd_predictor', 'predict']
