"""
Data generation module for synthetic time series.

This module provides functions for generating synthetic exponential
distribution samples for testing barycenter methods.
"""

from .exponential_generator import generate_exponential_series
from .shifted_series import generate_shifted_series

__all__ = ['generate_exponential_series', 'generate_shifted_series']
