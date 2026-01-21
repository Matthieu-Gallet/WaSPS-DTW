"""
Utility module for meteorological data analysis.

This module provides utility functions for timing, histogram binning, and path setup.
"""

from .timing import print_timing
from .binning import get_optimal_bins
from .paths import setup_paths

__all__ = ['print_timing', 'get_optimal_bins', 'setup_paths']
