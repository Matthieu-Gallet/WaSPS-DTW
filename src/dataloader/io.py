"""
Thin loaders for the two canonical dataset formats.

Format A (classification):
  X.npy        float64  (N, T, D)
  Y.npy        int      (N,)
  metadata.npy pickled dict

Format B (groups):
  group_<i>/series.npy  float64  (M, T, D)
  metadata.npy          pickled dict
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_classification(data_dir: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load a Format A classification dataset.

    Returns:
        X:        float64 array (N, T, D)
        Y:        int array (N,)
        metadata: dict — must contain 'idx_to_label' mapping int → class name
    """
    p = Path(data_dir)
    X = np.load(p / 'X.npy')
    Y = np.load(p / 'Y.npy')
    metadata = np.load(p / 'metadata.npy', allow_pickle=True).item()
    return X, Y, metadata


def load_groups(groups_dir: str) -> Tuple[Dict[str, np.ndarray], dict]:
    """
    Load a Format B groups dataset.

    Returns:
        groups:   {group_key: series array (M, T, D)}
        metadata: dict — may contain 'group_names' mapping key → human label
    """
    p = Path(groups_dir)
    metadata = np.load(p / 'metadata.npy', allow_pickle=True).item()
    groups = {}
    for group_dir in sorted(p.iterdir()):
        if group_dir.is_dir() and (group_dir / 'series.npy').exists():
            groups[group_dir.name] = np.load(group_dir / 'series.npy')
    return groups, metadata
