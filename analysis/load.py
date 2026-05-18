"""
Result and data loading functions for analysis.

  load_results(study_dir)                  glob **/results.zarr → list of dicts
  filter_results(results, **conditions)    exact-match filter on parameters
  to_dataframe(results)                    scalar attrs → pandas DataFrame
  aggregate_folds(results, group_by)       fold lists → {key: {metric: [values]}}
  load_classification(data_dir)            Format A → (X, Y, metadata)
  load_groups(groups_dir)                  Format B → ({group: series}, metadata)
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import zarr


def load_results(study_dir: str) -> List[dict]:
    """
    Recursively load all results.zarr files under study_dir.

    For flat zarr stores (single experiment) every array and attribute is
    returned at the top level.  For stores with sub-groups (e.g. one group
    per barycenter method), each sub-group is returned as a separate dict
    with a '_group' key carrying the group path.
    """
    results = []
    for zarr_path in Path(study_dir).rglob('results.zarr'):
        store = zarr.open(str(zarr_path), mode='r')
        top_attrs = dict(store.attrs)

        # Check for sub-groups (e.g. method groups in classify.py output)
        sub_groups = [k for k in store if isinstance(store[k], zarr.hierarchy.Group)]

        if sub_groups:
            for sg_name in sub_groups:
                sg = store[sg_name]
                result = {**top_attrs, **dict(sg.attrs), '_group': sg_name}
                for name in sg:
                    if not isinstance(sg[name], zarr.hierarchy.Group):
                        result[name] = sg[name][:]
                result['_path'] = str(zarr_path)
                results.append(result)
        else:
            result = dict(top_attrs)
            for name in store:
                if not isinstance(store[name], zarr.hierarchy.Group):
                    result[name] = store[name][:]
            result['_path'] = str(zarr_path)
            results.append(result)

    return results


def filter_results(results: List[dict], **conditions) -> List[dict]:
    """
    Exact-match filter: keep only results where all conditions match.

    Example:
        filter_results(results, gamma=1.0, method='euclidean_raw')
    """
    return [r for r in results
            if all(r.get(k) == v for k, v in conditions.items())]


def aggregate_folds(results: List[dict],
                    group_by: List[str],
                    metrics: List[str] = ('f1_weighted', 'f1_macro',
                                          'barycenter_time', 'classify_time'),
                    ) -> Dict[Any, Dict[str, List[float]]]:
    """
    Group results by a key tuple and collect per-fold metric lists.

    Args:
        results:  Output of load_results (or filter_results).
        group_by: List of attribute names that define one experimental condition,
                  e.g. ['gamma', 'method'] for a gamma-sweep study.
        metrics:  Scalar attributes to aggregate.

    Returns:
        {key_tuple: {metric: [fold_value, ...]}}

    Example:
        agg = aggregate_folds(results, group_by=['gamma', 'method'])
        # agg[(1.0, 'euclidean_params')]['f1_weighted'] → [0.81, 0.79, 0.83, ...]
    """
    grouped: Dict[Any, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = tuple(r.get(k) for k in group_by)
        for m in metrics:
            if m in r and not isinstance(r[m], np.ndarray):
                grouped[key][m].append(float(r[m]))
    return {k: dict(v) for k, v in grouped.items()}


def to_dataframe(results: List[dict]) -> pd.DataFrame:
    """
    Convert a results list to a pandas DataFrame.

    Array-valued keys and private keys (_path, _group) are dropped;
    every scalar attr becomes a column.
    """
    rows = []
    for r in results:
        row = {k: v for k, v in r.items()
               if not isinstance(v, np.ndarray) and not k.startswith('_')}
        rows.append(row)
    return pd.DataFrame(rows)


def load_classification(data_dir: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load a Format A classification dataset.

    Returns:
        X:        float64 array (N, T, D)
        Y:        int array (N,)
        metadata: dict — contains 'idx_to_label' {int: class_name}
    """
    p = Path(data_dir)
    X = np.load(p / 'X.npy')
    Y = np.load(p / 'Y.npy')
    metadata = np.load(p / 'metadata.npy', allow_pickle=True).item()
    # Normalise idx_to_label keys to int (may be stored as str)
    if 'idx_to_label' in metadata:
        metadata['idx_to_label'] = {int(k): v
                                     for k, v in metadata['idx_to_label'].items()}
    return X, Y, metadata


def load_groups(groups_dir: str) -> Tuple[Dict[str, np.ndarray], dict]:
    """
    Load a Format B groups dataset.

    Returns:
        groups:   {group_key: series array (M, T, D)}
        metadata: dict
    """
    p = Path(groups_dir)
    metadata = np.load(p / 'metadata.npy', allow_pickle=True).item()
    groups = {}
    for group_dir in sorted(p.iterdir()):
        if group_dir.is_dir() and (group_dir / 'series.npy').exists():
            groups[group_dir.name] = np.load(group_dir / 'series.npy')
    return groups, metadata
