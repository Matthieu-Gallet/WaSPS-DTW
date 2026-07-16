"""Tests for the CPAZMaL classification loader — class exclusion, per-class group
caps, and K-fold splitting.

No real HDF5 extraction: `_FakeLoader` stands in for `MLDatasetLoader` to test
`extract_time_series`'s orchestration logic (exclusion, capping, one-series-per-
window collapsing), and `_load_cpazmal`'s K-fold splitting is tested against a
synthetic on-disk cache (same file layout it reads, no HDF5 involved).
"""

import sys
import json
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "experiment" / "utils"))

from data.cpazmal_loader import extract_time_series


# ---------------------------------------------------------------------------
# Fake MLDatasetLoader — no HDF5 file needed
# ---------------------------------------------------------------------------

class _FakeLoader:
    """Minimal stand-in for MLDatasetLoader: `n_groups_per_class` groups per
    class, each yielding `windows_per_group` spatial windows of shape
    (window_size, window_size, T)."""

    def __init__(self, classes, n_groups_per_class=4, windows_per_group=2,
                 window_size=4, T=6):
        self.classes = classes
        self._windows_per_group = windows_per_group
        self._window_size = window_size
        self._T = T
        self._group_to_class = {
            f"{cls}{i:03d}": cls
            for cls in classes
            for i in range(n_groups_per_class)
        }

    def get_groups_by_class(self, class_name):
        return [g for g, c in self._group_to_class.items() if c == class_name]

    def get_all_groups_with_classes(self):
        return dict(self._group_to_class)

    def load_data(self, group_name, orbit, polarisation, start_date, end_date,
                   normalize, remove_nodata, scale_type):
        W, T = self._window_size, self._T
        rng = np.random.default_rng(abs(hash(group_name)) % (2**32))
        images = rng.uniform(0.1, 1.0, size=(2 * W, 2 * W, T))
        masks = np.zeros((2 * W, 2 * W, T), dtype=np.uint8)
        return {'images': images, 'masks': masks,
                'timestamps': [f"t{t}" for t in range(T)]}

    def extract_windows(self, image, mask, window_size, stride, max_mask_value,
                         max_mask_percentage, min_valid_percentage, skip_optim_offset):
        n = self._windows_per_group
        windows = np.stack([image[:window_size, :window_size, :] for _ in range(n)])
        wm = np.stack([mask[:window_size, :window_size, :] for _ in range(n)])
        return windows, wm, [(0, 0)] * n


# ---------------------------------------------------------------------------
# extract_time_series — exclusion, per-class cap, single-series contract
# ---------------------------------------------------------------------------

def test_exclude_classes_drops_study_and_hag_by_default():
    loader = _FakeLoader(['ACC', 'STUDY', 'HAG', 'FOR'])
    result = extract_time_series(loader, window_size=4, verbose=False)
    assert set(result['class_names'].values()) == {'ACC', 'FOR'}


def test_exclude_classes_is_overridable():
    loader = _FakeLoader(['ACC', 'HAG'])
    result = extract_time_series(loader, window_size=4, verbose=False, exclude_classes=())
    assert set(result['class_names'].values()) == {'ACC', 'HAG'}


def test_max_groups_per_class_covers_every_class():
    # A flat global cap (the old `max_groups`) on an alphabetically-sorted group
    # list only ever hits the alphabetically-first class when the cap is small
    # relative to that class's group count — verified against the live HDF5
    # (max_groups=6 cached 100% class ABL). max_groups_per_class must not do this.
    classes = ['ABL', 'ACC', 'FOR', 'ICA', 'ROC']
    loader = _FakeLoader(classes, n_groups_per_class=6)
    result = extract_time_series(loader, window_size=4, verbose=False, max_groups_per_class=2)
    seen = {result['class_names'][lbl] for lbl in np.unique(result['y'])}
    assert seen == set(classes)


def test_groups_map_to_exactly_one_class():
    loader = _FakeLoader(['ACC', 'FOR', 'ROC'], n_groups_per_class=3)
    result = extract_time_series(loader, window_size=4, verbose=False)
    y, groups = result['y'], result['groups']
    for g in np.unique(groups):
        assert len(np.unique(y[groups == g])) == 1


def test_single_series_per_window_no_train_predict_split():
    loader = _FakeLoader(['ACC', 'FOR'], n_groups_per_class=2, T=6)
    result = extract_time_series(loader, window_size=4, verbose=False)
    assert 'X_train' not in result and 'X_predict' not in result
    assert result['X'][0].shape == (6, 16)  # (T, window_size**2)


# ---------------------------------------------------------------------------
# _load_cpazmal — K-fold splitting (synthetic on-disk cache, no HDF5)
# ---------------------------------------------------------------------------

def _write_fake_cache(cache_dir, n_per_class, n_groups_per_class, T=5, N=8, seed=0):
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    class_names = {}
    gi = 0
    for cls_idx, n_samples in enumerate(n_per_class):
        class_names[cls_idx] = f"CLS{cls_idx}"
        cls_groups = list(range(gi, gi + n_groups_per_class))
        gi += n_groups_per_class
        for i in range(n_samples):
            X.append(rng.uniform(0.1, 1.0, size=(T, N)))
            y.append(cls_idx)
            groups.append(cls_groups[i % n_groups_per_class])
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Suffix must match _load_cpazmal's own cache naming (data_utils.py):
    # X_{mgpc_part}_w{window_size}{scale_suffix}.npy — "all_w12" here since _base_cfg
    # leaves max_groups_per_class/window_size/scale_type at their defaults (None, 12,
    # amplitude).
    np.save(cache_dir / "X_all_w12.npy", np.array(X, dtype=object), allow_pickle=True)
    np.save(cache_dir / "y_all_w12.npy", np.array(y, dtype=np.int32))
    np.save(cache_dir / "groups_all_w12.npy", np.array(groups, dtype=np.int32))
    (cache_dir / "meta_all_w12.json").write_text(json.dumps({
        "class_names": {str(k): v for k, v in class_names.items()},
        "group_names": {str(g): f"G{g}" for g in range(gi)},
    }))


def _base_cfg(cache_dir, n_splits, group_aware=True, samples_per_step=None):
    return {
        "dataset": {"type": "cpazmal", "hdf5_path": "unused.hdf5", "cache_dir": str(cache_dir)},
        "classification": {} if samples_per_step is None else {"samples_per_step": samples_per_step},
        "cross_validation": {"n_splits": n_splits, "group_aware": group_aware},
    }


def test_load_cpazmal_kfold_groups_disjoint_across_folds(tmp_path):
    from data_utils import _load_cpazmal
    cache_dir = tmp_path / "cache1"
    _write_fake_cache(cache_dir, n_per_class=(18, 18, 18), n_groups_per_class=6)
    cfg = _base_cfg(cache_dir, n_splits=3)

    test_groups_by_fold = []
    for fold in range(3):
        data = _load_cpazmal(cfg, seed=42, fold=fold)
        test_groups_by_fold.append(set(data["groups_test"].tolist()))
        assert len(data["class_names"]) == 3

    assert all(
        test_groups_by_fold[i].isdisjoint(test_groups_by_fold[j])
        for i in range(3) for j in range(3) if i != j
    )


def test_load_cpazmal_fold_membership_independent_of_seed(tmp_path):
    from data_utils import _load_cpazmal
    cache_dir = tmp_path / "cache2"
    _write_fake_cache(cache_dir, n_per_class=(18, 18, 18), n_groups_per_class=6)
    cfg = _base_cfg(cache_dir, n_splits=3, samples_per_step=5)

    d0 = _load_cpazmal(cfg, seed=1, fold=1)
    d1 = _load_cpazmal(cfg, seed=2, fold=1)
    assert set(d0["groups_test"].tolist()) == set(d1["groups_test"].tolist())
    assert not np.allclose(d0["X_train"][0], d1["X_train"][0])  # sample RNG still varies


def test_load_cpazmal_group_aware_false_uses_stratified_kfold(tmp_path):
    from data_utils import _load_cpazmal
    cache_dir = tmp_path / "cache3"
    _write_fake_cache(cache_dir, n_per_class=(12, 12), n_groups_per_class=4)
    cfg = _base_cfg(cache_dir, n_splits=3, group_aware=False)

    data = _load_cpazmal(cfg, seed=42, fold=0)
    assert len(data["y_train"]) + len(data["y_test"]) == 24
    assert set(np.unique(data["y_test"]).tolist()) == {0, 1}  # still stratified by class


def test_load_cpazmal_class_names_propagated_from_cache(tmp_path):
    from data_utils import _load_cpazmal
    cache_dir = tmp_path / "cache4"
    _write_fake_cache(cache_dir, n_per_class=(10, 10), n_groups_per_class=3)
    cfg = _base_cfg(cache_dir, n_splits=2)

    data = _load_cpazmal(cfg, seed=42, fold=0)
    assert data["class_names"] == {0: "CLS0", 1: "CLS1"}
