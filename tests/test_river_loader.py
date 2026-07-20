"""Tests for data/river_loader.py — synthetic .npy fixture, no real dataset needed."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data.river_loader import load_river_classification, _aggregate_days


def _write_river_npy(data_dir: Path, n=40, T=6, M=3, n_classes=4, with_groups=False,
                     n_groups=8, mode="balanced"):
    """Write a synthetic river-shaped dataset: X (N, T, M), Y (N,), metadata, optionally groups."""
    rng = np.random.default_rng(0)
    X = rng.exponential(1.0, size=(n, T, M)).astype(np.float64)
    Y = np.tile(np.arange(n_classes), n // n_classes)
    metadata = {"idx_to_regime": {i: f"class_{i}" for i in range(n_classes)}}

    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / f"X_{mode}.npy", X)
    np.save(data_dir / f"Y_{mode}.npy", Y)
    np.save(data_dir / f"metadata_{mode}.npy", metadata, allow_pickle=True)
    if with_groups:
        groups = np.tile(np.arange(n_groups), n // n_groups)
        np.save(data_dir / f"groups_{mode}.npy", groups)
    return X, Y, metadata


# ---------------------------------------------------------------------------
# _aggregate_days
# ---------------------------------------------------------------------------

def test_aggregate_days_pools_and_concatenates():
    arr = np.arange(12).reshape(6, 2).astype(np.float64)  # (T=6, M=2)
    out = _aggregate_days(arr, k=3)
    assert out.shape == (2, 6)  # T//k=2, k*M=6


def test_aggregate_days_noop_for_k_le_1():
    arr = np.arange(6).reshape(3, 2).astype(np.float64)
    assert np.array_equal(_aggregate_days(arr, k=1), arr)


def test_aggregate_days_discards_trailing_timesteps():
    arr = np.arange(10).reshape(5, 2).astype(np.float64)  # T=5, not divisible by k=2
    out = _aggregate_days(arr, k=2)
    assert out.shape == (2, 4)  # floor(5/2)=2, trailing timestep dropped


# ---------------------------------------------------------------------------
# load_river_classification — holdout (n_splits=1, default)
# ---------------------------------------------------------------------------

def test_holdout_split_shapes(tmp_path):
    _write_river_npy(tmp_path, n=40, T=6, M=3, n_classes=4)
    data = load_river_classification(str(tmp_path), test_size=0.25, seed=0)
    assert len(data["X_train"]) + len(data["X_test"]) == 40
    assert len(data["X_train"]) == len(data["y_train"])
    assert len(data["X_test"]) == len(data["y_test"])
    assert data["X_train"][0].shape == (6, 3)
    assert data["groups_train"] is None
    assert data["groups_test"] is None


def test_holdout_split_is_stratified(tmp_path):
    _write_river_npy(tmp_path, n=40, T=6, M=3, n_classes=4)
    data = load_river_classification(str(tmp_path), test_size=0.5, seed=0)
    # balanced source (10/class) + stratified split -> both splits keep all 4 classes
    assert set(np.unique(data["y_train"])) == {0, 1, 2, 3}
    assert set(np.unique(data["y_test"])) == {0, 1, 2, 3}


def test_holdout_split_disjoint(tmp_path):
    _write_river_npy(tmp_path, n=40, T=6, M=3, n_classes=4)
    data = load_river_classification(str(tmp_path), test_size=0.25, seed=1)
    train_ids = {tuple(x.ravel()) for x in data["X_train"]}
    test_ids = {tuple(x.ravel()) for x in data["X_test"]}
    assert train_ids.isdisjoint(test_ids)


def test_class_names_from_metadata(tmp_path):
    _write_river_npy(tmp_path, n=40, T=6, M=3, n_classes=4)
    data = load_river_classification(str(tmp_path), seed=0)
    assert data["class_names"] == {0: "class_0", 1: "class_1", 2: "class_2", 3: "class_3"}


def test_missing_file_raises_with_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="Explore2_HydroDataset"):
        load_river_classification(str(tmp_path), mode="balanced")


# ---------------------------------------------------------------------------
# K-fold — group-aware and StratifiedKFold fallback
# ---------------------------------------------------------------------------

def test_group_aware_kfold_keeps_groups_disjoint(tmp_path):
    _write_river_npy(tmp_path, n=40, T=6, M=3, n_classes=4, with_groups=True, n_groups=8)
    for fold in range(5):
        data = load_river_classification(str(tmp_path), n_splits=5, fold=fold,
                                         group_aware=True, seed=0)
        assert set(data["groups_train"]).isdisjoint(set(data["groups_test"]))


def test_stratified_kfold_fallback_without_groups_file(tmp_path, capsys):
    _write_river_npy(tmp_path, n=40, T=6, M=3, n_classes=4, with_groups=False)
    data = load_river_classification(str(tmp_path), n_splits=5, fold=0,
                                     group_aware=True, seed=0)
    assert "falling back to StratifiedKFold" in capsys.readouterr().out
    assert data["groups_train"] is None


def test_fold_out_of_range_raises(tmp_path):
    _write_river_npy(tmp_path, n=40, T=6, M=3, n_classes=4)
    with pytest.raises(ValueError, match="out of range"):
        load_river_classification(str(tmp_path), n_splits=5, fold=5, seed=0)


def test_cv_seed_decoupled_from_seed_for_kfold(tmp_path):
    # same fold assignment across different `seed` values (StratifiedKFold fallback
    # is controlled by cv_seed, not seed) — only the to_fixed_n RNG should vary
    _write_river_npy(tmp_path, n=40, T=6, M=3, n_classes=4)
    d1 = load_river_classification(str(tmp_path), n_splits=5, fold=0, seed=1, cv_seed=7)
    d2 = load_river_classification(str(tmp_path), n_splits=5, fold=0, seed=2, cv_seed=7)
    assert np.array_equal(d1["y_train"], d2["y_train"])
    assert np.array_equal(d1["y_test"], d2["y_test"])


# ---------------------------------------------------------------------------
# max_time_steps / samples_per_step / aggregate_days
# ---------------------------------------------------------------------------

def test_max_time_steps_truncates(tmp_path):
    _write_river_npy(tmp_path, n=20, T=10, M=3, n_classes=4)
    data = load_river_classification(str(tmp_path), max_time_steps=4, seed=0)
    assert data["X_train"][0].shape[0] == 4


def test_samples_per_step_rectangularises(tmp_path):
    _write_river_npy(tmp_path, n=20, T=6, M=3, n_classes=4)
    data = load_river_classification(str(tmp_path), samples_per_step=7, seed=0)
    assert data["X_train"][0].shape == (6, 7)
    assert np.all(np.isfinite(data["X_train"][0]))


def test_aggregate_days_reduces_T(tmp_path):
    _write_river_npy(tmp_path, n=20, T=6, M=3, n_classes=4)
    data = load_river_classification(str(tmp_path), aggregate_days=3, seed=0)
    assert data["X_train"][0].shape == (2, 9)  # T=6//3=2, M*3=9
