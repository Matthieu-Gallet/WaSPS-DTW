"""Sensitivity analysis: sweep γ and n_train on synthetic exponential data.

Outputs two CSV files under <output_dir>/:
  sensitivity_gamma.csv   — F1 vs gamma for each of the 3 methods (KNN mode)
  sensitivity_ntrain.csv  — F1 vs n_train for each of the 3 methods (KNN mode)

Usage:
    python experiments/run_sensitivity.py [--output-dir results/jax_sensitivity]

Sweeps KNN mode only (faster than barycenter, sufficient for sensitivity).
STA is excluded (O(T²) — too slow even on synthetic data; use run_classification
with a smoke config to evaluate STA).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import jax
import numpy as np

_HERE = Path(__file__).parent
_SRC  = _HERE.parent / "src"
sys.path.insert(0, str(_SRC))

jax.config.update("jax_enable_x64", True)  # must precede jax.numpy usage
import distributions
from classification.nn import knn_predict as sdtw_knn
from costs import SqEuclidean, WaSPS


# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------

_RATES      = [1.0, 3.0, 7.0]
_N_TEST     = 4
_T          = 6
_N_SAMPLES  = 50
_STA_EPS    = 0.05


def _gen_dataset(n_train: int, seed: int):
    rng = np.random.default_rng(seed)
    train_raw, test_raw, train_labels, test_labels = [], [], [], []
    for cls_idx, rate in enumerate(_RATES):
        for _ in range(n_train):
            train_raw.append(rng.exponential(1.0 / rate, (_T, _N_SAMPLES)))
            train_labels.append(cls_idx)
        for _ in range(_N_TEST):
            test_raw.append(rng.exponential(1.0 / rate, (_T, _N_SAMPLES)))
            test_labels.append(cls_idx)
    return (train_raw, test_raw,
            np.array(train_labels), np.array(test_labels))


def _to_params(raw: list) -> list:
    # Synthetic exponential data is already clean; clean_time_series is safe here too.
    return [distributions.exponential.fit_time_series(s, dtype=np.float64) for s in raw]


def _f1(preds, truth):
    from sklearn.metrics import f1_score
    return float(f1_score(truth, preds, average="weighted", zero_division=0))


def _run_knn(method: str, train_repr, test_repr, train_raw, test_raw,
             train_l, test_l, gamma: float) -> float:
    if method == 'wasps':
        preds = sdtw_knn(train_repr, train_l, test_repr,
                         cost_fn=WaSPS("exponential", log_correction=True),
                         gamma=gamma, k=1)
    elif method == 'eucl_params':
        preds = sdtw_knn(train_repr, train_l, test_repr,
                         cost_fn=SqEuclidean(), gamma=gamma, k=1)
    elif method == 'eucl_raw':
        preds = sdtw_knn(train_raw, train_l, test_raw,
                         cost_fn=SqEuclidean(), gamma=gamma, k=1)
    else:
        raise ValueError(f"unknown method '{method}'")
    return _f1(preds, test_l)


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

_ALL_METHODS = ['wasps', 'eucl_params', 'eucl_raw']


def sweep_gamma(gammas: list, n_train: int, seed: int, out_csv: Path,
                methods=_ALL_METHODS):
    train_raw, test_raw, train_l, test_l = _gen_dataset(n_train, seed)
    p_train = _to_params(train_raw)
    p_test  = _to_params(test_raw)
    rows = []
    for gamma in gammas:
        print(f"  gamma={gamma:.3f}", end="", flush=True)
        row = {"gamma": gamma}
        for m in methods:
            tr = p_train if m in ('wasps', 'eucl_params') else train_raw
            te = p_test  if m in ('wasps', 'eucl_params') else test_raw
            row[m] = _run_knn(m, tr, te, train_raw, test_raw, train_l, test_l, gamma)
        print("  " + "  ".join(f"{m}={row[m]:.3f}" for m in methods))
        rows.append(row)
    _write_csv(out_csv, ["gamma"] + methods, rows)
    print(f"  saved {out_csv}")


def sweep_ntrain(n_trains: list, gamma: float, seed: int, out_csv: Path,
                 methods=_ALL_METHODS):
    rows = []
    for n_train in n_trains:
        print(f"  n_train={n_train}", end="", flush=True)
        train_raw, test_raw, train_l, test_l = _gen_dataset(n_train, seed)
        p_train = _to_params(train_raw)
        p_test  = _to_params(test_raw)
        row = {"n_train": n_train}
        for m in methods:
            tr = p_train if m in ('wasps', 'eucl_params') else train_raw
            te = p_test  if m in ('wasps', 'eucl_params') else test_raw
            row[m] = _run_knn(m, tr, te, train_raw, test_raw, train_l, test_l, gamma)
        print("  " + "  ".join(f"{m}={row[m]:.3f}" for m in methods))
        rows.append(row)
    _write_csv(out_csv, ["n_train"] + methods, rows)
    print(f"  saved {out_csv}")


def _write_csv(path: Path, fields: list, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/jax_sensitivity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train-fixed", type=int, default=8)
    parser.add_argument("--gamma-fixed",   type=float, default=1.0)
    parser.add_argument("--methods", nargs="+", default=_ALL_METHODS,
                        choices=_ALL_METHODS, metavar="METHOD",
                        help="Subset of methods to run (default: all 3)")
    args = parser.parse_args()

    out = Path(args.output_dir)

    gammas   = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    n_trains = [2, 4, 6, 8, 12, 16, 24]

    print(f"[gamma sweep] n_train={args.n_train_fixed}, seed={args.seed}, "
          f"methods={args.methods}")
    sweep_gamma(gammas, args.n_train_fixed, args.seed,
                out / "sensitivity_gamma.csv", methods=args.methods)

    print(f"\n[n_train sweep] gamma={args.gamma_fixed}, seed={args.seed}, "
          f"methods={args.methods}")
    sweep_ntrain(n_trains, args.gamma_fixed, args.seed,
                 out / "sensitivity_ntrain.csv", methods=args.methods)


if __name__ == "__main__":
    main()
