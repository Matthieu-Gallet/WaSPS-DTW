"""Sensitivity analysis: sweep HP and data-distortion on synthetic exponential data.

Sweeps (select with --sweep):
  s0a  — Calibration: lr ∈ [1e-3, 5e-3, 1e-2, 5e-2, 1e-1] for barycenter mode (1 seed)
  s0b  — Calibration: k ∈ [1, 3, 5, 7]                    for KNN mode        (1 seed)
  s1   — Main: gamma ∈ [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]                 (n seeds)
  s2   — Main: n_train ∈ [2, 4, 6, 8, 12, 16, 24]                             (n seeds)
  s3   — Main: decimation fraction ∈ [0, 0.05, 0.1, 0.2, 0.3, 0.5]           (n seeds)

Without --sweep: runs s1 + s2 (backward compatible, KNN mode, 1 seed).

Usage:
    python experiments/run_sensitivity.py                              # legacy: s1+s2
    python experiments/run_sensitivity.py --sweep s0a                 # calibrate lr
    python experiments/run_sensitivity.py --sweep s1 --n-seeds 5      # gamma sweep
    python experiments/run_sensitivity.py --sweep s3 --n-seeds 5 \\
        --lr-fixed 1e-2 --k-fixed 1 --gamma-fixed 1.0

Outputs (under <output-dir>/):
  sensitivity_gamma.csv    / sensitivity_gamma_s1.csv
  sensitivity_ntrain.csv   / sensitivity_ntrain_s2.csv
  sensitivity_lr.csv       (s0a)
  sensitivity_k.csv        (s0b)
  sensitivity_decimation.csv (s3)

STA excluded: O(T²) even on tiny synthetic data — use run_classification smoke config.
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

jax.config.update("jax_enable_x64", True)
import distributions
from classification.nn import knn_predict as sdtw_knn
from costs import SqEuclidean, WaSPS

from method_defs import make_cost_fn as _make_cost_fn, make_softdtw_bary as _make_softdtw_bary


# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------

_RATES     = [1.0, 3.0, 7.0]
_N_TEST    = 4
_T         = 6
_N_SAMPLES = 50
_STA_EPS   = 0.05
_FAMILY    = "exponential"

_GAMMAS    = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
_N_TRAINS  = [2, 4, 6, 8, 12, 16, 24]
_LRS       = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
_KS        = [1, 3, 5, 7]
_FRACTIONS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]

_ALL_METHODS = ['wasps', 'eucl_params', 'eucl_raw']


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

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
    return [distributions.exponential.fit_time_series(s, dtype=np.float64) for s in raw]


def decimate_series(series: list, fraction: float, rng: np.random.Generator) -> list:
    """Remove `fraction` of timesteps independently per series.

    All output series have the same T' = max(2, T − floor(T*fraction)), but
    each series keeps a different random subset of timesteps → temporal
    distortion without shape inconsistency.

    Args:
        series:   List of (T, ...) arrays with identical T.
        fraction: Fraction of timesteps to remove (0.0 → no change, 0.5 → half removed).
        rng:      Seeded NumPy generator (consumed in-place — pass same rng for reproducibility).

    Returns:
        List of (T', ...) arrays.
    """
    T = np.asarray(series[0]).shape[0]
    n_keep = max(2, T - int(T * fraction))
    return [np.asarray(s)[np.sort(rng.choice(T, n_keep, replace=False))]
            for s in series]


# ---------------------------------------------------------------------------
# KNN / barycenter evaluation for one (method, split) combo
# ---------------------------------------------------------------------------

def _f1(preds, truth):
    from sklearn.metrics import f1_score
    return float(f1_score(truth, preds, average="weighted", zero_division=0))


def _run_knn(method: str, train_repr, test_repr, train_l, test_l,
             gamma: float, k: int = 1) -> float:
    cost_fn = _make_cost_fn(method, _FAMILY, _STA_EPS)
    preds = sdtw_knn(train_repr, train_l, test_repr,
                     cost_fn=cost_fn, gamma=gamma, k=k)
    return _f1(preds, test_l)


def _run_bary(method: str, train_repr, test_repr, train_l, test_l,
              gamma: float, lr: float, n_steps: int = 100) -> float:
    from classification.barycenter_clf import fit_barycenters, predict
    softdtw_bary = _make_softdtw_bary(method, _FAMILY, _STA_EPS, gamma)
    cost_fn      = _make_cost_fn(method, _FAMILY, _STA_EPS)
    bary = fit_barycenters(train_repr, train_l, softdtw_bary,
                           n_steps=n_steps, lr=lr)
    preds = predict(test_repr, bary, cost_fn, gamma)
    return _f1(preds, test_l)


def _get_reprs(train_raw, test_raw, method: str):
    if method == 'eucl_raw':
        return train_raw, test_raw
    p_train = _to_params(train_raw)
    p_test  = _to_params(test_raw)
    return p_train, p_test


# ---------------------------------------------------------------------------
# Multi-seed sweep helper
# ---------------------------------------------------------------------------

def _sweep_seeds(
    sweep_fn,
    n_seeds: int,
    base_seed: int,
    **kwargs,
) -> dict:
    """Run sweep_fn(seed, **kwargs) for n_seeds seeds; return {val: {method: [f1,...]}}."""
    from collections import defaultdict
    per_val: dict = defaultdict(lambda: {m: [] for m in _ALL_METHODS})
    for i in range(n_seeds):
        seed_results = sweep_fn(seed=base_seed + i, **kwargs)
        for val, row in seed_results.items():
            for m, f1 in row.items():
                per_val[val][m].append(f1)
    return per_val


def _agg_rows(per_val: dict, key_name: str, methods: list) -> list:
    """Aggregate {val: {method: [f1s]}} → list of CSV row dicts (mean ± std)."""
    rows = []
    for val in sorted(per_val.keys()):
        row: dict = {key_name: val}
        for m in methods:
            vals = per_val[val][m]
            row[f"{m}_mean"] = float(np.mean(vals)) if vals else float('nan')
            row[f"{m}_std"]  = float(np.std(vals))  if vals else float('nan')
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Sweep S1 — gamma
# ---------------------------------------------------------------------------

def _sweep_gamma_one(seed: int, n_train: int, gammas: list, k: int,
                     methods: list, mode: str, lr: float) -> dict:
    """Single-seed gamma sweep. Returns {gamma: {method: f1}}."""
    train_raw, test_raw, train_l, test_l = _gen_dataset(n_train, seed)
    results: dict = {}
    for gamma in gammas:
        row = {}
        for m in methods:
            tr, te = _get_reprs(train_raw, test_raw, m)
            if mode == 'knn':
                row[m] = _run_knn(m, tr, te, train_l, test_l, gamma, k=k)
            else:
                row[m] = _run_bary(m, tr, te, train_l, test_l, gamma, lr)
        results[gamma] = row
        print(f"    gamma={gamma:.3g}  " + "  ".join(f"{m}={row[m]:.3f}" for m in methods),
              flush=True)
    return results


def sweep_gamma_multi(
    gammas: list,
    n_train: int,
    n_seeds: int,
    base_seed: int,
    gamma_ref: float,      # unused here — passed for symmetry
    lr_ref: float,
    k_ref: int,
    out_csv: Path,
    methods: list = _ALL_METHODS,
    mode: str = 'knn',
):
    per_val = _sweep_seeds(
        _sweep_gamma_one,
        n_seeds=n_seeds, base_seed=base_seed,
        n_train=n_train, gammas=gammas, k=k_ref,
        methods=methods, mode=mode, lr=lr_ref,
    )
    fields = ["gamma"] + [f"{m}_{s}" for m in methods for s in ("mean", "std")]
    rows   = _agg_rows(per_val, "gamma", methods)
    _write_csv(out_csv, fields, rows)
    print(f"  saved {out_csv}")


# ---------------------------------------------------------------------------
# Sweep S2 — n_train
# ---------------------------------------------------------------------------

def _sweep_ntrain_one(seed: int, n_trains: list, gamma: float, k: int,
                      methods: list, mode: str, lr: float) -> dict:
    results: dict = {}
    for n_train in n_trains:
        train_raw, test_raw, train_l, test_l = _gen_dataset(n_train, seed)
        row = {}
        for m in methods:
            tr, te = _get_reprs(train_raw, test_raw, m)
            if mode == 'knn':
                row[m] = _run_knn(m, tr, te, train_l, test_l, gamma, k=k)
            else:
                row[m] = _run_bary(m, tr, te, train_l, test_l, gamma, lr)
        results[n_train] = row
        print(f"    n_train={n_train}  " + "  ".join(f"{m}={row[m]:.3f}" for m in methods),
              flush=True)
    return results


def sweep_ntrain_multi(
    n_trains: list,
    gamma: float,
    n_seeds: int,
    base_seed: int,
    lr_ref: float,
    k_ref: int,
    out_csv: Path,
    methods: list = _ALL_METHODS,
    mode: str = 'knn',
):
    per_val = _sweep_seeds(
        _sweep_ntrain_one,
        n_seeds=n_seeds, base_seed=base_seed,
        n_trains=n_trains, gamma=gamma, k=k_ref,
        methods=methods, mode=mode, lr=lr_ref,
    )
    fields = ["n_train"] + [f"{m}_{s}" for m in methods for s in ("mean", "std")]
    rows   = _agg_rows(per_val, "n_train", methods)
    _write_csv(out_csv, fields, rows)
    print(f"  saved {out_csv}")


# ---------------------------------------------------------------------------
# Sweep S3 — decimation
# ---------------------------------------------------------------------------

def _sweep_decimation_one(seed: int, fractions: list, n_train: int, gamma: float,
                          k: int, methods: list, mode: str, lr: float) -> dict:
    train_raw, test_raw, train_l, test_l = _gen_dataset(n_train, seed)
    results: dict = {}
    for fraction in fractions:
        rng = np.random.default_rng(seed + int(fraction * 1000))
        d_train = decimate_series(train_raw, fraction, rng)
        d_test  = decimate_series(test_raw,  fraction, rng)
        row = {}
        for m in methods:
            tr, te = _get_reprs(d_train, d_test, m)
            if mode == 'knn':
                row[m] = _run_knn(m, tr, te, train_l, test_l, gamma, k=k)
            else:
                row[m] = _run_bary(m, tr, te, train_l, test_l, gamma, lr)
        results[fraction] = row
        print(f"    fraction={fraction:.2f}  " + "  ".join(f"{m}={row[m]:.3f}" for m in methods),
              flush=True)
    return results


def sweep_decimation_multi(
    fractions: list,
    n_train: int,
    gamma: float,
    n_seeds: int,
    base_seed: int,
    lr_ref: float,
    k_ref: int,
    out_csv: Path,
    methods: list = _ALL_METHODS,
    mode: str = 'knn',
):
    per_val = _sweep_seeds(
        _sweep_decimation_one,
        n_seeds=n_seeds, base_seed=base_seed,
        fractions=fractions, n_train=n_train, gamma=gamma,
        k=k_ref, methods=methods, mode=mode, lr=lr_ref,
    )
    fields = ["fraction"] + [f"{m}_{s}" for m in methods for s in ("mean", "std")]
    rows   = _agg_rows(per_val, "fraction", methods)
    _write_csv(out_csv, fields, rows)
    print(f"  saved {out_csv}")


# ---------------------------------------------------------------------------
# Sweep S0a — lr calibration (barycenter)
# ---------------------------------------------------------------------------

def sweep_lr(
    lrs: list,
    n_train: int,
    gamma: float,
    seed: int,
    out_csv: Path,
    methods: list = _ALL_METHODS,
    n_steps: int = 100,
):
    """Single-seed lr sweep for barycenter mode — calibration only."""
    train_raw, test_raw, train_l, test_l = _gen_dataset(n_train, seed)
    rows = []
    for lr in lrs:
        print(f"  lr={lr:.5g}", end="", flush=True)
        row = {"lr": lr}
        for m in methods:
            tr, te = _get_reprs(train_raw, test_raw, m)
            row[m] = _run_bary(m, tr, te, train_l, test_l, gamma, lr, n_steps)
        print("  " + "  ".join(f"{m}={row[m]:.3f}" for m in methods))
        rows.append(row)
    _write_csv(out_csv, ["lr"] + methods, rows)
    print(f"  saved {out_csv}")
    best_lr = lrs[int(np.argmax([sum(r[m] for m in methods) for r in rows]))]
    print(f"  → best lr by sum-F1: {best_lr:.5g}")
    return best_lr


# ---------------------------------------------------------------------------
# Sweep S0b — k calibration (KNN)
# ---------------------------------------------------------------------------

def sweep_k(
    ks: list,
    n_train: int,
    gamma: float,
    seed: int,
    out_csv: Path,
    methods: list = _ALL_METHODS,
):
    """Single-seed k sweep for KNN mode — calibration only."""
    train_raw, test_raw, train_l, test_l = _gen_dataset(n_train, seed)
    p_train = _to_params(train_raw)
    p_test  = _to_params(test_raw)
    rows = []
    for k in ks:
        print(f"  k={k}", end="", flush=True)
        row = {"k": k}
        for m in methods:
            tr = p_train if m != 'eucl_raw' else train_raw
            te = p_test  if m != 'eucl_raw' else test_raw
            row[m] = _run_knn(m, tr, te, train_l, test_l, gamma, k=k)
        print("  " + "  ".join(f"{m}={row[m]:.3f}" for m in methods))
        rows.append(row)
    _write_csv(out_csv, ["k"] + methods, rows)
    print(f"  saved {out_csv}")
    best_k = ks[int(np.argmax([sum(r[m] for m in methods) for r in rows]))]
    print(f"  → best k by sum-F1: {best_k}")
    return best_k


# ---------------------------------------------------------------------------
# Legacy single-seed sweeps (backward compat)
# ---------------------------------------------------------------------------

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
            tr = p_train if m != 'eucl_raw' else train_raw
            te = p_test  if m != 'eucl_raw' else test_raw
            row[m] = _run_knn(m, tr, te, train_l, test_l, gamma, k=1)
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
            tr = p_train if m != 'eucl_raw' else train_raw
            te = p_test  if m != 'eucl_raw' else test_raw
            row[m] = _run_knn(m, tr, te, train_l, test_l, gamma, k=1)
        print("  " + "  ".join(f"{m}={row[m]:.3f}" for m in methods))
        rows.append(row)
    _write_csv(out_csv, ["n_train"] + methods, rows)
    print(f"  saved {out_csv}")


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------

def _write_csv(path: Path, fields: list, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="WaSPS-DTW sensitivity analysis")
    parser.add_argument("--output-dir",    default="results/jax_sensitivity")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--n-seeds",       type=int,   default=1,
                        help="Seeds for main sweeps (s1/s2/s3)")
    parser.add_argument("--n-train-fixed", type=int,   default=8)
    parser.add_argument("--gamma-fixed",   type=float, default=1.0)
    parser.add_argument("--lr-fixed",      type=float, default=1e-2,
                        help="Learning rate (barycenter mode, post-calibration)")
    parser.add_argument("--k-fixed",       type=int,   default=1,
                        help="k for KNN (post-calibration)")
    parser.add_argument("--mode",          default="knn",
                        choices=["knn", "barycenter", "both"],
                        help="Evaluation mode for s1/s2/s3")
    parser.add_argument("--sweep",         default=None,
                        choices=["s0a", "s0b", "s1", "s2", "s3"],
                        help="Sweep to run (default: s1+s2 legacy mode)")
    parser.add_argument("--methods", nargs="+", default=_ALL_METHODS,
                        choices=_ALL_METHODS, metavar="METHOD")
    args = parser.parse_args()

    out      = Path(args.output_dir)
    methods  = args.methods
    modes    = ["knn", "barycenter"] if args.mode == "both" else [args.mode]

    if args.sweep is None:
        # Legacy mode: s1 + s2, single seed, KNN only
        print(f"[gamma sweep] n_train={args.n_train_fixed}, seed={args.seed}")
        sweep_gamma(_GAMMAS, args.n_train_fixed, args.seed,
                    out / "sensitivity_gamma.csv", methods=methods)
        print(f"\n[n_train sweep] gamma={args.gamma_fixed}, seed={args.seed}")
        sweep_ntrain(_N_TRAINS, args.gamma_fixed, args.seed,
                     out / "sensitivity_ntrain.csv", methods=methods)
        return

    if args.sweep == "s0a":
        print(f"[S0a lr calibration] gamma={args.gamma_fixed}  n_train={args.n_train_fixed}  seed={args.seed}")
        sweep_lr(_LRS, args.n_train_fixed, args.gamma_fixed, args.seed,
                 out / "sensitivity_lr.csv", methods=methods)

    elif args.sweep == "s0b":
        print(f"[S0b k calibration] gamma={args.gamma_fixed}  n_train={args.n_train_fixed}  seed={args.seed}")
        sweep_k(_KS, args.n_train_fixed, args.gamma_fixed, args.seed,
                out / "sensitivity_k.csv", methods=methods)

    elif args.sweep == "s1":
        for mode in modes:
            tag = f"s1_{mode}" if args.mode == "both" else "s1"
            out_csv = out / f"sensitivity_gamma_{tag}.csv"
            print(f"[S1 gamma sweep] mode={mode}  n_train={args.n_train_fixed}  "
                  f"n_seeds={args.n_seeds}  k={args.k_fixed}  lr={args.lr_fixed}")
            sweep_gamma_multi(
                gammas=_GAMMAS, n_train=args.n_train_fixed,
                n_seeds=args.n_seeds, base_seed=args.seed,
                gamma_ref=args.gamma_fixed, lr_ref=args.lr_fixed,
                k_ref=args.k_fixed, out_csv=out_csv,
                methods=methods, mode=mode,
            )

    elif args.sweep == "s2":
        for mode in modes:
            tag = f"s2_{mode}" if args.mode == "both" else "s2"
            out_csv = out / f"sensitivity_ntrain_{tag}.csv"
            print(f"[S2 n_train sweep] mode={mode}  gamma={args.gamma_fixed}  "
                  f"n_seeds={args.n_seeds}  k={args.k_fixed}  lr={args.lr_fixed}")
            sweep_ntrain_multi(
                n_trains=_N_TRAINS, gamma=args.gamma_fixed,
                n_seeds=args.n_seeds, base_seed=args.seed,
                lr_ref=args.lr_fixed, k_ref=args.k_fixed,
                out_csv=out_csv, methods=methods, mode=mode,
            )

    elif args.sweep == "s3":
        for mode in modes:
            tag = f"s3_{mode}" if args.mode == "both" else "s3"
            out_csv = out / f"sensitivity_decimation_{tag}.csv"
            print(f"[S3 decimation sweep] mode={mode}  n_train={args.n_train_fixed}  "
                  f"gamma={args.gamma_fixed}  n_seeds={args.n_seeds}")
            sweep_decimation_multi(
                fractions=_FRACTIONS, n_train=args.n_train_fixed,
                gamma=args.gamma_fixed, n_seeds=args.n_seeds,
                base_seed=args.seed, lr_ref=args.lr_fixed,
                k_ref=args.k_fixed, out_csv=out_csv,
                methods=methods, mode=mode,
            )


if __name__ == "__main__":
    main()
