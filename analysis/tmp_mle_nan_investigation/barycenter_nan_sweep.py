"""Step 2 — WaSPS barycenter NaN-failure sweep (real regime, river.yaml hyperparameters).

Mechanism under test (see beta_distribution.py + findings.md for the derivation):
  W2sq_exponential(b1,b2) = 2(mu1-mu2)^2 with mu=1/b -> the Frechet barycenter
  minimizes in mu-space, i.e. barycenter mu = arithmetic mean of 1/b_i. MLE's fatter
  low-beta tail (min 0.0026 vs log_cumulant's 0.0067, ~5x lower at p1) means a single
  MLE-estimated sample pulls that mean harder -> smaller barycenter beta iterate ->
  gradient ~4(b1-b2)/(b1^3 b2) explodes as the ITERATE b1 (not the input data) -> 0.

_inverse_softplus's clip (1e-5 vs 1e-8) only bounds the INPUT theta-conversion
(init + training series); the optimizer iterate z is never clipped inside _step_jit
(jax.nn.softplus(z) applied raw). Since input beta min (0.0026) is >> both floors,
the floor arms are expected to show NO difference — that null result is itself the
evidence that the real lever (if failures occur) is an inside-loop clamp on the
iterate, not the input floor. This sweep tests that prediction.

Trimmed design (one arm is load-bearing, the rest are confirmatory — not a full
factorial, see findings.md for why):
  A1. mle / floor=1e-5 / sgd        — 5 folds x 4 seeds = 20 trials (THE result: does
      MLE fail at real settings, with the real default optimizer)
  A2. log_cumulant / floor=1e-5 / sgd — 5 folds x 4 seeds = 20 trials (THE discriminator:
      does the user's actual working default fail less)
  A3. mle / floor=1e-8 / sgd        — 3 trials, confirms the predicted null (input beta
      min 0.0026 >> both floors, so the floor arms should not differ)
  A4. mle / floor=1e-5 / adam       — 5 trials (secondary: optimizer interaction)
  A5. log_cumulant / floor=1e-5 / adam — 5 trials (secondary: optimizer interaction)
  B.  eucl_params / mle / 1e-5 / sgd — 5 trials (control: does the same input tail
      break a method whose gradient doesn't have the WaSPS 1/beta^3 structure)
  C.  wasps / mle / 1e-5 / sgd, river.yaml's REAL stratified TOTAL-20 cap (not this
      sweep's flat 5/class) — 5 trials (rare classes like NG can land at 2-3/class
      under the real cap, more outlier exposure than flat-5/class)

For NaN runs: re-fits the failing (fold, seed, class) alone with verbose=True to locate
the divergence step — capped at MAX_TRACES total (not per trial) via a module-level
counter, since at ~100% NaN rate tracing every failure means a second full fit per
trial, which is what made an earlier, untrimmed version of this sweep ~4x slower than
estimated. A handful of traces is enough to characterize where divergence happens.
For every run (NaN or not): records the resulting barycenter's min beta — informative
either way, since it shows how close the iterate came even when it didn't fully NaN.

Run: python analysis/tmp_mle_nan_investigation/barycenter_nan_sweep.py
Writes: barycenter_nan_sweep.csv, nan_trace_log.txt (this directory).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_SRC = _HERE.parent.parent / "src"
_EXPERIMENTS = _HERE.parent.parent / "experiments"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_EXPERIMENTS))

import jax
jax.config.update("jax_enable_x64", True)

import distributions
import costs as costs_module
from data.preprocess import clean_time_series
from data.river_loader import load_river_classification
from costs import WaSPS, SqEuclidean
from softdtw import SoftDTW
from barycenter import fit_barycenter
from classification.barycenter_clf import fit_barycenters
from data_utils import subsample as _subsample

DATA_DIR = str(_HERE.parent.parent / "data" / "river")
N_SPLITS = 5
SEEDS = [42, 43, 44, 45]
SAMPLES_PER_STEP = 50    # river.yaml
MAX_TRAIN_PER_CLASS = 5  # river.yaml max_train_samples=20 / 4 classes (flat per-class cap)
MAX_TRAIN_TOTAL = 20     # river.yaml's REAL cap (stratified total, arm C)
GAMMA = 1.0               # river.yaml
N_STEPS = 200             # river.yaml — kept full-scale, NOT reduced (see module docstring)
LR = 0.01                 # river.yaml
PATIENCE = 20             # river.yaml
MIN_REL_IMPROVE = 1e-4    # river.yaml

_ORIG_INV_SOFTPLUS = costs_module._inverse_softplus


def _make_floor_patch(floor: float):
    def patched(x):
        import jax.numpy as jnp
        return jnp.log(jnp.expm1(jnp.clip(x, floor, 500.0)))
    return patched


def load_fold_flat_cap(fold: int, seed: int):
    """Flat MAX_TRAIN_PER_CLASS per class (arms A, B)."""
    data = load_river_classification(
        DATA_DIR, mode="balanced", n_splits=N_SPLITS, fold=fold, group_aware=True,
        samples_per_step=SAMPLES_PER_STEP, seed=seed, cv_seed=42,
    )
    X_train, y_train = data["X_train"], data["y_train"]
    classes = sorted(set(y_train.tolist()))
    rng = np.random.default_rng(seed)
    A, B = [], []
    for cls in classes:
        idx = [i for i, l in enumerate(y_train) if l == cls]
        rng.shuffle(idx)
        idx = idx[:MAX_TRAIN_PER_CLASS]
        A.extend([X_train[i] for i in idx])
        B.extend([cls] * len(idx))
    return A, np.array(B)


def load_fold_real_cap(fold: int, seed: int):
    """river.yaml's real stratified TOTAL-20 cap (arm C)."""
    data = load_river_classification(
        DATA_DIR, mode="balanced", n_splits=N_SPLITS, fold=fold, group_aware=True,
        samples_per_step=SAMPLES_PER_STEP, seed=seed, cv_seed=42,
    )
    rng = np.random.default_rng(seed)
    X_train, y_train = _subsample(data["X_train"], data["y_train"], MAX_TRAIN_TOTAL, rng)
    return X_train, y_train


dist_exp = distributions.get("exponential")


def estimate(X: list, y: np.ndarray, method: str) -> tuple:
    """Returns (valid_params, valid_labels) — series with any-NaN params dropped."""
    params = [dist_exp.fit_time_series(clean_time_series(s), dtype=np.float64, method=method) for s in X]
    valid_mask = [not np.isnan(p).any() for p in params]
    valid_params = [p for p, ok in zip(params, valid_mask) if ok]
    valid_labels = np.array([l for l, ok in zip(y, valid_mask) if ok])
    return valid_params, valid_labels


def make_softdtw(bary_method: str, optimizer_unused=None):
    if bary_method == "wasps":
        return SoftDTW(WaSPS("exponential", use_positivity_constraint=True),
                       GAMMA, is_divergence=True, manual_grad=True)
    return SoftDTW(SqEuclidean(use_positivity_constraint=True),
                   GAMMA, is_divergence=True, manual_grad=False)


def run_trial(X_train, y_train, bary_method: str, optimizer: str):
    """Fits one barycenter per class. Returns (any_nan, min_beta_overall, per_class_min)."""
    sdtw = make_softdtw(bary_method)
    bary = fit_barycenters(
        X_train, y_train, sdtw,
        n_steps=N_STEPS, lr=LR, patience=PATIENCE, min_rel_improve=MIN_REL_IMPROVE,
        optimizer=optimizer, n_jobs=1, verbose=False,
    )
    per_class_min = {cls: (float(np.nanmin(b)) if not np.isnan(b).all() else float('nan'))
                     for cls, b in bary.items()}
    any_nan = any(np.isnan(b).any() for b in bary.values())
    finite_mins = [v for v in per_class_min.values() if not np.isnan(v)]
    min_beta = min(finite_mins) if finite_mins else float('nan')
    return any_nan, min_beta, bary


MAX_TRACES = 3
_trace_count = 0


def trace_nan_class(cls_series, bary_method: str, optimizer: str, trace_log, label: str) -> None:
    """Re-fit a single failing class alone with verbose=True to locate the divergence step.

    Capped at MAX_TRACES total (module-level counter) — see module docstring: tracing
    every NaN at a ~100% failure rate means a second full fit per trial, dominating cost.
    """
    global _trace_count
    if _trace_count >= MAX_TRACES:
        return
    _trace_count += 1
    import io
    import contextlib
    sdtw = make_softdtw(bary_method)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fit_barycenter(cls_series, sdtw, n_steps=N_STEPS, lr=LR, patience=PATIENCE,
                       min_rel_improve=MIN_REL_IMPROVE, optimizer=optimizer, verbose=True)
    trace_log.write(f"[{label}]\n")
    trace_log.write(buf.getvalue())
    trace_log.write("\n" + "=" * 60 + "\n")
    trace_log.flush()


def _run_arm(rows, trace_log, arm: str, bary_method: str, cap: str, est_method: str,
            floor: float, optimizer: str, fold_seed_pairs: list) -> None:
    load_fn = load_fold_real_cap if cap == "real_total20" else load_fold_flat_cap
    if floor != 1e-5:
        costs_module._inverse_softplus = _make_floor_patch(floor)
    try:
        for fold, seed in fold_seed_pairs:
            X_raw, y_raw = load_fn(fold, seed)
            valid, y_valid = estimate(X_raw, y_raw, est_method)
            t0 = time.time()
            any_nan, min_beta, bary = run_trial(valid, y_valid, bary_method, optimizer)
            elapsed = time.time() - t0
            if any_nan:
                for cls, b in bary.items():
                    if np.isnan(b).any():
                        cls_series = [s for s, y in zip(valid, y_valid) if y == cls]
                        label = (f"arm={arm} bary={bary_method} cap={cap} est={est_method} "
                                f"floor={floor} opt={optimizer} fold={fold} seed={seed} class={cls}")
                        trace_nan_class(cls_series, bary_method, optimizer, trace_log, label)
            rows.append({
                "arm": arm, "bary_method": bary_method, "cap": cap,
                "est_method": est_method, "floor": floor, "optimizer": optimizer,
                "fold": fold, "seed": seed, "any_nan": any_nan,
                "min_beta": min_beta, "n_series": len(valid), "time_s": round(elapsed, 1),
            })
            print(f"[{arm} bary={bary_method} cap={cap} est={est_method} floor={floor:.0e} "
                  f"opt={optimizer} fold={fold} seed={seed}] any_nan={any_nan} "
                  f"min_beta={min_beta:.4g} t={elapsed:.1f}s", flush=True)
    finally:
        costs_module._inverse_softplus = _ORIG_INV_SOFTPLUS


def main():
    rows = []
    trace_log = open(_HERE / "nan_trace_log.txt", "w")

    # Actual per-trial cost turned out much higher than a single pre-design timing
    # test suggested (150-300s, not ~40s) — a full, faithful WaSPS fit at n_steps=200
    # over T=365 (river.yaml's real, unaggregated resolution) for 4 sequential classes
    # is simply expensive, and some classes take close to the full step budget to
    # converge or diverge regardless of NaN-handling. n_steps is NOT reduced (that
    # would undercount slow-drift failures, see module docstring) — trial COUNTS are
    # cut instead, twice now. Final scope: 5 trials for the two arms that answer the
    # actual question, 2 each for the confirmatory ones.
    fold5   = [(f, 42) for f in range(N_SPLITS)]   # 5: 5 folds x 1 seed
    fold2   = [(f, 42) for f in range(2)]           # 2

    # A1 (load-bearing) + A2 (the discriminator).
    _run_arm(rows, trace_log, "A1", "wasps", "flat5", "mle", 1e-5, "sgd", fold5)
    _run_arm(rows, trace_log, "A2", "wasps", "flat5", "log_cumulant", 1e-5, "sgd", fold5)
    # A3: floor check — predicted null (input beta min 0.0026 >> both floors).
    _run_arm(rows, trace_log, "A3", "wasps", "flat5", "mle", 1e-8, "sgd", fold2)
    # A4/A5: optimizer interaction — secondary question.
    _run_arm(rows, trace_log, "A4", "wasps", "flat5", "mle", 1e-5, "adam", fold2)
    _run_arm(rows, trace_log, "A5", "wasps", "flat5", "log_cumulant", 1e-5, "adam", fold2)
    # B: eucl_params control — same input tail, non-WaSPS gradient structure.
    _run_arm(rows, trace_log, "B", "eucl_params", "flat5", "mle", 1e-5, "sgd", fold2)
    # C: river.yaml's REAL stratified total-20 cap (vs this sweep's flat-5/class).
    _run_arm(rows, trace_log, "C", "wasps", "real_total20", "mle", 1e-5, "sgd", fold2)

    trace_log.close()

    csv_path = _HERE / "barycenter_nan_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[saved] {csv_path}")

    # ---- Summary ----
    print("\n=== Failure-rate summary ===")
    from collections import defaultdict
    by_key = defaultdict(list)
    for r in rows:
        key = (r["arm"], r["bary_method"], r["cap"], r["est_method"], r["floor"], r["optimizer"])
        by_key[key].append(r["any_nan"])
    for key, nans in sorted(by_key.items()):
        rate = sum(nans) / len(nans)
        print(f"  {key}: {sum(nans)}/{len(nans)} NaN  ({100*rate:.0f}%)")


if __name__ == "__main__":
    main()
