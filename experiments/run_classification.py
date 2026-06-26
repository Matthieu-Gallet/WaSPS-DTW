"""Classification experiment runner: 4-method × 2-mode (KNN + Barycenter) × multi-seed.

Usage:
    python experiments/run_classification.py configs/classification.yaml
    python experiments/run_classification.py configs/cpazmal.yaml
    python experiments/run_classification.py configs/river.yaml

Outputs to <output.dir>/:
  classification_scores.csv      — mean ± std over seeds (F1, accuracy, time)
  classification_full.json       — per-seed raw results + config
  confusion_matrix_knn.pdf       — confusion matrices (last seed, KNN mode)
  confusion_matrix_barycenter.pdf — confusion matrices (last seed, barycenter mode)

Four methods (defined once):

  wasps       — WaSPS W₂² on estimated params  (use_positivity_constraint, manual grad)
  eucl_params — SqEuclidean on estimated params (autodiff)
  eucl_raw    — SqEuclidean on raw samples      (autodiff)
  sta         — OT Sinkhorn on raw samples      (autodiff through Sinkhorn)

Two modes per method:
  knn         — k-NN (k=1) classify by nearest training series
  barycenter  — classify by nearest per-class Fréchet barycenter

Multi-seed: each seed produces an independent train/test split (or independent
synthetic draw).  Reported metrics = mean ± std across seeds.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

_HERE = Path(__file__).parent
_SRC  = _HERE.parent / "src"
sys.path.insert(0, str(_SRC))

jax.config.update("jax_enable_x64", True)  # must precede jax.numpy usage
import distributions
from data.preprocess import clean_time_series, to_fixed_n
from baselines.sta_wrapper import knn_predict as sta_knn, make_cost_fn as sta_cost_fn
from classification.barycenter_clf import fit_barycenters, predict
from classification.nn import knn_predict as sdtw_knn
from costs import SqEuclidean, WaSPS
from softdtw import SoftDTW


# ---------------------------------------------------------------------------
# Method table — one entry per method, referenced everywhere
# ---------------------------------------------------------------------------

# repr:        'params' → estimate distribution parameters first
#              'raw'    → use raw sample arrays directly
_METHODS = {
    'wasps':       {'repr': 'params'},
    'eucl_params': {'repr': 'params'},
    'eucl_raw':    {'repr': 'raw'},
    'sta':         {'repr': 'raw'},
}


def _make_cost_fn(method: str, family: str, sta_epsilon: float):
    """Cost function for KNN and predict (data in positive-param space, no θ conversion)."""
    if method == 'wasps':
        return WaSPS(family, log_correction=True)
    if method in ('eucl_params', 'eucl_raw'):
        return SqEuclidean()
    if method == 'sta':
        return sta_cost_fn(sta_epsilon)
    raise ValueError(f"unknown method '{method}'")


def _make_softdtw_bary(method: str, family: str, sta_epsilon: float, gamma: float) -> SoftDTW:
    """SoftDTW instance for barycenter fitting."""
    if method == 'wasps':
        cost_fn = WaSPS(family, use_positivity_constraint=True)
        return SoftDTW(cost_fn, gamma, is_divergence=True, manual_grad=True)
    if method in ('eucl_params', 'eucl_raw'):
        return SoftDTW(SqEuclidean(), gamma, is_divergence=True, manual_grad=False)
    if method == 'sta':
        return SoftDTW(sta_cost_fn(sta_epsilon), gamma, is_divergence=True, manual_grad=False)
    raise ValueError(f"unknown method '{method}'")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _make_synthetic(cfg: dict, rng: np.random.Generator):
    family = cfg["dataset"]["family"]
    if family != "exponential":
        raise ValueError("synthetic dataset only supports family=exponential")
    rates   = cfg["dataset"]["rate_params"]
    n_train = cfg["dataset"]["n_train_per_class"]
    n_test  = cfg["dataset"]["n_test_per_class"]
    T, N    = cfg["dataset"]["T"], cfg["dataset"]["N_samples"]
    train_raw, test_raw, train_labels, test_labels = [], [], [], []
    for cls_idx, rate in enumerate(rates):
        for _ in range(n_train):
            train_raw.append(rng.exponential(1.0 / rate, (T, N)))
            train_labels.append(cls_idx)
        for _ in range(n_test):
            test_raw.append(rng.exponential(1.0 / rate, (T, N)))
            test_labels.append(cls_idx)
    return train_raw, test_raw, np.array(train_labels), np.array(test_labels)


def _load_river(cfg: dict, seed: int):
    from data.river_loader import load_river_classification
    ds   = cfg["dataset"]
    data = load_river_classification(
        data_dir=ds["data_dir"],
        mode=ds.get("mode", "balanced"),
        test_size=ds.get("test_size", 0.2),
        max_time_steps=ds.get("max_time_steps"),
        samples_per_step=cfg["classification"].get("samples_per_step"),
        seed=seed,
    )
    return (
        data["X_train"], data["X_test"],
        np.asarray(data["y_train"]), np.asarray(data["y_test"]),
    )


def _load_cpazmal(cfg: dict, seed: int = 42):
    """Load CPAZMaL data.  Caches arrays to disk; returns (X_train, X_test, labels, labels).

    Also caches and returns a `groups` array (geographic group index per sample)
    when available from the HDF5.  The `groups` array is used by the debug report
    to detect whether a class barycenter is built from a single geographic group.
    The return value is extended to (X_train, X_test, labels, labels, groups_or_None)
    when the caller passes `return_groups=True` — but for backward compatibility the
    normal call still returns 4 values.
    """
    from data.cpazmal_loader import MLDatasetLoader, extract_time_series
    ds        = cfg["dataset"]
    hdf5_path = ds["hdf5_path"]
    max_groups= ds.get("max_groups")
    cache_dir = Path(ds.get("cache_dir", "data/cpazmal"))
    suffix    = "all" if max_groups is None else f"mg{max_groups}"
    cx_train  = cache_dir / f"X_train_{suffix}.npy"
    cx_pred   = cache_dir / f"X_predict_{suffix}.npy"
    cy        = cache_dir / f"y_{suffix}.npy"
    cg        = cache_dir / f"groups_{suffix}.npy"  # geographic group identity

    groups = None
    if cx_train.exists():
        print(f"[cpazmal] loading from cache (max_groups={max_groups})")
        X_train = list(np.load(cx_train,  allow_pickle=True))
        X_test  = list(np.load(cx_pred,   allow_pickle=True))
        labels  = np.load(cy)
        if cg.exists():
            groups = np.load(cg)
    else:
        print(f"[cpazmal] extracting from HDF5 (max_groups={max_groups}) …")
        loader = MLDatasetLoader(hdf5_path)
        data   = extract_time_series(loader, max_groups=max_groups)
        X_train = list(data["X_train"])
        X_test  = list(data["X_predict"])
        labels  = np.asarray(data["y"])
        groups  = np.asarray(data["groups"]) if "groups" in data else None
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cx_train, np.array(X_train, dtype=object), allow_pickle=True)
        np.save(cx_pred,  np.array(X_test,  dtype=object), allow_pickle=True)
        np.save(cy, labels)
        if groups is not None:
            np.save(cg, groups)
        print(f"[cpazmal] cache saved to {cache_dir}")
    # Rectangularise raw arrays if samples_per_step is set
    n = cfg["classification"].get("samples_per_step")
    if n is not None:
        rng = np.random.default_rng(seed)
        X_train = [to_fixed_n(s, n, rng) for s in X_train]
        X_test  = [to_fixed_n(s, n, rng) for s in X_test]
    return X_train, X_test, labels, labels, groups


# ---------------------------------------------------------------------------
# Subsample helper
# ---------------------------------------------------------------------------

def _subsample(X, y, max_n, rng, extra=None):
    """Stratified subsample of X, y (and optionally an extra array such as group indices).

    Args:
        extra: optional 1-D numpy array of the same length as y (e.g. geographic
               group indices).  Subsampled alongside X and y; returned as a third
               value when not None.

    Returns:
        (X_sub, y_sub) when extra is None; (X_sub, y_sub, extra_sub) otherwise.
    """
    if max_n < 0 or max_n >= len(y):
        return (X, y, extra) if extra is not None else (X, y)
    idx, _ = train_test_split(
        np.arange(len(y)), train_size=max_n,
        random_state=int(rng.integers(2**31)), stratify=y,
    )
    sorted_idx = np.sort(idx)
    sub_extra  = extra[sorted_idx] if extra is not None else None
    if extra is not None:
        return [X[i] for i in sorted_idx], y[sorted_idx], sub_extra
    return [X[i] for i in sorted_idx], y[sorted_idx]


# ---------------------------------------------------------------------------
# Representation builder
# ---------------------------------------------------------------------------

def _build_repr(raw_series: list, repr_type: str, family: str) -> list:
    """params → estimate distribution parameters; raw → return as-is."""
    if repr_type == 'params':
        return [distributions.get(family).fit_time_series(clean_time_series(s), dtype=np.float64)
                for s in raw_series]
    return raw_series


# ---------------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------------

def _metrics(preds, truth):
    classes = sorted(set(truth.tolist()))
    return {
        "accuracy":         float(accuracy_score(truth, preds)),
        "f1_weighted":      float(f1_score(truth, preds, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, preds, labels=classes).tolist(),
        "classes":          classes,
    }


# ---------------------------------------------------------------------------
# Debug dump helpers (gated by cfg["debug"]=True, first seed only)
# ---------------------------------------------------------------------------

def _debug_dump_dataset(
    train_raw: list,
    train_labels: np.ndarray,
    family: str,
    debug_dir: Path,
    groups: np.ndarray = None,
) -> None:
    """Write shared dataset diagnostics to ``debug_dir``.

    Produces, for each class:
    - samples_pdf_<class>.pdf  — histogram + MLE & log-cumulant PDF overlay at 3 timesteps
    And writes ``debug_dir/dataset_report.log`` with:
    - Per-class: sample count, T, N/timestep, min/max/mean/std/skew, %NaN, nb resampled,
      goodness-of-fit KS and Anderson-Darling (MLE), averaged over timesteps.
    - If ``groups`` array is provided: nb distinct geographic groups per class in the subsampled
      training set.
    """
    import scipy.stats as scipy_stats
    try:
        import distributions as _dists
        from data.preprocess import clean_series as _clean
        from data.preprocess import to_fixed_n as _to_fixed_n
    except ImportError:
        print("[debug] WARNING: cannot import 'distributions'/'data.preprocess' — skipping dataset report")
        return
    try:
        from plot.classification_plots import plot_samples_with_fitted_pdf
    except ImportError:
        print("[debug] WARNING: cannot import 'plot.classification_plots' — skipping PDF plots")
        plot_samples_with_fitted_pdf = None

    debug_dir.mkdir(parents=True, exist_ok=True)
    dist  = _dists.get(family)
    classes = sorted(set(train_labels.tolist()))

    # Build {label: index-into-train_raw} per class (post-subsample view)
    class_indices = {c: [i for i, lbl in enumerate(train_labels) if lbl == c]
                     for c in classes}

    log_lines = []
    log_lines.append(f"=== Dataset debug report — family={family} ===\n")
    log_lines.append(f"Total training samples: {len(train_labels)}  "
                     f"classes: {sorted(classes)}\n\n")

    for cls in classes:
        idx_list = class_indices[cls]
        series   = [train_raw[i] for i in idx_list]
        n_samp   = len(series)
        if n_samp == 0:
            log_lines.append(f"--- class {cls}: NO SAMPLES ---\n")
            continue

        T  = series[0].shape[0]
        N_raw = series[0].shape[1] if series[0].ndim > 1 else 1

        # ── Per-class raw stats (over all timesteps × all samples) ──
        all_vals = np.concatenate([s.ravel() for s in series])
        nan_frac = float(np.mean(~np.isfinite(all_vals)))
        valid    = all_vals[np.isfinite(all_vals) & (all_vals > 0)]

        from scipy.stats import skew as _skew
        v_min  = float(np.min(valid)) if len(valid) else float('nan')
        v_max  = float(np.max(valid)) if len(valid) else float('nan')
        v_mean = float(np.mean(valid)) if len(valid) else float('nan')
        v_std  = float(np.std(valid))  if len(valid) else float('nan')
        v_skew = float(_skew(valid))   if len(valid) >= 3 else float('nan')

        # ── Per-timestep: n_valid, resample needed, KS & AD goodness-of-fit ──
        ks_stats  = []
        ad_stats  = []
        n_resamp_ts = 0   # timesteps where resampling with replacement was used
        for t in range(T):
            pooled = np.concatenate([_clean(s[t]) for s in series])
            if len(pooled) < 5:
                continue
            n_valid_t = len(pooled)
            # Count series-timestep pairs where the raw window had fewer valid
            # pixels than N_raw (before to_fixed_n); proxied by clean < N_raw.
            # Note: after to_fixed_n(n=N_raw), clean_series removes the shifted
            # minimum, so the count will be ~N (one per series) — this is
            # expected and not a data-quality concern.
            n_resamp_ts += sum(
                1 for s in series
                if len(_clean(s[t])) < N_raw
            )
            # MLE fit + goodness-of-fit — append each stat immediately so a
            # failure in one test does not suppress the other.
            try:
                params = dist.estimate(pooled, method='mle')
                if family == 'exponential':
                    beta = float(params)
                    ks_s, _ = scipy_stats.kstest(pooled, 'expon',
                                                 args=(0, 1.0 / beta))
                    ks_stats.append(float(ks_s))
                    ad_r = scipy_stats.anderson(pooled, dist='expon')
                    ad_stats.append(float(ad_r.statistic))
                else:
                    k_v, lam_v = float(params[0]), float(params[1])
                    ks_s, _ = scipy_stats.kstest(
                        pooled, 'weibull_min',
                        args=(k_v, 0, lam_v),
                    )
                    ks_stats.append(float(ks_s))
                    # 'weibull_min' (not 'weibull_max') is the scipy AD valid key
                    ad_r = scipy_stats.anderson(pooled, dist='weibull_min')
                    ad_stats.append(float(ad_r.statistic))
            except Exception:
                pass

        ks_mean  = float(np.mean(ks_stats))  if ks_stats  else float('nan')
        ad_mean  = float(np.mean(ad_stats))  if ad_stats  else float('nan')

        # ── Geographic groups (CPAZMaL only) ──
        group_info = ""
        if groups is not None:
            cls_groups = groups[np.array(idx_list)]
            n_distinct = len(set(cls_groups.tolist()))
            group_info = (f"\n  Geographic groups in subsample: {n_distinct} "
                          f"(group IDs: {sorted(set(cls_groups.tolist()))})")

        log_lines.append(
            f"--- class {cls} ---\n"
            f"  n_samples:      {n_samp}\n"
            f"  shape:          T={T}, N_raw={N_raw}\n"
            f"  values:         min={v_min:.4g}  max={v_max:.4g}  "
            f"mean={v_mean:.4g}  std={v_std:.4g}  skew={v_skew:.4g}\n"
            f"  NaN fraction:   {nan_frac:.2%}\n"
            f"  timesteps with resampling (per-series): {n_resamp_ts}\n"
            f"  goodness-of-fit (MLE, mean over T):\n"
            f"    KS statistic:  {ks_mean:.4f}   (closer to 0 = better fit)\n"
            f"    AD statistic:  {ad_mean:.4f}   (lower = better fit)"
            f"{group_info}\n\n"
        )

        # ── Per-class samples + PDF plot ──
        if plot_samples_with_fitted_pdf is not None:
            plot_samples_with_fitted_pdf(
                series, family, cls, f'class_{cls}',
                output_dir=str(debug_dir),
                n_timesteps=3,
            )

    # Write log
    log_path = debug_dir / "dataset_report.log"
    with open(log_path, "w") as f:
        f.writelines(log_lines)
    print(f"[debug] dataset report → {log_path}", flush=True)


def _debug_dump_barycenters(
    method: str,
    barycenters: dict,
    train_repr: list,
    train_labels: np.ndarray,
    family: str,
    debug_dir: Path,
) -> None:
    """Write per-method barycenter plots to ``debug_dir/<method>/``."""
    try:
        from plot.classification_plots import plot_barycenter_debug
    except ImportError:
        print(f"[debug] WARNING: cannot import 'plot.classification_plots' — skipping bary plots")
        return

    method_dir = debug_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted(barycenters.keys())
    class_indices = {c: [i for i, lbl in enumerate(train_labels) if lbl == c]
                     for c in classes}

    for cls in classes:
        bary = np.asarray(barycenters[cls])
        cls_series = [train_repr[i] for i in class_indices[cls]]
        plot_barycenter_debug(
            bary, cls_series, family,
            class_label=cls, class_name=f'class_{cls}',
            method=method,
            output_dir=str(method_dir),
        )

    print(f"[debug]   barycenters/{method}/ → {len(classes)} class plots", flush=True)


# ---------------------------------------------------------------------------
# Single-seed runner
# ---------------------------------------------------------------------------

def _run_one_seed(cfg: dict, seed: int, methods: list) -> list:
    family    = cfg["dataset"]["family"]
    clf_cfg   = cfg["classification"]
    ds_type   = cfg["dataset"]["type"]
    rng       = np.random.default_rng(seed)

    cpazmal_groups = None   # geographic group identity (CPAZMaL only)
    if ds_type == "synthetic":
        train_raw, test_raw, train_labels, test_labels = _make_synthetic(cfg, rng)
    elif ds_type == "river":
        train_raw, test_raw, train_labels, test_labels = _load_river(cfg, seed)
    elif ds_type == "cpazmal":
        train_raw, test_raw, train_labels, test_labels, cpazmal_groups = \
            _load_cpazmal(cfg, seed=seed)
    else:
        raise ValueError(f"dataset.type '{ds_type}' not supported")

    # Subsample train set; also subsample cpazmal_groups (geographic IDs) if present
    _subsample_result = _subsample(train_raw, train_labels,
                                   clf_cfg.get("max_train_samples", -1), rng,
                                   extra=cpazmal_groups)
    if cpazmal_groups is not None:
        train_raw, train_labels, cpazmal_groups = _subsample_result
    else:
        train_raw, train_labels = _subsample_result

    test_raw, test_labels = _subsample(test_raw, test_labels,
                                       clf_cfg.get("max_test_samples", -1), rng)

    # Debug: dataset-level diagnostics (once, for the first seed only)
    _debug = cfg.get("debug", False)
    _debug_dir = Path(cfg["output"]["dir"]) / "debug" if _debug else None
    _is_first_seed = (seed == cfg.get("seed", 42))
    if _debug and _is_first_seed:
        print(f"[debug] writing dataset diagnostics → {_debug_dir}", flush=True)
        _debug_dump_dataset(train_raw, train_labels, family, _debug_dir,
                            groups=cpazmal_groups)

    gamma       = clf_cfg["gamma"]
    k           = clf_cfg.get("k", 1)
    n_steps     = clf_cfg["n_steps"]
    lr          = clf_cfg["lr"]
    sta_epsilon = clf_cfg.get("sta_epsilon", 0.05)
    modes       = [m.lower() for m in cfg.get("modes", ["knn", "barycenter"])]

    results      = []
    _repr_cache: dict = {}

    def _get_repr(repr_type, split):
        if (repr_type, split) not in _repr_cache:
            raw = train_raw if split == 'train' else test_raw
            _repr_cache[(repr_type, split)] = _build_repr(raw, repr_type, family)
        return _repr_cache[(repr_type, split)]

    for method in methods:
        cfg_m     = _METHODS[method]
        repr_type = cfg_m['repr']
        cost_fn   = _make_cost_fn(method, family, sta_epsilon)
        train_repr = _get_repr(repr_type, 'train')
        test_repr  = _get_repr(repr_type, 'test')

        if 'knn' in modes:
            t0 = time.time()
            if method == 'sta':
                preds = sta_knn(train_raw, train_labels, test_raw,
                                gamma=gamma, epsilon=sta_epsilon, k=k)
            else:
                preds = sdtw_knn(train_repr, train_labels, test_repr,
                                 cost_fn=cost_fn, gamma=gamma, k=k)
            elapsed = time.time() - t0
            res = _metrics(preds, test_labels)
            res.update({"method": method, "mode": "knn",
                        "seed": seed, "time_s": elapsed})
            results.append(res)
            print(f"  [{method}/knn] acc={res['accuracy']:.3f}  "
                  f"f1={res['f1_weighted']:.3f}  t={elapsed:.1f}s", flush=True)

        if 'barycenter' in modes:
            t0 = time.time()
            softdtw_bary = _make_softdtw_bary(method, family, sta_epsilon, gamma)
            barycenters = fit_barycenters(
                train_repr, train_labels, softdtw_bary,
                n_steps=n_steps, lr=lr,
                patience=clf_cfg.get("early_stop_patience", 15),
                min_rel_improve=clf_cfg.get("early_stop_tol", 1e-4),
            )
            train_time = time.time() - t0

            # Debug: barycenter plots for the first seed
            if _debug and _is_first_seed:
                _debug_dump_barycenters(method, barycenters, train_repr,
                                        train_labels, family, _debug_dir)

            t1 = time.time()
            preds = predict(test_repr, barycenters, cost_fn, gamma)
            infer_time = time.time() - t1
            res = _metrics(preds, test_labels)
            res.update({"method": method, "mode": "barycenter",
                        "seed": seed, "time_s": train_time + infer_time,
                        "train_time_s": train_time, "infer_time_s": infer_time})
            results.append(res)
            print(f"  [{method}/bary] acc={res['accuracy']:.3f}  "
                  f"f1={res['f1_weighted']:.3f}  t_train={train_time:.1f}s", flush=True)

    return results


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _save_confusion_matrices(all_results: list, out_dir: Path) -> None:
    """Save one confusion-matrix PDF per mode, using the last seed's results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import defaultdict

    by_mode: dict = defaultdict(list)
    for r in all_results:
        by_mode[r['mode']].append(r)

    for mode, rows in by_mode.items():
        last_seed = max(r['seed'] for r in rows)
        last_rows = sorted(
            [r for r in rows if r['seed'] == last_seed],
            key=lambda r: r['method'],
        )
        if not last_rows:
            continue
        classes = last_rows[0]['classes']
        n_cls   = len(classes)
        n_m     = len(last_rows)

        fig, axes = plt.subplots(1, n_m, figsize=(3.0 * n_m, 3.5), squeeze=False)
        axes_flat = axes[0]

        for ax, row in zip(axes_flat, last_rows):
            cm      = np.array(row['confusion_matrix'])
            cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
            ax.set_xticks(range(n_cls))
            ax.set_yticks(range(n_cls))
            ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=7)
            ax.set_yticklabels(classes, fontsize=7)
            ax.set_title(f"{row['method']}\nf1={row['f1_weighted']:.3f}", fontsize=8)
            ax.set_xlabel('Predicted', fontsize=7)
            for i in range(n_cls):
                for j in range(n_cls):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            fontsize=7,
                            color='white' if cm_norm[i, j] > 0.5 else 'black')
        axes_flat[0].set_ylabel('True', fontsize=7)

        plt.tight_layout()
        fname = out_dir / f"confusion_matrix_{mode}.pdf"
        fig.savefig(fname, bbox_inches='tight')
        plt.close(fig)
        print(f"  figure: {fname}")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(all_results: list) -> list:
    from collections import defaultdict
    buckets: dict = defaultdict(list)
    for r in all_results:
        buckets[(r["method"], r["mode"])].append(r)
    summary = []
    for (method, mode), rows in sorted(buckets.items()):
        acc  = [r["accuracy"]    for r in rows]
        f1   = [r["f1_weighted"] for r in rows]
        time = [r["time_s"]      for r in rows]
        summary.append({
            "method": method, "mode": mode, "n_seeds": len(rows),
            "acc_mean":  float(np.mean(acc)),   "acc_std":  float(np.std(acc)),
            "f1_mean":   float(np.mean(f1)),    "f1_std":   float(np.std(f1)),
            "time_mean": float(np.mean(time)),  "time_std": float(np.std(time)),
        })
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base_seed = cfg.get("seed", 42)
    n_seeds   = cfg.get("n_seeds", 1)
    seeds     = [base_seed + i for i in range(n_seeds)]
    family    = cfg["dataset"]["family"]

    methods_req = [m.lower() for m in cfg.get("methods", list(_METHODS.keys()))]
    methods = [m for m in methods_req if m in _METHODS]
    unknown = [m for m in methods_req if m not in _METHODS]
    if unknown:
        print(f"[warn] unknown methods: {unknown} — skipping")

    print(f"[run] dataset={cfg['dataset']['type']}  family={family}  "
          f"methods={methods}  seeds={seeds}", flush=True)

    all_results = []
    for seed in seeds:
        print(f"\n[seed={seed}]", flush=True)
        all_results.extend(_run_one_seed(cfg, seed, methods))

    summary = _aggregate(all_results)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "classification_scores.csv"
    with open(csv_path, "w", newline="") as f:
        fields = ["method", "mode", "n_seeds",
                  "acc_mean", "acc_std", "f1_mean", "f1_std",
                  "time_mean", "time_std"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)

    json_path = out_dir / "classification_full.json"
    with open(json_path, "w") as f:
        json.dump({"config": cfg, "summary": summary, "per_seed": all_results}, f, indent=2)

    _save_confusion_matrices(all_results, out_dir)

    print(f"\n[done] results saved to {out_dir}")
    for row in summary:
        print(f"  {row['method']:12s}/{row['mode']:10s}  "
              f"f1={row['f1_mean']:.3f}±{row['f1_std']:.3f}  "
              f"acc={row['acc_mean']:.3f}±{row['acc_std']:.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
