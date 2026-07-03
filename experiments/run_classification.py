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

_HERE = Path(__file__).parent
_SRC  = _HERE.parent / "src"
sys.path.insert(0, str(_SRC))

jax.config.update("jax_enable_x64", True)  # must precede jax.numpy usage
import distributions
from data.preprocess import clean_time_series, to_fixed_n
from baselines.sta_wrapper import knn_predict as sta_knn
from classification.barycenter_clf import fit_barycenters, predict
from classification.nn import knn_predict as sdtw_knn

from data_utils import build_repr, load_dataset, metrics as _metrics, subsample as _subsample
from method_defs import _METHODS, make_cost_fn as _make_cost_fn, make_softdtw_bary as _make_softdtw_bary


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
        # Also persist the barycenter array so it can be reloaded without recomputing
        npy_path = method_dir / f"{method}_class{cls}.npy"
        np.save(npy_path, bary)

    print(f"[debug]   barycenters/{method}/ → {len(classes)} class plots + .npy", flush=True)


# ---------------------------------------------------------------------------
# Single-seed/fold runner
# ---------------------------------------------------------------------------

def _run_one_seed(cfg: dict, seed: int, methods: list, fold: int | None = None) -> list:
    family  = cfg["dataset"]["family"]
    clf_cfg = cfg["classification"]
    rng     = np.random.default_rng(seed)

    data = load_dataset(cfg, seed, fold=fold)
    train_raw    = data["X_train"]
    test_raw     = data["X_test"]
    train_labels = data["y_train"]
    test_labels  = data["y_test"]
    groups_train = data.get("groups_train")

    # Subsample
    _sub = _subsample(train_raw, train_labels,
                      clf_cfg.get("max_train_samples", -1), rng, extra=groups_train)
    if groups_train is not None:
        train_raw, train_labels, groups_train = _sub
    else:
        train_raw, train_labels = _sub

    test_raw, test_labels = _subsample(test_raw, test_labels,
                                       clf_cfg.get("max_test_samples", -1), rng)

    # Raw downsampling: auto-compute N_min when samples_per_step is not set in config
    # but the method list includes raw-type methods (eucl_raw, sta).
    # Computed independently for train and test — no information leakage.
    raw_methods = [m for m in methods if _METHODS[m]['repr'] == 'raw']
    if raw_methods and clf_cfg.get("samples_per_step") is None:
        from data_utils import compute_raw_n_min
        n_min_train = compute_raw_n_min(train_raw)
        n_min_test  = compute_raw_n_min(test_raw)
        if n_min_train > 0 and n_min_test > 0:
            rng_raw = np.random.default_rng(seed)
            train_raw = [to_fixed_n(s, n_min_train, rng_raw) for s in train_raw]
            test_raw  = [to_fixed_n(s, n_min_test,  rng_raw) for s in test_raw]
            print(f"[raw] auto N_min: train={n_min_train}  test={n_min_test}", flush=True)

    # Debug: dataset-level diagnostics (once, for the first seed only)
    _debug = cfg.get("debug", False)
    _debug_dir = Path(cfg["output"]["dir"]) / "debug" if _debug else None
    _is_first = (seed == cfg.get("seed", 42)) and (fold is None or fold == 0)
    if _debug and _is_first:
        print(f"[debug] writing dataset diagnostics → {_debug_dir}", flush=True)
        _debug_dump_dataset(train_raw, train_labels, family, _debug_dir,
                            groups=groups_train)

    gamma       = clf_cfg["gamma"]
    k_values    = clf_cfg.get("k_values") or [clf_cfg.get("k", 1)]
    n_steps     = clf_cfg["n_steps"]
    lr          = clf_cfg["lr"]
    sta_epsilon = clf_cfg.get("sta_epsilon", 0.05)
    modes       = [m.lower() for m in cfg.get("modes", ["knn", "barycenter"])]

    results     = []
    # Cache: (repr_type, split) → (repr_list, filtered_labels)
    _repr_cache: dict = {}

    def _get_repr(repr_type, split):
        if (repr_type, split) not in _repr_cache:
            raw = train_raw if split == 'train' else test_raw
            lbl = train_labels if split == 'train' else test_labels
            _repr_cache[(repr_type, split)] = build_repr(raw, lbl, repr_type, family)
        return _repr_cache[(repr_type, split)]

    run_id = fold if fold is not None else seed

    for method in methods:
        repr_type = _METHODS[method]['repr']
        cost_fn   = _make_cost_fn(method, family, sta_epsilon)
        train_repr, train_repr_labels = _get_repr(repr_type, 'train')
        test_repr,  test_repr_labels  = _get_repr(repr_type, 'test')

        if 'knn' in modes:
            for k in k_values:
                t0 = time.time()
                if method == 'sta':
                    preds = sta_knn(train_raw, train_labels, test_raw,
                                    gamma=gamma, epsilon=sta_epsilon, k=k)
                    truth = test_labels
                else:
                    preds = sdtw_knn(train_repr, train_repr_labels, test_repr,
                                     cost_fn=cost_fn, gamma=gamma, k=k)
                    truth = test_repr_labels
                elapsed = time.time() - t0
                res = _metrics(preds, truth)
                res.update({"method": method, "mode": "knn", "k": k,
                            "seed": seed, "fold": fold, "time_s": elapsed})
                results.append(res)
                print(f"  [{method}/knn k={k}] acc={res['accuracy']:.3f}  "
                      f"f1={res['f1_weighted']:.3f}  t={elapsed:.1f}s", flush=True)

        if 'barycenter' in modes:
            t0 = time.time()
            softdtw_bary = _make_softdtw_bary(method, family, sta_epsilon, gamma)
            barycenters = fit_barycenters(
                train_repr, train_repr_labels, softdtw_bary,
                n_steps=n_steps, lr=lr,
                patience=clf_cfg.get("early_stop_patience", 15),
                min_rel_improve=clf_cfg.get("early_stop_tol", 1e-4),
            )
            train_time = time.time() - t0

            if _debug and _is_first:
                _debug_dump_barycenters(method, barycenters, train_repr,
                                        train_repr_labels, family, _debug_dir)

            t1 = time.time()
            preds = predict(test_repr, barycenters, cost_fn, gamma)
            infer_time = time.time() - t1
            res = _metrics(preds, test_repr_labels)
            res.update({"method": method, "mode": "barycenter", "k": None,
                        "seed": seed, "fold": fold,
                        "time_s": train_time + infer_time,
                        "train_time_s": train_time, "infer_time_s": infer_time})
            results.append(res)
            print(f"  [{method}/bary] acc={res['accuracy']:.3f}  "
                  f"f1={res['f1_weighted']:.3f}  t_train={train_time:.1f}s", flush=True)

    return results


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _save_confusion_matrices(all_results: list, out_dir: Path) -> None:
    """Save one confusion-matrix PDF per (mode, k) group, using the last seed's results.

    Filename convention:
    - Single K for a mode → ``confusion_matrix_<mode>.pdf``   (backward compat)
    - Multiple K values    → ``confusion_matrix_<mode>_k<k>.pdf``
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import defaultdict

    by_mode_k: dict = defaultdict(list)
    for r in all_results:
        by_mode_k[(r['mode'], r.get('k'))].append(r)

    # Determine whether any mode has multiple K values (to decide filename format)
    mode_ks: dict = defaultdict(set)
    for (mode, k) in by_mode_k:
        mode_ks[mode].add(k)

    for (mode, k), rows in by_mode_k.items():
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
            k_label = f"k={k} " if k is not None else ""
            ax.set_title(f"{row['method']} ({k_label}f1={row['f1_weighted']:.3f})", fontsize=8)
            ax.set_xlabel('Predicted', fontsize=7)
            for i in range(n_cls):
                for j in range(n_cls):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            fontsize=7,
                            color='white' if cm_norm[i, j] > 0.5 else 'black')
        axes_flat[0].set_ylabel('True', fontsize=7)

        plt.tight_layout()
        if len(mode_ks[mode]) > 1 and k is not None:
            fname = out_dir / f"confusion_matrix_{mode}_k{k}.pdf"
        else:
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
        # Use k=None for barycenter rows (K-independent); use actual k for KNN rows.
        buckets[(r["method"], r["mode"], r.get("k"))].append(r)
    summary = []
    for (method, mode, k), rows in sorted(buckets.items(),
                                          key=lambda x: (x[0][0], x[0][1], x[0][2] or -1)):
        acc  = [r["accuracy"]    for r in rows]
        f1   = [r["f1_weighted"] for r in rows]
        time = [r["time_s"]      for r in rows]
        entry = {
            "method": method, "mode": mode, "k": k, "n_seeds": len(rows),
            "acc_mean":  float(np.mean(acc)),   "acc_std":  float(np.std(acc)),
            "f1_mean":   float(np.mean(f1)),    "f1_std":   float(np.std(f1)),
            "time_mean": float(np.mean(time)),  "time_std": float(np.std(time)),
        }
        summary.append(entry)
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

    cv       = cfg.get("cross_validation", {})
    n_splits = int(cv.get("n_splits", 1))
    fold_cfg = int(cv.get("fold", 0))

    # K-fold mode: fold=-1 → iterate over all folds (one seed); else use fixed fold
    if n_splits > 1 and fold_cfg < 0:
        iterations = [(base_seed, f) for f in range(n_splits)]
        print(f"[run] K-fold mode: n_splits={n_splits}  group_aware={cv.get('group_aware', False)}")
    else:
        fold_arg = fold_cfg if n_splits > 1 else None
        iterations = [(seed, fold_arg) for seed in seeds]

    print(f"[run] dataset={cfg['dataset']['type']}  family={family}  "
          f"methods={methods}  n_iter={len(iterations)}", flush=True)

    all_results = []
    for seed, fold in iterations:
        tag = f"fold={fold}" if fold is not None else f"seed={seed}"
        print(f"\n[{tag}]", flush=True)
        all_results.extend(_run_one_seed(cfg, seed, methods, fold=fold))

    summary = _aggregate(all_results)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "classification_scores.csv"
    with open(csv_path, "w", newline="") as f:
        fields = ["method", "mode", "k", "n_seeds",
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
        k_tag = f" k={row['k']}" if row.get("k") is not None else ""
        print(f"  {row['method']:12s}/{row['mode']:10s}{k_tag:6s}  "
              f"f1={row['f1_mean']:.3f}±{row['f1_std']:.3f}  "
              f"acc={row['acc_mean']:.3f}±{row['acc_std']:.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
