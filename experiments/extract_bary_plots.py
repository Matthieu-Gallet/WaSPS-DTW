"""Extract river-discharge barycenter-debug plots for classes PN and NG only.

Scans a `bary_debug/` directory (produced by any of run_optim_hyper.py --verbose,
run_sensitivity.py --verbose, or run_barycenters.py --verbose — all write through
experiment_common._save_bary_debug) for `.npz` files, keeps only river
(family='exponential') barycenters for classes NG (idx 0) and PN (idx 5), and calls
src/plot/classification_plots.plot_barycenter_debug for each (file, class) pair found.

wasps/eucl_params curves are automatically plotted in λ (discharge, m³/s) space instead
of raw β — see plot_barycenter_debug's `invert_to_lambda` logic (2026-07-08) — so they
show the same trend as eucl_raw's raw amplitude.

`.npz` files written before that same date lack the `family`/`method` keys directly —
this script falls back to parsing the method name out of the filename in that case
(family is assumed 'exponential' throughout, since this script is river-only).

Usage:
    python experiments/extract_bary_plots.py --bary-debug-dir results/jax_river_bary/bary_debug \\
        --output-dir results/jax_river_bary/pn_ng_plots
    python experiments/extract_bary_plots.py --bary-debug-dir results/jax_sensitivity/bary_debug \\
        --output-dir results/jax_sensitivity/pn_ng_plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_SRC  = _HERE.parent / "src"
sys.path.insert(0, str(_SRC))

from plot.classification_plots import plot_barycenter_debug

# NG=0, PN=5 — confirmed via data/river/metadata_balanced.npy's idx_to_regime
# (regime_to_idx: {'NG': 0, 'NP': 1, 'NV': 2, 'PC': 3, 'PM': 4, 'PN': 5}).
_TARGET_CLASSES = {0: 'NG', 5: 'PN'}
_TARGET_METHODS = ['wasps', 'eucl_params', 'eucl_raw']
_METHOD_NAMES_BY_LEN = sorted(
    ['wasps_nodiv', 'wasps', 'eucl_params_nodiv', 'eucl_params', 'eucl_raw_nodiv', 'eucl_raw', 'sta'],
    key=lambda m: -len(m.split('_')),
)


def _infer_method_from_stem(stem: str) -> str | None:
    """Fallback for .npz files saved before family/method were stored directly."""
    parts = stem.split('_')
    for m in _METHOD_NAMES_BY_LEN:
        m_parts = m.split('_')
        n = len(m_parts)
        for i in range(len(parts) - n + 1):
            if parts[i:i + n] == m_parts:
                return m
    return None


def extract(bary_debug_dir: Path, output_dir: Path) -> int:
    """Each source .npz gets its own output subdirectory (named after the .npz stem,
    which already encodes the scenario/method/gamma/sweep-value that produced it) —
    plot_barycenter_debug's output filename is only `bary_<class>_<method>.pdf` (no
    gamma/scenario in the name), so multiple source files for the same (class,method)
    — e.g. one per gamma value in a gamma sweep — would silently overwrite each other
    if written to a single shared directory."""
    n_plots = 0
    for npz_path in sorted(bary_debug_dir.glob('*.npz')):
        with np.load(npz_path, allow_pickle=True) as d:
            family = str(d['family']) if 'family' in d else 'exponential'
            method = str(d['method']) if 'method' in d else _infer_method_from_stem(npz_path.stem)
            if family != 'exponential' or method not in _TARGET_METHODS:
                continue

            train_labels = d['train_labels']
            series_keys = sorted(
                (k for k in d.files if k.startswith('series_')),
                key=lambda k: int(k.split('_')[1]),
            )
            all_series = [d[k] for k in series_keys]

            found_any = False
            for cls, cls_name in _TARGET_CLASSES.items():
                bary_key = f'class_{cls}'
                if bary_key not in d:
                    continue
                idx = [i for i, lbl in enumerate(train_labels) if int(lbl) == cls]
                if not idx:
                    continue
                class_series = [all_series[i] for i in idx]
                plot_barycenter_debug(
                    d[bary_key], class_series, family,
                    class_label=cls, class_name=cls_name, method=method,
                    output_dir=str(output_dir / npz_path.stem),
                )
                n_plots += 1
                found_any = True
            if found_any:
                print(f"  {npz_path.name} -> {output_dir / npz_path.stem}/")
    return n_plots


def main():
    parser = argparse.ArgumentParser(description="Extract river PN/NG barycenter-debug plots")
    parser.add_argument("--bary-debug-dir", required=True,
                        help="e.g. results/jax_river_bary/bary_debug")
    parser.add_argument("--output-dir", required=True,
                        help="e.g. results/jax_river_bary/pn_ng_plots")
    args = parser.parse_args()

    bary_debug_dir = Path(args.bary_debug_dir)
    output_dir = Path(args.output_dir)
    if not bary_debug_dir.is_dir():
        print(f"[error] {bary_debug_dir} does not exist — run with --verbose first "
              f"(run_optim_hyper.py / run_sensitivity.py / run_barycenters.py)")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_plots = extract(bary_debug_dir, output_dir)
    print(f"[done] {n_plots} plots written to {output_dir}")


if __name__ == "__main__":
    main()
