r"""Extract mean F1 ± std results into LaTeX tables (NiceTabular environment).

Two table families, matching the two places "mean F1 ± std, per method, per dataset,
per swept value" data lives in this repo:

1. `classif` — the gamma sweep in classification_scores.csv / barycenter_scores.csv
   (experiments/run_classification.py / run_barycenters.py): rows = method, columns =
   gamma value.
2. `sensitivity` — a run_sensitivity.py sweep scenario (sweep_n_samples/sweep_n_train/
   sweep_decimation): one summary CSV per method (sensitivity_<scenario>_<method>_<mode>.csv),
   combined here into a single table: rows = method, columns = swept value.
3. `full_baseline` — a run_full_baseline.py gamma sweep (full_baseline_<name>_detail.csv):
   two tables (F1 and total computation time, both mean +- std), rows = gamma, columns = method.

`nicematrix` is already a loaded LaTeX package in RESSOURCES/main.tex — the emitted
`\begin{NiceTabular}...\end{NiceTabular}` snippets compile as-is when \input/copied in.

Usage:
    python experiments/extract_latex_tables.py classif \\
        --csv results/jax_river/classification_scores.csv --mode knn \\
        --out results/tables/river_knn_gamma.tex --caption "River KNN, gamma sweep"

    python experiments/extract_latex_tables.py classif \\
        --csv results/jax_river_bary/barycenter_scores.csv \\
        --out results/tables/river_bary_gamma.tex --caption "River barycenter, gamma sweep"

    python experiments/extract_latex_tables.py sensitivity \\
        --dir results/jax_sensitivity --scenario n_train --mode knn \\
        --out results/tables/river_n_train_knn.tex --caption "River sweep\\_n\\_train, KNN"

    python experiments/extract_latex_tables.py full_baseline \\
        --detail-csv results/jax_full_baseline/full_baseline_river_detail.csv \\
        --dataset-label "River" --label-prefix tab:baseline_river \\
        --out results/tables/full_baseline_river.tex
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path


def _fmt(mean, std) -> str:
    try:
        mean = float(mean)
        std = float(std)
    except (TypeError, ValueError):
        return "--"
    if math.isnan(mean):
        return "--"
    std_str = f"{std:.3f}" if not math.isnan(std) else "0.000"
    return f"{mean:.3f} $\\pm$ {std_str}"


def _latex_escape(s: str) -> str:
    return re.sub(r'([_&%$#{}])', r'\\\1', str(s))


def _fmt_col(v) -> str:
    try:
        f = float(v)
        if f == int(f) and abs(f) < 1e6:
            return str(int(f))
        return f"{f:.4g}"
    except (TypeError, ValueError):
        return str(v)


def render_nice_tabular(row_labels: list, col_labels: list, cells: dict,
                        row_header: str = "Method", caption: str = None,
                        label: str = None) -> str:
    """cells: {(row, col): (f1_mean, f1_std)}."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{NiceTabular}{l" + "c" * len(col_labels) + "}",
        r"\toprule",
        f"{row_header} & " + " & ".join(_fmt_col(c) for c in col_labels) + r" \\",
        r"\midrule",
    ]
    for r in row_labels:
        row_cells = [_fmt(*cells.get((r, c), (float('nan'), float('nan')))) for c in col_labels]
        lines.append(f"{_latex_escape(r)} & " + " & ".join(row_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{NiceTabular}")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def table_from_classif_csv(csv_path: Path, mode: str = None, col: str = "gamma",
                           caption: str = None, label: str = None) -> str:
    """classification_scores.csv / barycenter_scores.csv → rows=method, columns=gamma."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if mode is not None and "mode" in (rows[0].keys() if rows else []):
        rows = [r for r in rows if r["mode"] == mode]

    methods = sorted({r["method"] for r in rows})
    col_values = sorted({r[col] for r in rows if r.get(col) not in (None, "")}, key=float)
    cells = {}
    for r in rows:
        if r.get(col) in (None, ""):
            continue
        cells[(r["method"], r[col])] = (r["f1_mean"], r["f1_std"])
    return render_nice_tabular(methods, col_values, cells,
                               row_header="Method", caption=caption, label=label)


def table_from_sensitivity_dir(sens_dir: Path, scenario: str, mode: str,
                               caption: str = None, label: str = None) -> str:
    """Combine sensitivity_<scenario>_<method>_<mode>.csv (one per method) into a
    single rows=method, columns=swept-value table."""
    pattern = f"sensitivity_{scenario}_*_{mode}.csv"
    files = sorted(sens_dir.glob(pattern))
    files = [f for f in files if not f.name.endswith("_detail.csv")]
    if not files:
        raise FileNotFoundError(f"no files matching {pattern} in {sens_dir}")

    methods, cells, value_name, col_values_set = [], {}, None, set()
    for fpath in files:
        # sensitivity_<scenario>_<method>_<mode>.csv → method is everything between
        # the scenario prefix and the trailing _<mode> suffix.
        method = fpath.stem[len(f"sensitivity_{scenario}_"):-len(f"_{mode}")]
        methods.append(method)
        with open(fpath, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            value_name = fieldnames[0]  # always the sweep-value column, by construction
            for r in reader:
                col_values_set.add(r[value_name])
                cells[(method, r[value_name])] = (r["f1_mean"], r["f1_std"])

    col_values = sorted(col_values_set, key=float)
    return render_nice_tabular(sorted(methods), col_values, cells,
                               row_header="Method", caption=caption, label=label)


_FULL_BASELINE_METHOD_ORDER = ['wasps', 'wasps_nodiv', 'eucl_params', 'eucl_params_nodiv',
                              'eucl_raw', 'eucl_raw_nodiv', 'sta']


def render_gamma_by_method_table(gamma_values: list, methods: list, cells: dict,
                                 caption: str = None, label: str = None) -> str:
    """cells: {(gamma_str, method): (mean, std)} — rows=gamma, columns=method."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{NiceTabular}{l" + "c" * len(methods) + "}",
        r"\toprule",
        r"$\gamma$ & " + " & ".join(_latex_escape(m) for m in methods) + r" \\",
        r"\midrule",
    ]
    for g in gamma_values:
        row_cells = [_fmt(*cells.get((g, m), (float('nan'), float('nan')))) for m in methods]
        lines.append(f"{_fmt_col(g)} & " + " & ".join(row_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{NiceTabular}")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _group_mean_std(rows: list, key_fn, value_key: str) -> dict:
    groups = {}
    for r in rows:
        try:
            v = float(r[value_key])
        except (TypeError, ValueError):
            continue
        groups.setdefault(key_fn(r), []).append(v)
    return {k: (statistics.mean(v), statistics.pstdev(v) if len(v) > 1 else 0.0)
           for k, v in groups.items()}


def tables_from_full_baseline_detail(detail_csv: Path, dataset_label: str = None,
                                     label_prefix: str = None) -> str:
    """run_full_baseline.py's full_baseline_<name>_detail.csv (one row per
    method x gamma x seed) -> two NiceTabular tables (rows=gamma, columns=method):
    F1 score mean+-std, and total computation time (s) mean+-std."""
    with open(detail_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if not r.get("gamma"):
            r["gamma"] = "1.0"  # pre-gamma-sweep detail CSVs have no gamma column

    gammas = sorted({r["gamma"] for r in rows}, key=float)
    present_methods = {r["method"] for r in rows}
    methods = [m for m in _FULL_BASELINE_METHOD_ORDER if m in present_methods]
    methods += sorted(present_methods - set(methods))

    f1_cells = _group_mean_std(rows, lambda r: (r["gamma"], r["method"]), "f1")
    time_cells = _group_mean_std(rows, lambda r: (r["gamma"], r["method"]), "total_time")

    prefix = f"{dataset_label} --- " if dataset_label else ""
    f1_table = render_gamma_by_method_table(
        gammas, methods, f1_cells,
        caption=f"{prefix}F1 score (mean $\\pm$ std)",
        label=f"{label_prefix}_f1" if label_prefix else None)
    time_table = render_gamma_by_method_table(
        gammas, methods, time_cells,
        caption=f"{prefix}Computation time in seconds (mean $\\pm$ std)",
        label=f"{label_prefix}_time" if label_prefix else None)
    return f1_table + "\n\n" + time_table + "\n"


def main():
    parser = argparse.ArgumentParser(description="Extract F1 mean±std results into LaTeX (NiceTabular) tables")
    sub = parser.add_subparsers(dest="kind", required=True)

    p1 = sub.add_parser("classif", help="gamma sweep from classification/barycenter CSV")
    p1.add_argument("--csv", required=True)
    p1.add_argument("--mode", default=None, help="filter to 'knn' or 'barycenter' (classification_scores.csv only)")
    p1.add_argument("--col", default="gamma")
    p1.add_argument("--out", required=True)
    p1.add_argument("--caption", default=None)
    p1.add_argument("--label", default=None)

    p2 = sub.add_parser("sensitivity", help="sweep table combining per-method sensitivity CSVs")
    p2.add_argument("--dir", required=True)
    p2.add_argument("--scenario", required=True, choices=["n_samples", "n_train", "decimation"])
    p2.add_argument("--mode", required=True, choices=["knn", "barycenter"])
    p2.add_argument("--out", required=True)
    p2.add_argument("--caption", default=None)
    p2.add_argument("--label", default=None)

    p3 = sub.add_parser("full_baseline",
                        help="gamma x method tables (F1 + time) from run_full_baseline.py detail CSV")
    p3.add_argument("--detail-csv", required=True)
    p3.add_argument("--out", required=True)
    p3.add_argument("--dataset-label", default=None)
    p3.add_argument("--label-prefix", default=None)

    args = parser.parse_args()

    if args.kind == "classif":
        tex = table_from_classif_csv(Path(args.csv), mode=args.mode, col=args.col,
                                     caption=args.caption, label=args.label)
    elif args.kind == "full_baseline":
        tex = tables_from_full_baseline_detail(Path(args.detail_csv),
                                               dataset_label=args.dataset_label,
                                               label_prefix=args.label_prefix)
    else:
        tex = table_from_sensitivity_dir(Path(args.dir), args.scenario, args.mode,
                                         caption=args.caption, label=args.label)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex + "\n")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
