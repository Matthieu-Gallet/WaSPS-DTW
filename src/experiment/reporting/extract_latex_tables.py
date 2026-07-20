r"""Extract mean F1 ± std results into LaTeX tables (NiceTabular environment).

Table families:

1. `sensitivity` — a run_decimation.py sweep (decimation only, now — n_samples/n_train
   sweeps were dropped along with the legacy sensitivity pipeline): one summary CSV per
   method (sensitivity_<scenario>_<method>_<mode>.csv), combined here into a single
   table: rows = method, columns = swept value (decimation fraction). F1 as %, 1
   decimal, best value per row in bold.
2. `exp1` — the baseline experiment's fixed-gamma detail CSV (one dataset, one mode):
   rows = Metric (F1/Time/RAM), columns = method. F1 as %, 1 decimal; time 1 decimal;
   RAM integer; best value per row in bold.
3. `exp1_combined` — wraps two `exp1` tables (river + cpazmal) into one `table*` with
   two `subtable` environments, matching the paper's two-dataset comparison layout.

`nicematrix` is already a loaded LaTeX package in RESSOURCES/main.tex — the emitted
`\begin{NiceTabular}...\end{NiceTabular}` snippets compile as-is when \input/copied in.

Usage:
    python src/experiment/reporting/extract_latex_tables.py exp1 \\
        --detail-csv results/jax_exp1_baseline/full_baseline_river_detail.csv \\
        --mode knn --dataset-label River --label-prefix tab:exp1_river \\
        --out results/tables/exp1_river_knn.tex

    python src/experiment/reporting/extract_latex_tables.py exp1_combined \\
        --river-detail-csv results/jax_exp1_baseline/full_baseline_river_detail.csv \\
        --cpazmal-detail-csv results/jax_exp1_baseline/full_baseline_cpazmal_detail.csv \\
        --mode knn --out results/tables/exp1_knn.tex

    python src/experiment/reporting/extract_latex_tables.py sensitivity \\
        --dir results/jax_exp3_decimation --scenario decimation --mode barycenter \\
        --out results/tables/exp3_decimation.tex
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_pct(mean, std) -> str:
    """F1 as a percentage, 1 decimal — e.g. "59.6 $\\pm$ 4.1"."""
    try:
        mean = float(mean)
        std = float(std)
    except (TypeError, ValueError):
        return "--"
    if math.isnan(mean):
        return "--"
    std_str = f"{std*100:.1f}" if not math.isnan(std) else "0.0"
    return f"{mean*100:.1f} $\\pm$ {std_str}"


def _fmt_time(mean, std) -> str:
    """Time in seconds, 1 decimal — e.g. "1.7 $\\pm$ 0.1"."""
    try:
        mean = float(mean)
        std = float(std)
    except (TypeError, ValueError):
        return "--"
    if math.isnan(mean):
        return "--"
    std_str = f"{std:.1f}" if not math.isnan(std) else "0.0"
    return f"{mean:.1f} $\\pm$ {std_str}"


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


# Column order + display labels shared by every method x ... table. The
# $\mathcal{D}$- prefix marks the divergence-mode variant; the plain name is the
# _nodiv variant. STA carries a fixed citation key (no div/nodiv distinction).
_FULL_BASELINE_METHOD_ORDER = ['wasps', 'wasps_nodiv', 'eucl_params', 'eucl_params_nodiv',
                              'eucl_raw', 'eucl_raw_nodiv', 'sta']

_DISPLAY_NAME = {
    'wasps':              r'$\mathcal{D}$-WASPS (ours)',
    'wasps_nodiv':        'WASPS (ours)',
    'eucl_params':        r'$\mathcal{D}$-SoftDTW (params)',
    'eucl_params_nodiv':  'SoftDTW (params)',
    'eucl_raw':           r'$\mathcal{D}$-SoftDTW (raw)',
    'eucl_raw_nodiv':     'SoftDTW (raw)',
    'sta':                r'STA~\cite{janati2020spatio}',
}


def _display(m: str) -> str:
    """Method display label — verbatim LaTeX for known methods (already escaped/
    macro'd), else the raw key escaped as a fallback."""
    return _DISPLAY_NAME.get(m, _latex_escape(m))


def _order_methods(methods) -> list:
    ordered = [m for m in _FULL_BASELINE_METHOD_ORDER if m in methods]
    ordered += sorted(set(methods) - set(ordered))
    return ordered


# ---------------------------------------------------------------------------
# rows=method, columns=swept-value tables (classif gamma sweep, decimation sweep)
# ---------------------------------------------------------------------------

def render_nice_tabular(row_labels: list, col_labels: list, cells: dict,
                        row_header: str = "Method", caption: str = None,
                        label: str = None, fmt_fn=_fmt_pct,
                        bold_best_per_row: bool = True, higher_is_better: bool = True) -> str:
    """cells: {(row, col): (mean, std)}. Bolds the best cell per row (by raw mean,
    before formatting) when bold_best_per_row is True."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{NiceTabular}{l" + "c" * len(col_labels) + "}",
        r"\toprule",
        f"{row_header} & " + " & ".join(_fmt_col(c) for c in col_labels) + r" \\",
        r"\midrule",
    ]
    for r in row_labels:
        raw = [cells.get((r, c), (float('nan'), float('nan'))) for c in col_labels]
        best_idx = None
        if bold_best_per_row:
            means = []
            for m, _ in raw:
                try:
                    means.append(float(m))
                except (TypeError, ValueError):
                    means.append(float('nan'))
            valid_idxs = [i for i, v in enumerate(means) if not math.isnan(v)]
            if valid_idxs:
                best_idx = (max(valid_idxs, key=lambda i: means[i]) if higher_is_better
                           else min(valid_idxs, key=lambda i: means[i]))
        row_cells = []
        for i, (m, s) in enumerate(raw):
            cell = fmt_fn(m, s)
            if i == best_idx and cell != "--":
                cell = f"\\textbf{{{cell}}}"
            row_cells.append(cell)
        lines.append(f"{_display(r)} & " + " & ".join(row_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{NiceTabular}")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def table_from_sensitivity_dir(sens_dir: Path, scenario: str, mode: str,
                               caption: str = None, label: str = None) -> str:
    """Combine sensitivity_<scenario>_<method>_<mode>.csv (one per method) into a
    single rows=method, columns=swept-value table (decimation: F1 as %, 1 decimal,
    best value per row in bold)."""
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
    return render_nice_tabular(_order_methods(methods), col_values, cells,
                               row_header="Method", caption=caption, label=label,
                               fmt_fn=_fmt_pct, bold_best_per_row=True, higher_is_better=True)


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


# ---------------------------------------------------------------------------
# rows=Metric, columns=method table (exp1 — fixed gamma per method)
# ---------------------------------------------------------------------------

def render_metric_rows_table(methods: list, row_specs: list,
                             caption: str = None, label: str = None,
                             standalone: bool = True) -> str:
    """row_specs: [(row_label, {method: (raw_mean_or_nan, formatted_str)}, higher_is_better), ...]
    — rows=metric, columns=method. `raw_mean_or_nan` is used only to pick the bold
    "best" cell per row; NaN/missing methods are excluded from that comparison.
    `standalone=False` omits the `table`/caption/label wrapper, returning just the
    NiceTabular block (for embedding inside a `subtable`)."""
    lines = []
    if standalone:
        lines += [r"\begin{table}[htbp]", r"\centering"]
    lines += [
        r"\begin{NiceTabular}{l" + "c" * len(methods) + "}",
        r"\toprule",
        r"Metric & " + " & ".join(_display(m) for m in methods) + r" \\",
        r"\midrule",
    ]
    for row_label, values, higher_is_better in row_specs:
        raw = {m: values.get(m, (float('nan'), '--'))[0] for m in methods}
        valid = [m for m in methods
                if not (raw[m] is None or (isinstance(raw[m], float) and math.isnan(raw[m])))]
        best_m = None
        if valid:
            best_m = max(valid, key=lambda m: raw[m]) if higher_is_better else min(valid, key=lambda m: raw[m])
        row_cells = []
        for m in methods:
            s = values.get(m, (float('nan'), '--'))[1]
            if m == best_m and s != "--":
                s = f"\\textbf{{{s}}}"
            row_cells.append(s)
        lines.append(f"{_latex_escape(row_label)} & " + " & ".join(row_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{NiceTabular}")
    if standalone:
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{{label}}}")
        lines.append(r"\end{table}")
    return "\n".join(lines)


def table_from_exp1_detail(detail_csv: Path, mode: str, dataset_label: str = None,
                           label_prefix: str = None, standalone: bool = True) -> str:
    """Baseline experiment's fixed-gamma detail CSV (method x mode x seed) -> one
    NiceTabular table for the given mode: columns=method (7, union across modes so
    knn/barycenter subtables line up), rows=F1 (%, 1dp), Time (s, 1dp), RAM (MB,
    integer, mean only — rss_mb is batch-granular, not per-seed, see
    run_full_baseline.py module docstring). Best value per row in bold."""
    with open(detail_csv, newline="") as f:
        all_rows = list(csv.DictReader(f))
    rows = [r for r in all_rows if r.get("mode", "knn") == mode]

    # Always show all 7 methods as columns (union across modes in this CSV) — a
    # method absent from THIS mode (e.g. sta, excluded from barycenter fitting;
    # see run_full_baseline.py module docstring) still gets a column, rendered
    # "--", so the knn and barycenter subtables line up column-for-column.
    all_methods = {r["method"] for r in all_rows}
    methods = _order_methods(all_methods)
    if not rows:
        empty = {m: (float('nan'), '--') for m in methods}
        return render_metric_rows_table(
            methods, [("F1", empty, True), ("Time (s)", empty, False), ("RAM (MB)", empty, False)],
            caption=f"{dataset_label or ''} --- {mode}", label=None, standalone=standalone)

    f1 = _group_mean_std(rows, lambda r: r["method"], "f1")
    time = _group_mean_std(rows, lambda r: r["method"], "total_time")
    ram = _group_mean_std(rows, lambda r: r["method"], "rss_mb")

    f1_specs = {m: (f1[m][0] if m in f1 else float('nan'),
                    _fmt_pct(*f1.get(m, (float('nan'), float('nan'))))) for m in methods}
    time_specs = {m: (time[m][0] if m in time else float('nan'),
                      _fmt_time(*time.get(m, (float('nan'), float('nan'))))) for m in methods}
    ram_specs = {m: (ram[m][0] if m in ram else float('nan'),
                     (f"{ram[m][0]:.0f}" if m in ram else "--")) for m in methods}

    prefix = f"{dataset_label} --- " if dataset_label else ""
    mode_label = "KNN" if mode == "knn" else "Barycenter"
    return render_metric_rows_table(
        methods, [("F1", f1_specs, True), ("Time (s)", time_specs, False), ("RAM (MB)", ram_specs, False)],
        caption=f"{prefix}{mode_label}", label=f"{label_prefix}_{mode}" if label_prefix else None,
        standalone=standalone)


def tables_from_exp1_detail(detail_csv: Path, dataset_label: str = None,
                            label_prefix: str = None) -> str:
    """Both mode subtables (knn, barycenter) for one dataset's exp1 detail CSV."""
    knn_table = table_from_exp1_detail(detail_csv, "knn", dataset_label, label_prefix)
    bary_table = table_from_exp1_detail(detail_csv, "barycenter", dataset_label, label_prefix)
    return knn_table + "\n\n" + bary_table + "\n"


def table_from_exp1_both_datasets(river_detail_csv: Path, cpazmal_detail_csv: Path, mode: str,
                                  caption: str = None, label: str = None) -> str:
    """Combined `table*` with two `subtable` environments (River, CPAZMaL), each a
    Metric x Method NiceTabular for the given mode — matches the paper's two-dataset
    comparison layout (tab:exp1_knn-style skeleton)."""
    mode_label = "KNN, $K=1$" if mode == "knn" else "Barycenter"
    river_inner = table_from_exp1_detail(river_detail_csv, mode, standalone=False)
    cpazmal_inner = table_from_exp1_detail(cpazmal_detail_csv, mode, standalone=False)
    default_caption = (f"Comparative results ({mode_label}) on River and CPAZMaL. "
                       r"F1 in \%, best value per row in bold. The $\mathcal{D}$- prefix "
                       "denotes the divergence mode.")
    default_label = f"tab:exp1_{mode}"
    lbl = label or default_label
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"% \scalebox{\textwidth}",
        "{",
        f"\\caption{{{caption or default_caption}}}",
        f"\\label{{{lbl}}}",
        r"\begin{subtable}{\textwidth}",
        r"\centering",
        river_inner,
        r"\caption{River}",
        f"\\label{{{lbl}_river}}",
        r"\vspace{0.75em}",
        r"\end{subtable}",
        r"\begin{subtable}{\textwidth}",
        r"\centering",
        cpazmal_inner,
        r"\caption{CPAZMaL}",
        f"\\label{{{lbl}_cpazmal}}",
        r"\end{subtable}",
        "}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Extract F1 mean±std results into LaTeX (NiceTabular) tables")
    sub = parser.add_subparsers(dest="kind", required=True)

    p2 = sub.add_parser("sensitivity", help="decimation sweep table combining per-method CSVs")
    p2.add_argument("--dir", required=True)
    p2.add_argument("--scenario", required=True, choices=["decimation"])
    p2.add_argument("--mode", required=True, choices=["knn", "barycenter"])
    p2.add_argument("--out", required=True)
    p2.add_argument("--caption", default=None)
    p2.add_argument("--label", default=None)

    p4 = sub.add_parser("exp1",
                        help="per-method fixed-gamma summary table(s) (F1/time/RAM x method) "
                             "for one dataset's baseline-experiment detail CSV")
    p4.add_argument("--detail-csv", required=True)
    p4.add_argument("--out", required=True)
    p4.add_argument("--dataset-label", default=None)
    p4.add_argument("--label-prefix", default=None)
    p4.add_argument("--mode", default=None, choices=[None, "knn", "barycenter"],
                    help="single-subtable output for this mode only — omit for the "
                         "2-subtable knn+barycenter combined format (back-compat)")

    p5 = sub.add_parser("exp1_combined",
                        help="combined table* (river + cpazmal subtables) for one mode")
    p5.add_argument("--river-detail-csv", required=True)
    p5.add_argument("--cpazmal-detail-csv", required=True)
    p5.add_argument("--mode", required=True, choices=["knn", "barycenter"])
    p5.add_argument("--out", required=True)
    p5.add_argument("--caption", default=None)
    p5.add_argument("--label", default=None)

    args = parser.parse_args()

    if args.kind == "exp1":
        if args.mode:
            tex = table_from_exp1_detail(Path(args.detail_csv), args.mode,
                                         dataset_label=args.dataset_label,
                                         label_prefix=args.label_prefix)
        else:
            tex = tables_from_exp1_detail(Path(args.detail_csv),
                                          dataset_label=args.dataset_label,
                                          label_prefix=args.label_prefix)
    elif args.kind == "exp1_combined":
        tex = table_from_exp1_both_datasets(Path(args.river_detail_csv), Path(args.cpazmal_detail_csv),
                                            args.mode, caption=args.caption, label=args.label)
    else:
        tex = table_from_sensitivity_dir(Path(args.dir), args.scenario, args.mode,
                                         caption=args.caption, label=args.label)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex + "\n")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
