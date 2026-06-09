#!/usr/bin/env python3
"""
Run sdtw_barycenter_classification.py for multiple random seeds and aggregate metrics.

Example:
  python src/experiments/seed_sweep_runner.py --n-seeds 50
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _parse_seeds(seeds_arg: str | None, seed_start: int, n_seeds: int) -> List[int]:
    if seeds_arg:
        return [int(x.strip()) for x in seeds_arg.split(",") if x.strip()]
    return list(range(seed_start, seed_start + n_seeds))


def _extract_total_experiment_time(stdout: str) -> float:
    match = re.search(r"Total experiment time:\s*([0-9]+(?:\.[0-9]+)?)s", stdout)
    if match:
        return float(match.group(1))
    return float("nan")


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    default_script = repo_root / "src" / "experiments" / "sdtw_barycenter_classification.py"
    default_runs_dir = repo_root / "results" / "regime_classification" / "seed_sweep" / "runs"
    default_out_dir = repo_root / "results" / "regime_classification" / "seed_sweep"

    parser = argparse.ArgumentParser(
        description="Run regime classification for many seeds and aggregate results in pandas DataFrames."
    )
    parser.add_argument("--script", type=str, default=str(default_script), help="Path to experiment script")
    parser.add_argument("--dataset-mode", type=str, default="balanced", choices=["basic", "balanced"])
    parser.add_argument("--mode", type=str, default="one-shot", choices=["one-shot", "kfold", "gamma-sens", "sample-sens"])
    parser.add_argument("--seeds", type=str, default=None, help="Explicit comma-separated seeds, e.g. 1,7,42")
    parser.add_argument("--seed-start", type=int, default=0, help="Start seed when --seeds is not provided")
    parser.add_argument("--n-seeds", type=int, default=50, help="Number of seeds when --seeds is not provided")
    parser.add_argument("--python-exec", type=str, default=sys.executable, help="Python executable to run experiments")
    parser.add_argument("--runs-dir", type=str, default=str(default_runs_dir), help="Directory that stores per-seed run outputs")
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(default_out_dir / "seed_sweep_scores.csv"),
        help="Output CSV path for per-seed, per-method scores",
    )
    parser.add_argument(
        "--output-meta-csv",
        type=str,
        default=str(default_out_dir / "seed_sweep_runs.csv"),
        help="Output CSV path for per-seed run metadata",
    )
    parser.add_argument(
        "--output-summary-csv",
        type=str,
        default=str(default_out_dir / "seed_sweep_summary_by_method.csv"),
        help="Output CSV path for aggregated summary by method",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args forwarded to sdtw_barycenter_classification.py (use after --extra-args)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    script_path = Path(args.script).resolve()
    repo_root = script_path.parents[2]
    runs_dir = Path(args.runs_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_meta_csv = Path(args.output_meta_csv).resolve()
    output_summary_csv = Path(args.output_summary_csv).resolve()

    runs_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_meta_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seeds, args.seed_start, args.n_seeds)

    run_meta_records = []
    scores_frames = []

    print(f"Running {len(seeds)} experiments from seed list: {seeds}")

    for i, seed in enumerate(seeds, start=1):
        run_base = runs_dir / f"seed_{seed:04d}"
        run_base.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.python_exec,
            str(script_path),
            "--dataset-mode",
            args.dataset_mode,
            "--mode",
            args.mode,
            "--random-seed",
            str(seed),
            "--output-dir",
            str(run_base),
        ] + args.extra_args

        print(f"[{i}/{len(seeds)}] Seed {seed} -> {' '.join(cmd)}")

        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        wall_time = time.perf_counter() - t0

        stdout_path = run_base / "stdout.log"
        stderr_path = run_base / "stderr.log"
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")

        total_experiment_time = _extract_total_experiment_time(proc.stdout)

        run_meta = {
            "seed": seed,
            "status": "ok" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "run_wall_time_s": wall_time,
            "total_experiment_time_s": total_experiment_time,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }

        scores_path = run_base / args.mode / "classification_scores.csv"
        if proc.returncode == 0 and scores_path.exists():
            df_scores = pd.read_csv(scores_path)
            df_scores.insert(0, "seed", seed)
            df_scores["scores_csv"] = str(scores_path)
            scores_frames.append(df_scores)
            run_meta["scores_found"] = True
        else:
            run_meta["scores_found"] = False

        run_meta_records.append(run_meta)

    run_meta_df = pd.DataFrame(run_meta_records)
    run_meta_df.to_csv(output_meta_csv, index=False)

    if scores_frames:
        scores_df = pd.concat(scores_frames, ignore_index=True)
        scores_df = scores_df.merge(
            run_meta_df[["seed", "status", "run_wall_time_s", "total_experiment_time_s"]],
            on="seed",
            how="left",
        )
        scores_df.to_csv(output_csv, index=False)

        summary_df = (
            scores_df.groupby("method", as_index=False)
            .agg(
                n_runs=("seed", "count"),
                f1_weighted_mean=("f1_weighted", "mean"),
                f1_weighted_std=("f1_weighted", "std"),
                f1_macro_mean=("f1_macro", "mean"),
                f1_macro_std=("f1_macro", "std"),
                barycenter_time_mean=("barycenter_time", "mean"),
                barycenter_time_std=("barycenter_time", "std"),
                classify_time_mean=("classify_time", "mean"),
                classify_time_std=("classify_time", "std"),
                run_wall_time_mean=("run_wall_time_s", "mean"),
                run_wall_time_std=("run_wall_time_s", "std"),
            )
            .sort_values("f1_weighted_mean", ascending=False)
        )
        summary_df.to_csv(output_summary_csv, index=False)

        print("\nDone.")
        print(f"Per-run metadata saved to: {output_meta_csv}")
        print(f"Per-seed scores saved to: {output_csv}")
        print(f"Summary by method saved to: {output_summary_csv}")
    else:
        empty_scores_df = pd.DataFrame(
            columns=[
                "seed",
                "method",
                "f1_weighted",
                "f1_macro",
                "barycenter_time",
                "classify_time",
                "scores_csv",
                "status",
                "run_wall_time_s",
                "total_experiment_time_s",
            ]
        )
        empty_scores_df.to_csv(output_csv, index=False)

        empty_summary_df = pd.DataFrame(
            columns=[
                "method",
                "n_runs",
                "f1_weighted_mean",
                "f1_weighted_std",
                "f1_macro_mean",
                "f1_macro_std",
                "barycenter_time_mean",
                "barycenter_time_std",
                "classify_time_mean",
                "classify_time_std",
                "run_wall_time_mean",
                "run_wall_time_std",
            ]
        )
        empty_summary_df.to_csv(output_summary_csv, index=False)

        print("\nNo successful run with readable classification_scores.csv.")
        print(f"Per-run metadata saved to: {output_meta_csv}")
        print(f"Empty score file saved to: {output_csv}")
        print(f"Empty summary file saved to: {output_summary_csv}")


if __name__ == "__main__":
    main()
