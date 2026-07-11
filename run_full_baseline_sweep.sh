#!/usr/bin/env bash
# Sequential per-(dataset, gamma) baseline runs — one OS process per combination.
#
# Why: run_full_baseline.py's STA method accumulates RSS within a loky worker over the
# course of a single method's batch (~5GB/worker growing over ~60min for 4-5 seeds on
# river) — even with the in-process pool teardown between methods, running the whole
# gamma sweep in ONE long-lived parent process still repeatedly pays that risk and keeps
# growing whatever the parent itself accumulates (loaded CSV rows, cfg dicts, etc.).
# Process exit is the only guaranteed full memory reclaim, so this script gives each
# (dataset, gamma) pair its own process and lets the OS clean up completely in between.
#
# Safe to interrupt/rerun: run_full_baseline.py skips (method, gamma) pairs already
# present in the output CSVs unless --force, so re-running this script resumes rather
# than recomputing everything.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate

CONFIG="configs/full_baseline.yaml"
N_JOBS=4
LOG="results/jax_full_baseline/full_baseline_sweep.log"
mkdir -p "$(dirname "$LOG")"

DATASETS=(river cpazmal)
GAMMAS=(1.0e-2 1.0 1.0e+2)

echo "===== $(date '+%Y-%m-%d %H:%M:%S') full_baseline_sweep starting =====" | tee -a "$LOG"

for ds in "${DATASETS[@]}"; do
  for g in "${GAMMAS[@]}"; do
    echo "----- $(date '+%H:%M:%S') dataset=${ds} gamma=${g} starting -----" | tee -a "$LOG"
    STA_JOBS_FLAG=()
    if [[ "$ds" == "cpazmal" ]]; then
      # cpazmal has ~5.75x more (test,train) pairs than river at equal per-class
      # caps (7 vs 4 classes) — STA OOM'd twice at n_jobs=4 (2026-07-10), reduce
      # STA's own concurrency while keeping the other 6 methods at N_JOBS.
      STA_JOBS_FLAG=(--sta-n-jobs 2)
    fi
    python experiments/run_full_baseline.py --config "$CONFIG" \
        --dataset "$ds" --gamma "$g" --n-jobs "$N_JOBS" "${STA_JOBS_FLAG[@]}" --debug --verbose \
        2>&1 | tee -a "$LOG"
    echo "----- $(date '+%H:%M:%S') dataset=${ds} gamma=${g} done -----" | tee -a "$LOG"
  done
done

echo "===== $(date '+%Y-%m-%d %H:%M:%S') full_baseline_sweep done =====" | tee -a "$LOG"
