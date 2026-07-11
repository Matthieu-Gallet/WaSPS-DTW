#!/usr/bin/env bash
# Extra cpazmal-only gamma sweep — gamma = 0.001, 0.1, 10, 10000 (4 values not covered
# by run_full_baseline_sweep.sh's 0.01/1/100). river is untouched. Same sequential
# per-gamma OS process pattern, same --sta-n-jobs 2 mitigation validated on the main
# sweep (cpazmal STA has ~5.75x more (test,train) pairs than river at these sample
# caps due to 7 vs 4 classes — n_jobs=4 OOM'd twice, n_jobs=2 held stable for all 3
# main-sweep gamma values).
#
# Expensive: each gamma value's STA takes ~170min (2 rounds x ~85min at n_jobs=2) on
# top of the 6 fast methods (~seconds). 4 values ~= 11-12h total.
#
# Run AFTER run_sensitivity_pipeline.sh completes — do not run concurrently with it
# (grid_bary's nested joblib parallelism + this script's STA workers would compete
# for memory).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate

CONFIG="configs/full_baseline.yaml"
N_JOBS=4
LOG="results/jax_full_baseline/full_baseline_sweep.log"
mkdir -p "$(dirname "$LOG")"

GAMMAS=(1.0e-3 1.0e-1 1.0e+1 1.0e+4)

echo "===== $(date '+%Y-%m-%d %H:%M:%S') full_baseline_cpazmal_extra starting =====" | tee -a "$LOG"

for g in "${GAMMAS[@]}"; do
  echo "----- $(date '+%H:%M:%S') dataset=cpazmal gamma=${g} starting -----" | tee -a "$LOG"
  python experiments/run_full_baseline.py --config "$CONFIG" \
      --dataset cpazmal --gamma "$g" --n-jobs "$N_JOBS" --sta-n-jobs 2 --debug --verbose \
      2>&1 | tee -a "$LOG"
  echo "----- $(date '+%H:%M:%S') dataset=cpazmal gamma=${g} done -----" | tee -a "$LOG"
done

echo "===== $(date '+%Y-%m-%d %H:%M:%S') full_baseline_cpazmal_extra done =====" | tee -a "$LOG"
