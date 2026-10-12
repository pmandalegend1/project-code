#!/usr/bin/env bash
# run_crossfit.sh -- five world models, each trained WITHOUT one fold of the
# planner's decision matches, plus the matching C_res table for each.
#
# Why: the planner is scored on held-out decisions, but the single world model
# had trained on windows from 92% of those decisions' matches. It had seen how
# each formation actually did in them, so its critic scores were contaminated
# and the one "pass" we got vanished on the clean subset. With five
# cross-fitted models every one of the 2,133 decisions can be scored by a
# model that never saw its match -- clean, and 15x more data than the 146
# clean decisions we had.
#
# Needs crossfit/fold{0..4}_matches.txt, written by make_crossfit_folds.py in
# the ramfp repo (same GroupKFold as the opponent model, so the folds agree).
#
# Start (survives logout / laptop off):
#   nohup bash run_crossfit.sh > logs/crossfit.out 2>&1 &
# Progress:  cat logs/crossfit_status.txt
# One fold:  tail -f logs/wm_excl0.log
# Re-running skips folds that are already finished.

set -u
cd "$(dirname "$0")"
mkdir -p logs checkpoints results crossfit
FOLDS=${FOLDS:-"0 1 2 3 4"}
export CONFIG=${CONFIG:-configs/world_model.yaml}
WIN=$(grep -E '^stable_windows_path:' "$CONFIG" | awk '{print $2}')
ST=logs/crossfit_status.txt
echo $$ > logs/crossfit.pid

for k in $FOLDS; do
  EX=crossfit/fold${k}_matches.txt
  RMS=data/processed/role_matchup_stats_excl${k}.parquet
  CK=checkpoints/world_model_best_excl${k}.pt
  CRES=results/counter_resistance_lookup_excl${k}.json
  if [ ! -f "$EX" ]; then
    echo "fold $k: $EX missing -- run make_crossfit_folds.py first" | tee -a "$ST"; exit 1
  fi
  if [ -f "$CRES" ]; then
    echo "fold $k: already done, skipping ($(date '+%F %T'))" >> "$ST"; continue
  fi

  echo "fold $k: role-matchup stats without its matches ($(date '+%F %T'))" >> "$ST"
  python compute_role_matchup_stats.py --windows "$WIN" --exclude-matches "$EX" --out "$RMS" \
      >> logs/wm_excl${k}.log 2>&1 || { echo "fold $k: rms FAILED" >> "$ST"; exit 1; }

  echo "fold $k: training world model ($(date '+%F %T'))" >> "$ST"
  EXTRA="--exclude-matches $EX --tag excl${k} --rms $RMS" \
  LOG=logs/wm_excl${k}.log STATUS=logs/train_status_excl${k}.txt \
    bash run_training.sh
  if ! grep -q FINISHED logs/train_status_excl${k}.txt; then
    echo "fold $k: training did not finish -- $(cat logs/train_status_excl${k}.txt)" >> "$ST"; exit 1
  fi

  echo "fold $k: C_res table ($(date '+%F %T'))" >> "$ST"
  CUDA_VISIBLE_DEVICES=${GPU:-1} python counter_resistance.py --config "$CONFIG" --checkpoint "$CK" \
      --exclude-matches "$EX" --out "$CRES" >> logs/wm_excl${k}.log 2>&1 \
      || { echo "fold $k: C_res FAILED" >> "$ST"; exit 1; }
  echo "fold $k: DONE ($(date '+%F %T'))" >> "$ST"
done
echo "ALL FOLDS DONE ($(date '+%F %T')). Next: learn_ranker.py --crossfit" >> "$ST"
