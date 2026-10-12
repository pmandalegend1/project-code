#!/usr/bin/env bash
# run_training.sh -- start world model training so it survives everything
# except the server itself going down.
#
# Two separate problems, often confused:
#
#   1. Your laptop closing / SSH dropping. `nohup` already handles this: the
#      job runs on THIS server, detached from the login shell. Closing the
#      terminal, losing wifi or shutting the laptop down does not touch it.
#
#   2. The job dying on its own. That is what has actually happened here: the
#      GPU is shared, another user allocated 21 GB, and the run was killed by
#      CUDA OOM at epoch 1. `nohup` does nothing about that. This script does:
#      it restarts, resumes from the last checkpoint, and halves the batch
#      size each time it is killed by OOM.
#
# Usage:
#   bash run_training.sh              # start (or resume if a checkpoint exists)
#   bash run_training.sh --fresh      # ignore any existing checkpoint
#
# Watch it:   tail -f logs/world_model.log
# Status:     cat logs/train_status.txt
# Stop it:    kill $(cat logs/train.pid)

set -u
cd "$(dirname "$0")"
mkdir -p logs checkpoints results

CONFIG=${CONFIG:-configs/world_model.yaml}
GPU=${GPU:-1}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-6}
MIN_BATCH=${MIN_BATCH:-4}

RESUME="--resume"
if [ "${1:-}" = "--fresh" ]; then
  RESUME=""
  echo "starting fresh; existing checkpoint will be overwritten"
fi

LOG=${LOG:-logs/world_model.log}
STATUS=${STATUS:-logs/train_status.txt}
# extra arguments for train_world_model.py, e.g. the cross-fit exclusion:
#   EXTRA="--exclude-matches crossfit/fold0_matches.txt --tag excl0 --rms ..." bash run_training.sh
EXTRA=${EXTRA:-}

# --- preflight -------------------------------------------------------------
# The usual way this run dies is not the GPU: it is a fresh SSH session sitting
# in the (base) conda environment, where torch and numpy are not installed.
# Without this check the script retried six times, failing in a second each
# time on ModuleNotFoundError, and reported it as if the training had failed.
if ! python -c "import torch, numpy, pandas, torch_geometric" 2>/dev/null; then
  echo "environment is missing torch/numpy/pandas/torch_geometric; trying to activate ramfp"
  # shellcheck disable=SC1091
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate ramfp 2>/dev/null || true
  fi
fi
if ! python -c "import torch, numpy, pandas, torch_geometric" 2>/dev/null; then
  echo
  echo "ERROR: this shell cannot import torch/numpy/pandas/torch_geometric."
  echo "You are probably in the (base) environment. Run:"
  echo
  echo "    conda activate ramfp"
  echo
  echo "and start this script again. Nothing was launched."
  printf 'NOT STARTED: wrong conda environment (run: conda activate ramfp)\n' > "$STATUS"
  exit 2
fi
echo "environment OK: $(python -c 'import torch; print("torch", torch.__version__)')"


batch_size() { grep -E '^batch_size:' "$CONFIG" | awk '{print $2}'; }
set_batch_size() { sed -i "s/^batch_size: *[0-9]\+/batch_size: $1/" "$CONFIG"; }

echo "config      : $CONFIG"
echo "gpu         : $GPU"
echo "batch size  : $(batch_size)"
echo "max attempts: $MAX_ATTEMPTS"
echo "log         : $LOG"
[ -n "$EXTRA" ] && echo "extra args  : $EXTRA"
echo

attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  bs=$(batch_size)
  echo "=== attempt $attempt/$MAX_ATTEMPTS, batch_size=$bs, $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG"
  printf 'attempt %s/%s | batch_size %s | started %s | RUNNING\n' \
    "$attempt" "$MAX_ATTEMPTS" "$bs" "$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"

  CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python train_world_model.py --config "$CONFIG" $RESUME $EXTRA >> "$LOG" 2>&1
  code=$?

  if [ "$code" -eq 0 ]; then
    printf 'FINISHED cleanly after %s attempt(s) at %s\n' \
      "$attempt" "$(date '+%Y-%m-%d %H:%M:%S')" > "$STATUS"
    echo "training finished; see $LOG" >> "$LOG"
    exit 0
  fi

  # every attempt after the first continues from the checkpoint
  RESUME="--resume"

  if tail -40 "$LOG" | grep -qi "OutOfMemoryError\|CUDA out of memory"; then
    new_bs=$(( bs / 2 ))
    if [ "$new_bs" -lt "$MIN_BATCH" ]; then
      printf 'GAVE UP: out of memory even at batch_size %s. The GPU is too full.\n' \
        "$bs" > "$STATUS"
      echo "out of memory at the minimum batch size; stopping" >> "$LOG"
      exit 1
    fi
    set_batch_size "$new_bs"
    echo "killed by CUDA OOM (the GPU is shared); batch_size $bs -> $new_bs, waiting 120s" >> "$LOG"
    printf 'attempt %s hit CUDA OOM | batch_size %s -> %s | retrying\n' \
      "$attempt" "$bs" "$new_bs" > "$STATUS"
    sleep 120
  elif tail -40 "$LOG" | grep -q "ModuleNotFoundError\|ImportError"; then
    # An import error will not fix itself on the next attempt, so retrying six
    # times only delays the message the user needs to read.
    missing=$(tail -40 "$LOG" | grep -o "No module named '[^']*'" | tail -1)
    printf 'NOT A TRAINING FAILURE: %s. Run: conda activate ramfp\n' "${missing:-import error}" > "$STATUS"
    echo "import error, not retrying: ${missing:-see above}" >> "$LOG"
    exit 2
  else
    echo "exited with code $code for a reason other than OOM; retrying in 60s" >> "$LOG"
    printf 'attempt %s failed (exit %s, not OOM) | retrying\n' "$attempt" "$code" > "$STATUS"
    sleep 60
  fi

  attempt=$(( attempt + 1 ))
done

printf 'GAVE UP after %s attempts. Read the end of %s.\n' "$MAX_ATTEMPTS" "$LOG" > "$STATUS"
exit 1
