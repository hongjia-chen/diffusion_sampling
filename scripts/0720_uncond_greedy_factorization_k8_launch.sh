#!/bin/bash
# =============================================================================
# 0720 — Dream-7B unconditional greedy exhaustive factorization (max_k=8)
# =============================================================================
# One detached, niced rollout on one GPU, followed by automatic plotting.
#
# Experiment:
#   mode                        : unconditional
#   canvas length               : 72
#   denoising steps             : 64
#   max_k                       : 8
#   subset sizes                : 1, 2, 3, 4, 5, 6, 7, 8
#   subsets per step            : 2^8 - 1 = 255
#   total estimate rows         : 64 * 255 = 16,320
#   conditional microbatch size : 1
#   rollout                     : deterministic greedy commitment
#
# Place this launcher inside the repository's scripts/ directory. It resolves
# the repository root relative to its own location, so it does not depend on
# the caller's working directory.
# =============================================================================

set -euo pipefail

# --- Run identity / resources ---
GPU_ID="${GPU_ID:-0}"
RUN_ID="${RUN_ID:-uncond-greedy-k8-64step-001}"
DATE_TAG="${DATE_TAG:-0720}"
NICE_LEVEL="${NICE_LEVEL:-10}"
VERIFY_DELAY_SECONDS="${VERIFY_DELAY_SECONDS:-30}"

CANVAS_LENGTH=72
NUM_STEPS=64
MAX_K=8
SUBSET_SIZE_ARGS=(1 2 3 4 5 6 7 8)

# Optionally enforce a particular conda environment at launch time:
#   EXPECTED_CONDA_ENV=dream bash scripts/0720_uncond_greedy_factorization_k8_launch.sh
EXPECTED_CONDA_ENV="${EXPECTED_CONDA_ENV:-}"

# Resolve the repository root from scripts/<this-file>.sh.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

PYTHON_SCRIPT="scripts/run_factorization_rollout.py"
PLOT_SCRIPT="scripts/plot_factorization_results.py"
OUTPUT_DIR="outputs/factorization"
RUN_OUTPUT_DIR="$OUTPUT_DIR/$RUN_ID"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/${DATE_TAG}_${RUN_ID}.log"
PID_FILE="$LOG_DIR/${DATE_TAG}_${RUN_ID}.pid"

EXPECTED_SUBSETS=$(((1 << MAX_K) - 1))
EXPECTED_ROWS=$((NUM_STEPS * EXPECTED_SUBSETS))

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# --- Cluster/runtime safeguards ---
# LOAD-BEARING on this cluster: capped virtual memory can make large framework
# initializations fail with pthread_create/EAGAIN despite ample physical RAM.
ulimit -v unlimited

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

# --- Fail-early checks ---
for required_script in "$PYTHON_SCRIPT" "$PLOT_SCRIPT"; do
    if [ ! -f "$required_script" ]; then
        echo "ERROR: $required_script was not found under:"
        echo "  $PROJECT_DIR"
        echo "Place this launcher in the repository's scripts/ directory or set PROJECT_DIR."
        exit 1
    fi
done

if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "ERROR: no conda environment is active."
    echo "Activate the environment used for the Dream factorization code, then relaunch."
    exit 1
fi

if [ -n "$EXPECTED_CONDA_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "$EXPECTED_CONDA_ENV" ]; then
    echo "ERROR: expected conda env '$EXPECTED_CONDA_ENV', but '$CONDA_DEFAULT_ENV' is active."
    exit 1
fi

# Non-root users can reliably lower priority with nice levels 0 through 19.
if ! [[ "$NICE_LEVEL" =~ ^([0-9]|1[0-9])$ ]]; then
    echo "ERROR: NICE_LEVEL must be an integer from 0 through 19; received '$NICE_LEVEL'."
    exit 1
fi

echo "[OK] conda environment: $CONDA_DEFAULT_ENV"

# Refuse to overwrite the PID record for an already-running launch with the
# same run ID. A stale PID file is harmless and will be replaced below.
if [ -f "$PID_FILE" ]; then
    OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ERROR: run '$RUN_ID' already appears active as PID $OLD_PID."
        echo "Monitor: tail -f $LOG_FILE"
        echo "Stop:    kill $OLD_PID"
        exit 1
    fi
fi

python -m py_compile "$PYTHON_SCRIPT" "$PLOT_SCRIPT"
python "$PYTHON_SCRIPT" --help >/dev/null
python "$PLOT_SCRIPT" --help >/dev/null
printf '%s\n' "[OK] rollout and plotting scripts import and parse arguments."

# Check the selected physical GPU in isolation. CUDA_VISIBLE_DEVICES maps it to
# logical cuda:0 for both this check and the detached job.
CUDA_VISIBLE_DEVICES="$GPU_ID" python - <<'PY'
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    sys.exit("ERROR: PyTorch cannot see the requested CUDA GPU.")

props = torch.cuda.get_device_properties(0)
print(
    f"[OK] CUDA logical device 0: {props.name} "
    f"({props.total_memory / 1024**3:.1f} GiB)"
)

if not torch.cuda.is_bf16_supported():
    sys.exit("ERROR: the selected GPU/PyTorch build does not report bfloat16 support.")
print("[OK] bfloat16 is supported.")
PY

# --- Launch ---
echo ""
echo "Launching Dream-7B unconditional exhaustive factorization rollout..."
echo "  physical GPU : $GPU_ID"
echo "  logical device: cuda:0"
echo "  nice level   : $NICE_LEVEL"
echo "  max_k        : $MAX_K"
echo "  subset sizes : ${SUBSET_SIZE_ARGS[*]}"
echo "  subsets/step : $EXPECTED_SUBSETS"
echo "  expected rows: $EXPECTED_ROWS"
echo "  run id       : $RUN_ID"
echo "  log          : $LOG_FILE"
echo "  run output   : $RUN_OUTPUT_DIR"
echo "  plot output  : $RUN_OUTPUT_DIR/plots"

# nice is inside nohup so both the rollout and the subsequent plotting stage
# inherit the requested lower scheduling priority for their entire lifetimes.
# The plotter runs only if the rollout exits successfully.
CUDA_VISIBLE_DEVICES="$GPU_ID" \
nohup nice -n "$NICE_LEVEL" \
bash -c '
set -euo pipefail

runner=$1
plotter=$2
run_output_dir=$3
canvas_length=$4
num_steps=$5
max_k=$6
run_id=$7
output_dir=$8
shift 8
subset_sizes=("$@")

echo "[pipeline] Rollout started: $(date "+%Y-%m-%dT%H:%M:%S%z")"
python -u "$runner" \
    --mode unconditional \
    --device cuda:0 \
    --dtype bfloat16 \
    --canvas-length "$canvas_length" \
    --num-steps "$num_steps" \
    --max-k "$max_k" \
    --subset-sizes "${subset_sizes[@]}" \
    --conditional-microbatch-size 1 \
    --run-id "$run_id" \
    --output-dir "$output_dir" \
    --local-files-only \
    --overwrite

echo "[pipeline] Rollout completed: $(date "+%Y-%m-%dT%H:%M:%S%z")"
echo "[pipeline] Generating plots and report..."
python -u "$plotter" "$run_output_dir"
echo "[pipeline] Plotting completed: $(date "+%Y-%m-%dT%H:%M:%S%z")"
echo "[pipeline] Images: $run_output_dir/plots"
echo "[pipeline] Report: $run_output_dir/plots/report.tex"
' pipeline \
    "$PYTHON_SCRIPT" \
    "$PLOT_SCRIPT" \
    "$RUN_OUTPUT_DIR" \
    "$CANVAS_LENGTH" \
    "$NUM_STEPS" \
    "$MAX_K" \
    "$RUN_ID" \
    "$OUTPUT_DIR" \
    "${SUBSET_SIZE_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
echo "  PID           : $PID"

# --- Verify ---
echo ""
echo "Sleeping ${VERIFY_DELAY_SECONDS}s, then checking startup and niceness..."
sleep "$VERIFY_DELAY_SECONDS"

if ! kill -0 "$PID" 2>/dev/null; then
    echo "ERROR: PID $PID exited during startup."
    echo ""
    echo "Last 80 log lines:"
    tail -n 80 "$LOG_FILE" || true
    exit 1
fi

ACTUAL_NICE="$(ps -o ni= -p "$PID" | tr -d '[:space:]')"
if ! [[ "$ACTUAL_NICE" =~ ^-?[0-9]+$ ]] || [ "$ACTUAL_NICE" -lt "$NICE_LEVEL" ]; then
    echo "ERROR: PID $PID is running at nice level '${ACTUAL_NICE:-unknown}', expected at least $NICE_LEVEL."
    echo "Stopping the improperly prioritized job."
    kill "$PID" 2>/dev/null || true
    exit 1
fi

echo "[OK] PID $PID is alive at nice level $ACTUAL_NICE."
echo ""
echo "Recent log output:"
tail -n 20 "$LOG_FILE" || true

echo ""
echo "Monitor log:     tail -f $LOG_FILE"
echo "Monitor process: ps -o pid,ni,etime,%cpu,%mem,command -p \$(cat $PID_FILE)"
echo "Monitor GPU:     watch -n 2 nvidia-smi"
echo "Stop cleanly:    kill \$(cat $PID_FILE)"
echo "Run output:      $RUN_OUTPUT_DIR"
echo "Images/report:   $RUN_OUTPUT_DIR/plots"
