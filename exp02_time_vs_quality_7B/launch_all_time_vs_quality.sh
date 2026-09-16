#!/usr/bin/env bash
# Detached launcher for the complete fixed + adaptive Dream experiment.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults for the 0729 overnight design. Environment values still override
# every setting without requiring another edit to this file.
export DATASET="${DATASET:-$SCRIPT_DIR/data/owt_p64_l128.jsonl}"
export DATASET_METADATA="${DATASET_METADATA:-$SCRIPT_DIR/data/owt_p64_l128.metadata.json}"
export GPU_IDS="${GPU_IDS:-0 1 2 3}"
export NUM_EXAMPLES="${NUM_EXAMPLES:-500}"
export TIMING_REPEATS="${TIMING_REPEATS:-1}"
export DATE_TAG="${DATE_TAG:-0729}"

RUN_TAG="${RUN_TAG:-${DATE_TAG}_time_vs_quality_7B_n${NUM_EXAMPLES}_4gpu}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${RUN_TAG}.log}"
PID_FILE="${PID_FILE:-$LOG_DIR/${RUN_TAG}.pid}"
NICE_LEVEL="${NICE_LEVEL:-10}"
VERIFY_DELAY_SECONDS="${VERIFY_DELAY_SECONDS:-5}"

for path in "$DATASET" "$DATASET_METADATA"; do
    if [ ! -f "$path" ]; then
        echo "ERROR: dataset artifact does not exist: $path" >&2
        exit 1
    fi
done

mkdir -p "$LOG_DIR"
if [ ! -f run_all_time_vs_quality_pipeline.sh ]; then
    echo "ERROR: run_all_time_vs_quality_pipeline.sh is missing." >&2
    exit 1
fi
if [ -f "$PID_FILE" ]; then
    OLD_PID="$(tr -cd '0-9' < "$PID_FILE" || true)"
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ERROR: $RUN_TAG is already running as PID $OLD_PID." >&2
        exit 1
    fi
fi

echo "Launching the complete fixed + adaptive experiment..."
echo "  dataset      : $DATASET"
echo "  metadata     : $DATASET_METADATA"
echo "  physical GPUs: $GPU_IDS"
echo "  examples     : $NUM_EXAMPLES"
echo "  repeats      : $TIMING_REPEATS"
echo "  P / L        : ${PREFIX_LENGTH:-64} / ${CONTINUATION_LENGTH:-128}"
echo "  log          : $LOG_FILE"

nohup nice -n "$NICE_LEVEL" \
bash "$SCRIPT_DIR/run_all_time_vs_quality_pipeline.sh" \
    > "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

sleep "$VERIFY_DELAY_SECONDS"
if ! kill -0 "$PID" 2>/dev/null; then
    echo "ERROR: combined pipeline exited during startup." >&2
    tail -n 100 "$LOG_FILE" || true
    exit 1
fi

echo "[OK] combined pipeline is running as PID $PID."
echo "Monitor: tail -f '$LOG_FILE'"
echo "GPU:     watch -n 2 nvidia-smi"
echo "Stop:    kill $PID"
