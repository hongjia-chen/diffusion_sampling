#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/v3_matched}
ENVIRONMENT=${ENVIRONMENT:-diffusion}
GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-3}
mkdir -p "$OUTPUT_ROOT/trajectory_replay_logs"

run_worker() {
  local physical_gpu=$1
  local log_name=$2
  shift 2
  CUDA_VISIBLE_DEVICES="$physical_gpu" \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  conda run -n "$ENVIRONMENT" python scripts/replay_adaptive_trajectories.py \
    --output-root "$OUTPUT_ROOT" \
    --device cuda:0 \
    --local-files-only \
    --methods "$@" \
    >"$OUTPUT_ROOT/trajectory_replay_logs/$log_name.log" 2>&1
}

# Balanced using the measured total latency of each delta across all three
# benchmarks.  Each worker is independently resumable at task granularity.
run_worker "$GPU_A" gpu_a \
  adaptive_delta_0p10 adaptive_delta_0p20 \
  adaptive_delta_0p75 adaptive_delta_1p50 &
PID_A=$!
run_worker "$GPU_B" gpu_b \
  adaptive_delta_0p15 adaptive_delta_0p30 \
  adaptive_delta_0p50 adaptive_delta_1p00 &
PID_B=$!

STATUS=0
wait "$PID_A" || STATUS=$?
wait "$PID_B" || STATUS=$?
if [[ "$STATUS" -ne 0 ]]; then
  echo "trajectory replay failed; inspect $OUTPUT_ROOT/trajectory_replay_logs" >&2
  exit "$STATUS"
fi

conda run -n "$ENVIRONMENT" python report/make_trajectory_tables.py \
  --output-root "$OUTPUT_ROOT"
echo "trajectory replay and report tables complete"
