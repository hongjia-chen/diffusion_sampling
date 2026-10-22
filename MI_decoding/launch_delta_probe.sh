#!/usr/bin/env bash
# δ probe for the MI rule at B=64 (8B proposes, 1.7B verifies): where does the
# mean number of tokens committed per step stop moving, at either end?
#
# The same TASKS-per-benchmark sample for every δ, so the arms are paired, and
# the b64 baselines (outputs/opencompass/SDAR-*-b64) cover the same ids.
# Two passes so the cheap part of the curve lands first: large δ commits whole
# blocks, runs degenerate answers out to the 4096 canvas and scores all 64
# verifier rows every step (minutes per task).
#   pass 1: δ 0 .. 10   (τ = log(1+δ) 0 .. 2.4 nats)
#   pass 2: δ 30 .. 1e4 (τ 3.4 .. 9.2 nats)
# Each pass is resumable; a crashed pass is retried up to twice.
#
#   bash launch_delta_probe.sh
#   TASKS=10 PAIR=0,1 bash launch_delta_probe.sh
#   PASS1= BENCHMARKS="gsm8k humaneval mbpp_sanitized" bash launch_delta_probe.sh   # pass 2 only
#   python analyze_delta_probe.py      # the curve, with the baselines
set -uo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
export TASKS="${TASKS:-20}"
export PAIR="${PAIR:-0,1}"
export BENCHMARKS="${BENCHMARKS:-gsm8k humaneval mbpp_sanitized math_500}"
export TRACE_LIMIT="${TRACE_LIMIT:-0}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/mi_probe/8b_proposes}"
LOG_BASE="${LOG_ROOT:-$PROJECT_DIR/logs/mi_probe/8b_proposes}"
PASS1="${PASS1-0 0.001 0.01 0.03 0.10 0.30 1.00 3.00 10.00}"
PASS2="${PASS2-30 100 1000 10000}"

run_pass() {
    local name="$1" deltas="$2" attempt
    for attempt in 1 2 3; do
        echo "== $name, attempt $attempt ($(date '+%F %T')): δ $deltas"
        # launch_mi.sh rewrites $LOG_ROOT/<bench>.log, so each attempt gets its own.
        DELTAS="$deltas" LOG_ROOT="$LOG_BASE/${name// /}_attempt$attempt" bash launch_mi.sh && return 0
        echo "== $name attempt $attempt failed ($(date '+%F %T'))"
    done
    return 1
}

# An empty PASS1 or PASS2 skips that pass.
[ -z "$PASS1" ] || run_pass "pass 1" "$PASS1" || exit 1
[ -z "$PASS2" ] || run_pass "pass 2" "$PASS2" || exit 1
"${PYTHON_BIN:-$HOME/miniconda3/envs/diffusion/bin/python}" analyze_delta_probe.py --probe-root "$OUTPUT_ROOT"
echo "== probe done $(date '+%F %T')"
