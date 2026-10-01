#!/usr/bin/env bash
# Density-ratio VGB, Track A (DESIGN.md §0, §8): SDAR-8B target, SDAR-1.7B
# reference, root start, the paper's shortlist forward rule (8 positions x 8
# draws from 1.7B, reweighted by 8B/1.7B), interior leaf rule, momentum,
# exact leave-one-out. Same 100 task IDs and canvases as the 8B fixed-k run.
#
#   bash launch_vgb.sh                    # refuses if another user's job is on the cards
#   PAIR=2,3 bash launch_vgb.sh
#   DRY_RUN=1 bash launch_vgb.sh
#
# One replica, one card pair: 8B is pipeline-split over the pair and 1.7B
# shares the second card. Benchmarks run in order, cheapest first; each runs
# the no-remasking control, then VGB. Resumable: rerunning skips finished tasks.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
# The shared harness (package, runner and scoring scripts) lives one level up.
HARNESS_DIR="$(cd -- "$PROJECT_DIR/../harness" && pwd)"
export PYTHONPATH="$HARNESS_DIR:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
# No expandable_segments: on this host it fails at model load with "CUDA
# driver error: out of memory" on empty cards. ROW_CHUNK bounds the transients.

PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/diffusion/bin/python}"
PAIR="${PAIR:-0,1}"
TASKS="${TASKS:-100}"
SEED="${SEED:-271828}"
# MATH-500 (canvas 1024) is deferred: its ~1,300-token prefixes are untested
# against the 16 GB cards. Add it with BENCHMARKS=... once memory is checked.
BENCHMARKS="${BENCHMARKS:-gsm8k mbpp_plus humaneval_plus}"
METHODS=(${METHODS:-vgb:fwdonly:sl8x8 vgb:lam0.5:mom:sl8x8 vgb:lam1:mom:sl8x8})
BUDGET="${BUDGET:-512}"
ROW_CHUNK="${ROW_CHUNK:-16}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/../data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/vgb_trackA}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/vgb_trackA}"
TARGET_MODEL="${TARGET_MODEL:-$PROJECT_DIR/../models/SDAR-8B-Chat-b64-local}"
REFERENCE_MODEL="${REFERENCE_MODEL:-$PROJECT_DIR/../models/SDAR-1.7B-Chat-b64-local}"
SDAR_OVERLAY="${SDAR_OVERLAY:-$HARNESS_DIR/vendor_env/tf452}"
EVALPLUS_WORKERS="${EVALPLUS_WORKERS:-4}"
VOCAB_LIMIT=151665

canvas_for() {
    case "$1" in
        gsm8k) echo 256 ;;
        humaneval_plus) echo 512 ;;
        mbpp_plus) echo 256 ;;
        math_500) echo 1024 ;;
        *) echo "ERROR: no canvas for $1" >&2; exit 2 ;;
    esac
}

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
echo "== preconditions =="
for path in "$TARGET_MODEL" "$REFERENCE_MODEL" "$SDAR_OVERLAY"; do
    [ -d "$path" ] || { echo "ERROR: missing $path" >&2; exit 2; }
done
for bench in $BENCHMARKS; do
    [ -s "$DATA_ROOT/sample100/$bench.jsonl" ] || { echo "ERROR: missing $DATA_ROOT/sample100/$bench.jsonl" >&2; exit 2; }
    canvas_for "$bench" >/dev/null
done

# Refuse to share a card with anyone else's job (process owner, not memory).
IFS=, read -r -a CARDS <<< "$PAIR"
[ "${#CARDS[@]}" -eq 2 ] || { echo "ERROR: PAIR must name two cards" >&2; exit 2; }
foreign=""
for gpu in "${CARDS[@]}"; do
    uuid="$(nvidia-smi --id="$gpu" --query-gpu=uuid --format=csv,noheader)"
    while IFS=, read -r pid app_uuid; do
        [ -n "$pid" ] || continue
        [ "$(echo "$app_uuid" | xargs)" = "$uuid" ] || continue
        owner="$(ps -o user= -p "$(echo "$pid" | xargs)" 2>/dev/null | xargs || true)"
        [ "$owner" = "$USER" ] || foreign="$foreign GPU$gpu:pid$(echo "$pid" | xargs)($owner)"
    done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader)
done
if [ -n "$foreign" ]; then
    echo "ERROR: other users' jobs on the requested cards:$foreign" >&2
    exit 2
fi

cat >"$LOG_ROOT/RUNINFO.txt" <<INFO
exp06_vgb Track A: density-ratio VGB, root start, shortlist forward rule
started: $(date '+%Y-%m-%d %H:%M:%S %Z')   host: $(hostname)   cards: $PAIR
methods: ${METHODS[*]}
benchmarks: $BENCHMARKS (canvas $(for b in $BENCHMARKS; do printf '%s=%s ' "$b" "$(canvas_for "$b")"; done))
tasks: $TASKS   seed: $SEED   budget: $BUDGET moves/block   row_chunk: $ROW_CHUNK   vocab_limit: $VOCAB_LIMIT
target: $TARGET_MODEL (cuda:0+cuda:1)   reference: $REFERENCE_MODEL (cuda:1)
output: $OUTPUT_ROOT
INFO
cat "$LOG_ROOT/RUNINFO.txt"
[ "${DRY_RUN:-0}" = "1" ] && exit 0

for bench in $BENCHMARKS; do
    echo "== $bench ($(date '+%H:%M:%S')) =="
    EXPERIMENT_PHYSICAL_GPU_ID="${PAIR//,/+}" CUDA_VISIBLE_DEVICES="$PAIR" \
    PYTHONPATH="$SDAR_OVERLAY:$PYTHONPATH" \
    "$PYTHON_BIN" -u "$HARNESS_DIR/scripts/run_benchmark.py" \
        --dataset "$DATA_ROOT/sample100/$bench.jsonl" \
        --output-root "$OUTPUT_ROOT/$bench" \
        --continuation-length "$(canvas_for "$bench")" \
        --family sdar --block-length 64 --dtype bfloat16 --seed "$SEED" \
        --timing-repeats 1 --trace-limit 0 --require-lean-logits \
        --sample-size "$TASKS" --local-files-only --vocab-limit "$VOCAB_LIMIT" \
        --model "$TARGET_MODEL" --devices cuda:0 cuda:1 \
        --verifier-model "$REFERENCE_MODEL" --verifier-family sdar --verifier-device cuda:1 \
        --vgb-budget "$BUDGET" --vgb-row-chunk "$ROW_CHUNK" \
        --methods "${METHODS[@]}" \
        >"$LOG_ROOT/$bench.log" 2>&1
done

echo "== scoring ($(date '+%H:%M:%S')) =="
for bench in mbpp_plus humaneval_plus; do
    [ -d "$OUTPUT_ROOT/$bench" ] || continue
    for method_dir in "$OUTPUT_ROOT/$bench"/*/; do
        if [ -f "$method_dir/evalplus_samples.jsonl" ] && [ ! -f "$method_dir/records.scored.jsonl" ]; then
            "$PYTHON_BIN" "$HARNESS_DIR/scripts/score_evalplus.py" --benchmark "$bench" \
                --parallel "$EVALPLUS_WORKERS" "$method_dir"
        fi
    done
done
"$PYTHON_BIN" "$HARNESS_DIR/scripts/summarize_results.py" --output-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/comparison"
echo "Done ($(date '+%H:%M:%S')): $OUTPUT_ROOT/comparison/summary.csv"
