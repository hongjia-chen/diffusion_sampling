#!/usr/bin/env bash
# exp05's cross-model adaptive delta with an aligned verifier (DESIGN.md §10):
# SDAR-1.7B proposes, SDAR-8B verifies and remasks (ROLES=8b_proposes swaps
# them, §10a). Everything else is exp05's
# `xadaptive-remask` recipe: greedy logp proposals, Dream-only-style ratio with
# no veto, groups of 4 verifier rows, one remask per block from the bottom 4,
# same 100 task IDs and canvases. The no-remask arms (`xadaptive`) at the same
# deltas are the paired counterfactual.
#
#   bash launch_cross8b.sh                    # refuses if another user's job is on the cards
#   TASKS=5 OUTPUT_ROOT=outputs/cross8b_pilot LOG_ROOT=logs/cross8b_pilot bash launch_cross8b.sh
#   ROLES=8b_proposes bash launch_cross8b.sh   # ablation: 8B proposes, 1.7B verifies and remasks
#   PAIR=2,3 bash launch_cross8b.sh
#   DRY_RUN=1 bash launch_cross8b.sh
#
# One replica, one card pair: 8B is pipeline-split over the pair and 1.7B
# shares the second card, whichever role each plays. Resumable: rerunning
# skips finished tasks.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
# The shared harness (package, runner and scoring scripts) lives one level up.
HARNESS_DIR="$(cd -- "$PROJECT_DIR/../harness" && pwd)"
export PYTHONPATH="$HARNESS_DIR:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/diffusion/bin/python}"
PAIR="${PAIR:-0,1}"
TASKS="${TASKS:-100}"
SEED="${SEED:-271828}"
BENCHMARKS="${BENCHMARKS:-gsm8k mbpp_plus humaneval_plus}"
DELTAS="${DELTAS:-0.10 1.00}"
# Remask arms first: they are the experiment; the counterfactual follows.
VARIANTS="${VARIANTS:-xadaptive-remask xadaptive}"
TRACE_LIMIT="${TRACE_LIMIT:-5}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/../data}"
SMALL_MODEL="$PROJECT_DIR/../models/SDAR-1.7B-Chat-b64-local"
BIG_MODEL="$PROJECT_DIR/../models/SDAR-8B-Chat-b64-local"
# Which model proposes. The 8B always takes both cards; the 1.7B the second.
ROLES="${ROLES:-1p7b_proposes}"
case "$ROLES" in
    1p7b_proposes)
        PROPOSER_MODEL="$SMALL_MODEL"; PROPOSER_ARGS=(--device cuda:1)
        VERIFIER_MODEL="$BIG_MODEL"; VERIFIER_ARGS=(--verifier-devices cuda:0 cuda:1)
        DEFAULT_NAME=cross8b ;;
    8b_proposes)
        PROPOSER_MODEL="$BIG_MODEL"; PROPOSER_ARGS=(--devices cuda:0 cuda:1)
        VERIFIER_MODEL="$SMALL_MODEL"; VERIFIER_ARGS=(--verifier-device cuda:1)
        DEFAULT_NAME=cross8b_swap ;;
    *) echo "ERROR: ROLES must be 1p7b_proposes or 8b_proposes" >&2; exit 2 ;;
esac
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/$DEFAULT_NAME}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/$DEFAULT_NAME}"
SDAR_OVERLAY="${SDAR_OVERLAY:-$HARNESS_DIR/vendor_env/tf452}"
EVALPLUS_WORKERS="${EVALPLUS_WORKERS:-4}"
BLOCK=64
GROUP_ROWS=4
VOCAB_LIMIT=151665

canvas_for() {
    case "$1" in
        gsm8k) echo 256 ;;
        humaneval_plus) echo 512 ;;
        mbpp_plus) echo 256 ;;
        *) echo "ERROR: no canvas for $1" >&2; exit 2 ;;
    esac
}

METHODS=()
for variant in $VARIANTS; do
    for delta in $DELTAS; do METHODS+=("$variant:$delta"); done
done

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
echo "== preconditions =="
for path in "$PROPOSER_MODEL" "$VERIFIER_MODEL" "$SDAR_OVERLAY"; do
    [ -d "$path" ] || { echo "ERROR: missing $path" >&2; exit 2; }
done
for bench in $BENCHMARKS; do
    [ -s "$DATA_ROOT/$bench/test.jsonl" ] || { echo "ERROR: missing $DATA_ROOT/$bench/test.jsonl" >&2; exit 2; }
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
exp06 cross8b: exp05 xadaptive recipe, roles $ROLES
started: $(date '+%Y-%m-%d %H:%M:%S %Z')   host: $(hostname)   cards: $PAIR
methods: ${METHODS[*]}
benchmarks: $BENCHMARKS (canvas $(for b in $BENCHMARKS; do printf '%s=%s ' "$b" "$(canvas_for "$b")"; done))
tasks: $TASKS   seed: $SEED   block: $BLOCK   verifier group rows: $GROUP_ROWS   vocab_limit: $VOCAB_LIMIT
proposer: $PROPOSER_MODEL (${PROPOSER_ARGS[*]}, greedy, logp)   verifier: $VERIFIER_MODEL (${VERIFIER_ARGS[*]}, veto off)
output: $OUTPUT_ROOT
INFO
cat "$LOG_ROOT/RUNINFO.txt"
[ "${DRY_RUN:-0}" = "1" ] && exit 0

for bench in $BENCHMARKS; do
    echo "== $bench ($(date '+%H:%M:%S')) =="
    EXPERIMENT_PHYSICAL_GPU_ID="${PAIR//,/+}" CUDA_VISIBLE_DEVICES="$PAIR" \
    PYTHONPATH="$SDAR_OVERLAY:$PYTHONPATH" \
    "$PYTHON_BIN" -u "$HARNESS_DIR/scripts/run_benchmark.py" \
        --dataset "$DATA_ROOT/$bench/test.jsonl" \
        --output-root "$OUTPUT_ROOT/$bench" \
        --continuation-length "$(canvas_for "$bench")" \
        --verifier-microbatch-size "$GROUP_ROWS" \
        --model "$PROPOSER_MODEL" --family sdar --block-length "$BLOCK" "${PROPOSER_ARGS[@]}" \
        --verifier-model "$VERIFIER_MODEL" --verifier-family sdar "${VERIFIER_ARGS[@]}" \
        --dtype bfloat16 --seed "$SEED" \
        --timing-repeats 1 --trace-limit "$TRACE_LIMIT" --require-lean-logits \
        --max-candidates "$BLOCK" --sample-size "$TASKS" --local-files-only \
        --vocab-limit "$VOCAB_LIMIT" \
        --adaptive-temperature 0.0 --adaptive-confidence logp \
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
