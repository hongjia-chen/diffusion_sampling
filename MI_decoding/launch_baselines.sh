#!/usr/bin/env bash
# SDAR decoding on its own: the static (fixed:k) and dynamic (dynamic:θ)
# baselines the MI rule is compared against. One checkpoint per launch; its
# block length is read from the name and the harness refuses a mismatch.
#
#   fixed:k      SDAR's low_confidence_static: k most confident tokens per step
#                (B=4, k=1 is the published Table 2 setting). k <= block.
#   dynamic:θ    SDAR's low_confidence_dynamic: every token with p > θ, at least 1
#
# Everything else is the exp05/exp06 recipe: greedy T=0, logp confidence,
# max_candidates = block, 100 tasks per benchmark sampled with seed 271828,
# canvases GSM8K 256 / HumanEval+ 512 / MBPP+ 256 / MATH-500 1024, early stop at
# the first settled stop token, logits over IDs < 151665, bf16, one timed repeat.
# It is NOT SDAR's OpenCompass protocol (../harness/AUDIT.md).
#
#   bash launch_baselines.sh                                         # 1.7B-b64 on card 0
#   MODEL=../models/SDAR-1.7B-Chat-local bash launch_baselines.sh   # b4: fixed:1,2,4 + dynamic
#   MODEL=../models/SDAR-8B-Chat-b64-local CARDS=0,1 bash launch_baselines.sh
#   TASKS=3 OUTPUT_ROOT=outputs/smoke LOG_ROOT=logs/smoke bash launch_baselines.sh
#   DRY_RUN=1 bash launch_baselines.sh
#
# A single card loads the model whole; two cards pipeline-split it (8B in bf16
# does not fit one 16 GB A4000). Resumable: rerunning skips finished tasks.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
# The shared harness (package, runner and scoring scripts) lives one level up.
HARNESS_DIR="$(cd -- "$PROJECT_DIR/../harness" && pwd)"
export PYTHONPATH="$HARNESS_DIR:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/diffusion/bin/python}"
MODEL="${MODEL:-$PROJECT_DIR/../models/SDAR-1.7B-Chat-b64-local}"
CARDS="${CARDS:-0}"
TASKS="${TASKS:-100}"
SEED="${SEED:-271828}"
BENCHMARKS="${BENCHMARKS:-gsm8k mbpp_plus humaneval_plus}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/../data}"
SDAR_OVERLAY="${SDAR_OVERLAY:-$HARNESS_DIR/vendor_env/tf452}"
EVALPLUS_WORKERS="${EVALPLUS_WORKERS:-4}"
MICROBATCH=4
VOCAB_LIMIT=151665

[ -d "$MODEL" ] || { echo "ERROR: no checkpoint at $MODEL (build it with $HARNESS_DIR/scripts/prepare_sdar_local.py)" >&2; exit 2; }
MODEL_NAME="$(basename -- "$MODEL")"
# The block the checkpoint was trained with; decoding at any other is out of distribution.
BLOCK="$(PYTHONPATH="$HARNESS_DIR" "$PYTHON_BIN" -c \
    "import sys; from sdar_harness.model import trained_block_length as t; b = t(sys.argv[1]); print(b or '')" "$MODEL")"
[ -n "$BLOCK" ] || { echo "ERROR: cannot read a block length from the name $MODEL_NAME" >&2; exit 2; }
# k cannot exceed the block: positions past it are in the next block, which
# only ever sees this one once it is fully committed.
if [ -z "${METHODS:-}" ]; then
    METHODS=""
    for k in 1 2 4 8; do [ "$k" -le "$BLOCK" ] && METHODS="$METHODS fixed:$k"; done
    METHODS="$METHODS dynamic:0.95 dynamic:0.90 dynamic:0.80 dynamic:0.70"
fi
read -r -a METHOD_LIST <<< "$METHODS"
RUN_NAME="${RUN_NAME:-${MODEL_NAME%-local}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/baselines/$RUN_NAME}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/baselines/$RUN_NAME}"

canvas_for() {
    case "$1" in
        gsm8k) echo 256 ;;
        humaneval_plus) echo 512 ;;
        mbpp_plus) echo 256 ;;
        math_500) echo 1024 ;;
        *) echo "ERROR: no canvas for $1" >&2; exit 2 ;;
    esac
}

IFS=, read -r -a CARD_LIST <<< "$CARDS"
case "${#CARD_LIST[@]}" in
    1) DEVICE_ARGS=(--device cuda:0) ;;
    2) DEVICE_ARGS=(--devices cuda:0 cuda:1) ;;
    *) echo "ERROR: CARDS must name one card or a pair" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
echo "== preconditions =="
[ -d "$SDAR_OVERLAY" ] || { echo "ERROR: missing $SDAR_OVERLAY" >&2; exit 2; }
for bench in $BENCHMARKS; do
    [ -s "$DATA_ROOT/$bench/test.jsonl" ] || { echo "ERROR: missing $DATA_ROOT/$bench/test.jsonl" >&2; exit 2; }
    canvas_for "$bench" >/dev/null
done
# Refuse to share a card with anyone else's job (process owner, not memory).
foreign=""
for gpu in "${CARD_LIST[@]}"; do
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
MI_decoding SDAR baselines: $MODEL_NAME (trained block $BLOCK, decoded at $BLOCK)
started: $(date '+%Y-%m-%d %H:%M:%S %Z')   host: $(hostname)   cards: $CARDS
methods: ${METHOD_LIST[*]}
benchmarks: $BENCHMARKS (canvas $(for b in $BENCHMARKS; do printf '%s=%s ' "$b" "$(canvas_for "$b")"; done))
tasks: $TASKS   seed: $SEED   block: $BLOCK   vocab_limit: $VOCAB_LIMIT
recipe: greedy T=0, logp confidence, max_candidates=$BLOCK, early stop on
output: $OUTPUT_ROOT
INFO
cat "$LOG_ROOT/RUNINFO.txt"
[ "${DRY_RUN:-0}" = "1" ] && exit 0

for bench in $BENCHMARKS; do
    echo "== $bench ($(date '+%H:%M:%S')) =="
    EXPERIMENT_PHYSICAL_GPU_ID="${CARDS//,/+}" CUDA_VISIBLE_DEVICES="$CARDS" \
    PYTHONPATH="$SDAR_OVERLAY:$PYTHONPATH" \
    "$PYTHON_BIN" -u "$HARNESS_DIR/scripts/run_benchmark.py" \
        --dataset "$DATA_ROOT/$bench/test.jsonl" \
        --output-root "$OUTPUT_ROOT/$bench" \
        --continuation-length "$(canvas_for "$bench")" \
        --verifier-microbatch-size "$MICROBATCH" \
        --model "$MODEL" --family sdar --block-length "$BLOCK" "${DEVICE_ARGS[@]}" \
        --dtype bfloat16 --seed "$SEED" \
        --timing-repeats 1 --trace-limit 0 --require-lean-logits \
        --max-candidates "$BLOCK" --sample-size "$TASKS" --local-files-only \
        --vocab-limit "$VOCAB_LIMIT" \
        --fixed-temperature 0.0 --fixed-confidence logp \
        --dynamic-temperature 0.0 --dynamic-confidence logp \
        --methods "${METHOD_LIST[@]}" \
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
