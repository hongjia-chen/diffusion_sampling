#!/usr/bin/env bash
# SDAR decoding on its own: the static (fixed:k) and dynamic (dynamic:θ)
# baselines the MI rule is compared against. One checkpoint per launch; its
# block length is read from the name and the harness refuses a mismatch.
#
#   fixed:k      SDAR's low_confidence_static: k most confident tokens per step
#                (B=4, k=1 is the published Table 2 setting). k <= block.
#   dynamic:θ    SDAR's low_confidence_dynamic: every token with p > θ, at least 1
#
# Protocol: SDAR's OpenCompass eval config (eval_sdar_hf.py), since 2026-10-03.
# GSM8K (1,319), MATH-500, HumanEval (164) and sanitized MBPP (257) with its
# prompts (../data/opencompass/) and graders (math_verify, MATHEvaluator v2,
# human-eval / MBPP base tests), a 4096-token canvas, fp16, the whole
# vocabulary. Ours on top: greedy T=0 (= its top_k=1), logp confidence,
# max_candidates = block, early stop at the first settled stop token (same
# output), 3 untimed warmups and one timed repeat per task. TASKS=N samples N
# per benchmark with seed 271828. ../harness/AUDIT.md lists what still differs.
#
#   bash launch_baselines.sh                                         # 1.7B-b64 on card 0
#   MODEL=../models/SDAR-1.7B-Chat-local bash launch_baselines.sh   # b4: fixed:1,2,4 + dynamic
#   MODEL=../models/SDAR-8B-Chat-b64-local CARDS=0,1 bash launch_baselines.sh
#   TASKS=200 bash launch_baselines.sh                               # screening subset
#   TASKS=250 bash launch_baselines.sh                               # dev sets: GSM8K/MATH 250 (../data/sample250), HE/MBPP all
#   TASKS=100 BENCHMARKS="gsm8k math_500 mbpp_plus humaneval_plus" bash launch_baselines.sh   # legacy exp04-06 samples and prompts
#   TASKS=3 OUTPUT_ROOT=outputs/smoke LOG_ROOT=logs/smoke bash launch_baselines.sh
#   DRY_RUN=1 bash launch_baselines.sh
#   WAIT_FOR_CARDS=1 bash launch_baselines.sh                        # wait out other users' jobs instead of refusing
#
# Contention: the runner waits before each timed run until no other process
# holds our cards, samples them while it is timed, and redoes a run that shared
# them (../harness/sdar_harness/contention.py; --contention-guard off disables it).
#
# A single card loads the model whole; two cards pipeline-split it (8B in fp16
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
TASKS="${TASKS:-all}"
SEED="${SEED:-271828}"
BENCHMARKS="${BENCHMARKS:-gsm8k math_500 humaneval mbpp_sanitized}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/../data}"
SDAR_OVERLAY="${SDAR_OVERLAY:-$HARNESS_DIR/vendor_env/tf452}"
EVALPLUS_WORKERS="${EVALPLUS_WORKERS:-4}"
# Verifier rows per call for adaptive:δ (self-verification); unused by fixed/dynamic.
MICROBATCH="${MICROBATCH:-4}"
# Opt-in attention options (ModelOptions; need the patched local model):
# ATTENTION_KERNEL=sdpa_nomask drops the all-True mask of block-local forwards
# so SDPA can use its flash kernel; PREFIX_BUFFER=1 copies the cached prefix
# into a per-block buffer once instead of every step (bit-identical).
ATTENTION_KERNEL="${ATTENTION_KERNEL:-sdpa_mask}"
ATTENTION_ARGS=(--attention-kernel "$ATTENTION_KERNEL")
[ "${PREFIX_BUFFER:-0}" = "1" ] && ATTENTION_ARGS+=(--prefix-buffer)
# SDAR's eval config: fp16 weights, logits over the whole vocabulary (151936
# = the lm_head width, so the limit is a no-op; only the mask token is barred).
DTYPE="${DTYPE:-float16}"
VOCAB_LIMIT=151936

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
    # SDAR uses 0.9 (README / JetEngine) and 0.95 (eval config); the lower end
    # reaches the tokens-per-step range the MI rule operates in.
    METHODS="$METHODS dynamic:0.99 dynamic:0.95 dynamic:0.90 dynamic:0.80 dynamic:0.70 dynamic:0.50 dynamic:0.30"
fi
read -r -a METHOD_LIST <<< "$METHODS"
RUN_NAME="${RUN_NAME:-${MODEL_NAME%-local}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/opencompass/$RUN_NAME}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/opencompass/$RUN_NAME}"

# SDAR's eval config generates up to 4096 tokens on every benchmark
# (gen_length=4096, eval_sdar_hf.py); so do we since 2026-10-03 (was
# 512/2048/512/768). Early stop makes unused canvas free, and blocks are
# causal, so answers that ended earlier are unchanged
# (../harness/scripts/extend_canvas.py).
canvas_for() {
    case "$1" in
        gsm8k|math_500|humaneval|mbpp_sanitized|humaneval_plus|mbpp_plus) echo 4096 ;;
        *) echo "ERROR: no canvas for $1" >&2; exit 2 ;;
    esac
}

# ../data/README.md: opencompass/ holds SDAR's four benchmarks with SDAR's
# prompts (the default); full/ the legacy exp04-06 files (mbpp_plus,
# humaneval_plus, and the old gsm8k/math_500 prompts); sample100/ the exact
# 100-task legacy samples exp04-06 ran on (TASKS=100, so runs stay paired);
# sample250/ the 250-task development subsets of GSM8K and MATH-500 (TASKS=250:
# those two, with all of HumanEval's 164 and MBPP's 257, about one precision).
data_for() {
    if [ "$TASKS" = "100" ]; then echo "$DATA_ROOT/sample100/$1.jsonl"
    elif [ "$TASKS" = "250" ] && [ -f "$DATA_ROOT/sample250/$1.jsonl" ]; then echo "$DATA_ROOT/sample250/$1.jsonl"
    elif [ -f "$DATA_ROOT/opencompass/$1.jsonl" ]; then echo "$DATA_ROOT/opencompass/$1.jsonl"
    else echo "$DATA_ROOT/full/$1.jsonl"; fi
}

SAMPLE_ARGS=()
case "$TASKS" in all|100|250) ;; *) SAMPLE_ARGS=(--sample-size "$TASKS") ;; esac

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
    [ -s "$(data_for "$bench")" ] || { echo "ERROR: missing $(data_for "$bench")" >&2; exit 2; }
    canvas_for "$bench" >/dev/null
done
# Refuse to share a card with anyone else's job (process owner, not memory).
# WAIT_FOR_CARDS=1 waits for them to clear instead, here and before each
# benchmark's model load, so the weights never go onto a card someone else holds.
foreign_jobs() {
    local found="" gpu uuid pid app_uuid owner
    for gpu in "${CARD_LIST[@]}"; do
        uuid="$(nvidia-smi --id="$gpu" --query-gpu=uuid --format=csv,noheader)"
        while IFS=, read -r pid app_uuid; do
            [ -n "$pid" ] || continue
            [ "$(echo "$app_uuid" | xargs)" = "$uuid" ] || continue
            owner="$(ps -o user= -p "$(echo "$pid" | xargs)" 2>/dev/null | xargs || true)"
            [ "$owner" = "$USER" ] || found="$found GPU$gpu:pid$(echo "$pid" | xargs)($owner)"
        done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader)
    done
    echo "$found"
}
wait_for_cards() {
    local found
    found="$(foreign_jobs)"
    [ -n "$found" ] || return 0
    echo "== waiting for the cards to clear ($(date '+%m-%d %H:%M:%S')):$found"
    while [ -n "$(foreign_jobs)" ]; do sleep 60; done
    echo "== cards clear ($(date '+%m-%d %H:%M:%S'))"
}
foreign="$(foreign_jobs)"
if [ -n "$foreign" ] && [ "${WAIT_FOR_CARDS:-0}" != "1" ]; then
    echo "ERROR: other users' jobs on the requested cards:$foreign" >&2
    exit 2
fi

cat >"$LOG_ROOT/RUNINFO.txt" <<INFO
MI_decoding SDAR baselines: $MODEL_NAME (trained block $BLOCK, decoded at $BLOCK)
started: $(date '+%Y-%m-%d %H:%M:%S %Z')   host: $(hostname)   cards: $CARDS
methods: ${METHOD_LIST[*]}
benchmarks: $BENCHMARKS (canvas $(for b in $BENCHMARKS; do printf '%s=%s ' "$b" "$(canvas_for "$b")"; done))
tasks: $TASKS per benchmark   seed: $SEED   block: $BLOCK   vocab_limit: $VOCAB_LIMIT   warmups: 3   attention: $ATTENTION_KERNEL   prefix buffer: ${PREFIX_BUFFER:-0}   verifier rows: $MICROBATCH
recipe: OpenCompass (SDAR eval config) prompts and graders, $DTYPE, greedy T=0, logp confidence, max_candidates=$BLOCK, early stop on
data: $(dirname -- "$(data_for gsm8k)")
output: $OUTPUT_ROOT
INFO
cat "$LOG_ROOT/RUNINFO.txt"
[ "${DRY_RUN:-0}" = "1" ] && exit 0

# SKIP_GENERATION=1 / SKIP_SCORING=1 split a run into cells that share one
# OUTPUT_ROOT (one benchmark x method per card, scored once at the end; see
# run_cells.sh). Scoring while another cell still writes would grade a partial
# records.jsonl.
[ "${SKIP_GENERATION:-0}" = "1" ] && BENCH_TO_RUN="" || BENCH_TO_RUN="$BENCHMARKS"
for bench in $BENCH_TO_RUN; do
    [ "${WAIT_FOR_CARDS:-0}" = "1" ] && wait_for_cards
    echo "== $bench ($(date '+%H:%M:%S')) =="
    EXPERIMENT_PHYSICAL_GPU_ID="${CARDS//,/+}" CUDA_VISIBLE_DEVICES="$CARDS" \
    PYTHONPATH="$SDAR_OVERLAY:$PYTHONPATH" \
    "$PYTHON_BIN" -u "$HARNESS_DIR/scripts/run_benchmark.py" \
        --dataset "$(data_for "$bench")" \
        --output-root "$OUTPUT_ROOT/$bench" \
        --continuation-length "$(canvas_for "$bench")" \
        --verifier-microbatch-size "$MICROBATCH" \
        --model "$MODEL" --family sdar --block-length "$BLOCK" "${DEVICE_ARGS[@]}" \
        --dtype "$DTYPE" --seed "$SEED" "${ATTENTION_ARGS[@]}" \
        --timing-repeats 1 --warmup-runs 3 --trace-limit "${TRACE_LIMIT:-0}" --require-lean-logits \
        --profile-stages "${PROFILE_STAGES:-0}" \
        --max-candidates "$BLOCK" "${SAMPLE_ARGS[@]}" --local-files-only \
        --vocab-limit "$VOCAB_LIMIT" \
        --fixed-temperature 0.0 --fixed-confidence logp \
        --dynamic-temperature 0.0 --dynamic-confidence logp \
        --contention-guard "${CONTENTION_GUARD:-wait}" \
        --methods "${METHOD_LIST[@]}" \
        >"$LOG_ROOT/$bench.log" 2>&1
done
[ "${SKIP_SCORING:-0}" = "1" ] && { echo "Done ($(date '+%H:%M:%S')), scoring skipped"; exit 0; }

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
# SDAR's own grading (OpenCompass, base tests): humaneval and mbpp_sanitized's
# score; for legacy humaneval_plus, the headline after evalplus adds base/plus.
for bench in humaneval mbpp_sanitized humaneval_plus; do
    [ -d "$OUTPUT_ROOT/$bench" ] || continue
    for method_dir in "$OUTPUT_ROOT/$bench"/*/; do
        [ -f "$method_dir/records.jsonl" ] || continue
        "$PYTHON_BIN" "$HARNESS_DIR/scripts/score_opencompass.py" --benchmark "$bench" \
            --parallel "$EVALPLUS_WORKERS" "$method_dir"
    done
done
"$PYTHON_BIN" "$HARNESS_DIR/scripts/summarize_results.py" --output-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/comparison"
echo "Done ($(date '+%H:%M:%S')): $OUTPUT_ROOT/comparison/summary.csv"
