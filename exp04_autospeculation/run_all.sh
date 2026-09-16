#!/usr/bin/env bash
# Autospeculation experiment: every method on every benchmark, one balanced pass.
#
# Methods are distributed round-robin across however many GPUs are listed; each
# worker then runs all four benchmarks in sequence. Once a calibration run has
# produced measured per-method costs, prefer scripts/pack_work.py --costs for a
# longest-processing-time packing instead.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
# Generation is bit-reproducible only with this set. torch.use_deterministic_
# algorithms is never called, so nothing in the code enforces it.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "${HOME}/miniconda3/envs/diffusion/bin/python" ]; then
        PYTHON_BIN="${HOME}/miniconda3/envs/diffusion/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

# No default checkpoint. This experiment is meant to run a smaller model than
# the 7B of exp03_dream_adaptive_sampling_ablation, and which one is not settled; a
# default here is how the wrong 20-hour run gets launched.
MODEL_ID="${MODEL_ID:-}"
MODEL_FAMILY="${MODEL_FAMILY:-dream}"
if [ -z "$MODEL_ID" ]; then
    echo "ERROR: set MODEL_ID (and MODEL_FAMILY if the checkpoint is not Dream-like)." >&2
    echo "       MODEL_FAMILY selects the logits-alignment and attention-mask" >&2
    echo "       conventions in autospec/model.py. A wrong family still generates" >&2
    echo "       fluent text, so verify it against the checkpoint's own reference" >&2
    echo "       generation code before spending GPU hours." >&2
    exit 2
fi

TASKS="${TASKS:-100}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/v1}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/v1}"
DTYPE="${DTYPE:-bfloat16}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
EVALPLUS_WORKERS="${EVALPLUS_WORKERS:-4}"
SEED="${SEED:-271828}"
MAX_CANDIDATES="${MAX_CANDIDATES:-64}"

# Methods. Fixed-k and adaptive are exactly the schedulers of the previous
# experiment -- same code, same confidence ordering, no permutation. Only
# autospec is new, and it is absent until Algorithm 3 exists; adding it now
# just fails every worker at the first task.
METHODS="${METHODS:-fixed:1 fixed:2 fixed:4 fixed:8 adaptive:0.10 adaptive:0.30 adaptive:0.50 adaptive:1.00}"

# One decoding recipe for the fixed and adaptive arms. The previous experiment
# gave them different recipes and could not then attribute the difference
# between them to the scheduler; outputs/v3_matched exists only to undo that.
# These default to that matched recipe -- greedy, logp -- so the baselines here
# are the same schedulers under the same settings as before.
#
# autospec does NOT read these. Algorithm 3 forces T=1.0 with no top-p in
# AutospecConfig (DESIGN.md D1), because that is the only setting at which its
# target is nameable and the round bound applies.
#
# OPEN (DESIGN.md D7): that leaves autospec at T=1 and the baselines at T=0, so
# an autospec-vs-fixed quality gap mixes the sampler with the decoding recipe.
# Resolving it means a second baseline pass at T=1.0 into its own OUTPUT_ROOT.
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-}"
CONFIDENCE="${CONFIDENCE:-logp}"

# Verifier microbatch per benchmark. What bounds memory is prompt+canvas, not
# canvas alone, so these need re-measuring for the chosen model and prompts:
#   CUDA_VISIBLE_DEVICES=0 python3 scripts/preflight.py --model "$MODEL_ID" \
#       --dataset data/gsm8k/test.jsonl --continuation-length 256
MICROBATCH_GSM8K="${MICROBATCH_GSM8K:-8}"
MICROBATCH_MATH_500="${MICROBATCH_MATH_500:-4}"
MICROBATCH_MBPP_PLUS="${MICROBATCH_MBPP_PLUS:-4}"
MICROBATCH_HUMANEVAL_PLUS="${MICROBATCH_HUMANEVAL_PLUS:-4}"

# Canvas length per benchmark, fixed across every example and method within a
# benchmark.
#
# D5 sized these so Algorithm 3's root block -- the whole canvas -- fit in one
# physical verifier batch. **That reasoning applies to a full-canvas dLLM like
# Dream and NOT to SDAR (D8):** under block diffusion the root block is the
# SDAR block (64), not the canvas, so canvas length no longer constrains the
# recursion at all. It is now chosen on two other grounds only:
#
#   1. match the prior Dream run wherever a benchmark existed there, so its
#      latency numbers port over as a reference frontier (GSM8K 256,
#      MBPP+ 512);
#   2. fit the answer. Measured reference lengths say 2.1% of GSM8K answers
#      exceed 256, and for MATH-500 28.6% exceed 256, 13.6% exceed 384 and
#      8.2% exceed 512 -- so MATH-500 gets 512. `maximum_length_rate` in the
#      summary is the check; if it is ~0 on a pilot, 384 would save ~33%.
LENGTH_GSM8K="${LENGTH_GSM8K:-256}"
LENGTH_MATH_500="${LENGTH_MATH_500:-512}"
LENGTH_MBPP_PLUS="${LENGTH_MBPP_PLUS:-512}"
LENGTH_HUMANEVAL_PLUS="${LENGTH_HUMANEVAL_PLUS:-512}"

if ! [ "$TASKS" -ge 1 ] 2>/dev/null; then
    echo "ERROR: TASKS must be a positive integer, got '$TASKS'" >&2
    exit 2
fi
read -r -a GPUS <<< "$GPU_IDS"
read -r -a METHOD_LIST <<< "$METHODS"
if [ "${#GPUS[@]}" -lt 1 ]; then
    echo "ERROR: GPU_IDS is empty." >&2
    exit 2
fi
for dataset in mbpp_plus humaneval_plus gsm8k math_500; do
    if [ ! -s "$DATA_ROOT/$dataset/test.jsonl" ]; then
        echo "ERROR: missing $DATA_ROOT/$dataset/test.jsonl -- run scripts/prepare_all.sh" >&2
        exit 2
    fi
done
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

common=(
    --model "$MODEL_ID"
    --family "$MODEL_FAMILY"
    --device cuda:0
    --dtype "$DTYPE"
    --seed "$SEED"
    --timing-repeats 1
    --trace-limit 0
    --require-lean-logits
    --max-candidates "$MAX_CANDIDATES"
    --sample-size "$TASKS"
)
for kind in fixed adaptive autospec; do
    common+=("--${kind}-temperature" "$TEMPERATURE" "--${kind}-confidence" "$CONFIDENCE")
    if [ -n "$TOP_P" ]; then
        common+=("--${kind}-top-p" "$TOP_P")
    fi
done
if [ "$LOCAL_FILES_ONLY" = "1" ]; then
    common+=(--local-files-only)
fi

run_method_group() {
    local physical_gpu="$1"
    shift
    local methods=("$@")
    local benchmark length microbatch
    while read -r benchmark length microbatch; do
        [ -z "$benchmark" ] && continue
        echo "[GPU $physical_gpu] $benchmark (canvas $length, microbatch $microbatch): ${methods[*]}"
        EXPERIMENT_PHYSICAL_GPU_ID="$physical_gpu" \
        CUDA_VISIBLE_DEVICES="$physical_gpu" \
        "$PYTHON_BIN" -u scripts/run_benchmark.py \
            --dataset "$DATA_ROOT/$benchmark/test.jsonl" \
            --output-root "$OUTPUT_ROOT/$benchmark" \
            --continuation-length "$length" \
            --verifier-microbatch-size "$microbatch" \
            --methods "${methods[@]}" \
            "${common[@]}"
    done <<DATASETS
gsm8k $LENGTH_GSM8K $MICROBATCH_GSM8K
math_500 $LENGTH_MATH_500 $MICROBATCH_MATH_500
humaneval_plus $LENGTH_HUMANEVAL_PLUS $MICROBATCH_HUMANEVAL_PLUS
mbpp_plus $LENGTH_MBPP_PLUS $MICROBATCH_MBPP_PLUS
DATASETS
}

rm -f "$LOG_ROOT"/*.pid
pids=()
for index in "${!GPUS[@]}"; do
    assigned=()
    for method_index in "${!METHOD_LIST[@]}"; do
        if [ $(( method_index % ${#GPUS[@]} )) -eq "$index" ]; then
            assigned+=("${METHOD_LIST[$method_index]}")
        fi
    done
    if [ "${#assigned[@]}" -eq 0 ]; then
        continue
    fi
    run_method_group "${GPUS[$index]}" "${assigned[@]}" \
        >"$LOG_ROOT/worker_${GPUS[$index]}.log" 2>&1 &
    pids+=("$!")
    printf '%s\n' "$!" >"$LOG_ROOT/worker_${GPUS[$index]}.pid"
done
printf '%s\n' "$$" >"$LOG_ROOT/launcher.pid"

stop_workers() {
    trap - INT TERM
    for pid in "${pids[@]}"; do
        pkill -TERM -P "$pid" 2>/dev/null || true
        kill "$pid" 2>/dev/null || true
    done
    exit 130
}
trap stop_workers INT TERM

cat >"$LOG_ROOT/RUNINFO.txt" <<INFO
Autospeculation experiment
started: $(date '+%Y-%m-%d %H:%M:%S %Z')
host: $(hostname)
model: $MODEL_ID (family $MODEL_FAMILY, $DTYPE)
tasks per benchmark: $TASKS
methods: $METHODS
max candidates: $MAX_CANDIDATES
canvas: gsm8k=$LENGTH_GSM8K math_500=$LENGTH_MATH_500 humaneval_plus=$LENGTH_HUMANEVAL_PLUS mbpp_plus=$LENGTH_MBPP_PLUS
microbatch: gsm8k=$MICROBATCH_GSM8K math_500=$MICROBATCH_MATH_500 humaneval_plus=$MICROBATCH_HUMANEVAL_PLUS mbpp_plus=$MICROBATCH_MBPP_PLUS
sampling (all arms): T=$TEMPERATURE top_p=${TOP_P:-none} confidence=$CONFIDENCE
seed: $SEED, drawn per (seed, method, task)
timing repeats: 1
decision traces: off
GPUs: $GPU_IDS
output: $OUTPUT_ROOT
INFO
cat "$LOG_ROOT/RUNINFO.txt"

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
trap - INT TERM
rm -f "$LOG_ROOT"/*.pid
if [ "$failed" -ne 0 ]; then
    echo "ERROR: a GPU worker failed; inspect $LOG_ROOT" >&2
    exit 1
fi

# EvalPlus is the source of pass@1 for both code benchmarks. Do not recompute a
# finished score on resume; records.scored.jsonl is the input the summary uses.
for benchmark in mbpp_plus humaneval_plus; do
    for method_dir in "$OUTPUT_ROOT/$benchmark"/*; do
        if [ -f "$method_dir/evalplus_samples.jsonl" ] && \
           [ ! -f "$method_dir/records.scored.jsonl" ]; then
            "$PYTHON_BIN" scripts/score_evalplus.py --benchmark "$benchmark" \
                --parallel "$EVALPLUS_WORKERS" "$method_dir"
        fi
    done
done

"$PYTHON_BIN" scripts/summarize_results.py \
    --output-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/comparison"

echo "Complete: $OUTPUT_ROOT/comparison/summary.json"
