#!/usr/bin/env bash
# Core three-GPU scheduler comparison. The method grid is intentionally fixed;
# TASKS, paths, and physical GPU IDs are the only normal run-time choices.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "${HOME}/miniconda3/envs/diffusion/bin/python" ]; then
        PYTHON_BIN="${HOME}/miniconda3/envs/diffusion/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

TASKS="${TASKS:-100}"
GPU_IDS="${GPU_IDS:-0 1 2}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/core}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/core}"
MODEL_ID="${MODEL_ID:-Dream-org/Dream-v0-Instruct-7B}"
DTYPE="${DTYPE:-bfloat16}"
# Verifier batching is bounded by GPU memory, measured on this card:
# S=350 (256-token canvas) fits batch 8 with 0.84 GiB spare, OOMs at 16.
# S=600 (512-token canvas) fits batch 8 with only 0.47 GiB spare, so MBPP+
# gets 4. The cap is bounded by time, not memory: cap 32 with microbatch 8 is
# 4 sequential passes per step, the same cost profile as the previous
# cap 8 / microbatch 2 run.
MAX_CANDIDATES="${MAX_CANDIDATES:-64}"
# Keyed per benchmark, because what bounds memory is prompt+canvas, not canvas.
# Measured p95/max prompt lengths: gsm8k 126/157, schema_json 381/383 (it
# embeds the whole JSON Schema), mbpp_plus 117/192. So the sequence lengths are
# gsm8k ~413 max, schema_json ~639, mbpp_plus ~704. Only gsm8k has room for 8;
# schema_json OOMed at 8.
MICROBATCH_GSM8K="${MICROBATCH_GSM8K:-8}"
MICROBATCH_SCHEMA_JSON="${MICROBATCH_SCHEMA_JSON:-4}"
MICROBATCH_MBPP_PLUS="${MICROBATCH_MBPP_PLUS:-4}"
# Sampling is per method kind. The fixed arm follows the Dream/ADJUST instruct
# recipe; the adaptive arm stays on greedy argmax for now and is ablated
# separately. Deliberately non-comparable across arms until that ablation runs.
FIXED_TEMPERATURE="${FIXED_TEMPERATURE:-0.1}"
FIXED_TOP_P="${FIXED_TOP_P:-0.9}"
FIXED_CONFIDENCE="${FIXED_CONFIDENCE:-entropy}"
ADAPTIVE_TEMPERATURE="${ADAPTIVE_TEMPERATURE:-0.0}"
ADAPTIVE_CONFIDENCE="${ADAPTIVE_CONFIDENCE:-logp}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
EVALPLUS_WORKERS="${EVALPLUS_WORKERS:-4}"

if ! [ "$TASKS" -ge 1 ] 2>/dev/null; then
    echo "ERROR: TASKS must be a positive integer, got '$TASKS'" >&2
    exit 2
fi
read -r -a GPUS <<< "$GPU_IDS"
if [ "${#GPUS[@]}" -ne 3 ]; then
    echo "ERROR: GPU_IDS must contain exactly three GPU IDs." >&2
    exit 2
fi
for dataset in mbpp_plus gsm8k schema_json; do
    if [ ! -s "$DATA_ROOT/$dataset/test.jsonl" ]; then
        echo "ERROR: missing $DATA_ROOT/$dataset/test.jsonl" >&2
        exit 2
    fi
done
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

common=(
    --model "$MODEL_ID"
    --device cuda:0
    --dtype "$DTYPE"
    --timing-repeats 1
    --trace-limit 0
    --require-lean-logits
    --max-candidates "$MAX_CANDIDATES"
    --fixed-temperature "$FIXED_TEMPERATURE"
    --fixed-top-p "$FIXED_TOP_P"
    --fixed-confidence "$FIXED_CONFIDENCE"
    --adaptive-temperature "$ADAPTIVE_TEMPERATURE"
    --adaptive-confidence "$ADAPTIVE_CONFIDENCE"
    --sample-size "$TASKS"
)
if [ "$LOCAL_FILES_ONLY" = "1" ]; then
    common+=(--local-files-only)
fi

run_method_group() {
    local physical_gpu="$1"
    shift
    local methods=("$@")
    local benchmark length microbatch
    while read -r benchmark length microbatch; do
        echo "[GPU $physical_gpu] $benchmark: ${methods[*]} (microbatch $microbatch)"
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
mbpp_plus 512 $MICROBATCH_MBPP_PLUS
gsm8k 256 $MICROBATCH_GSM8K
schema_json 256 $MICROBATCH_SCHEMA_JSON
DATASETS
}

workers=(low_tolerance_and_fixed2 middle_tolerance_and_fixed_edges high_tolerance_and_fixed4)
rm -f "$LOG_ROOT"/*.pid

run_method_group "${GPUS[0]}" adaptive:0.15 fixed:2 >"$LOG_ROOT/${workers[0]}.log" 2>&1 &
pids=("$!")
run_method_group "${GPUS[1]}" adaptive:0.30 fixed:1 fixed:8 >"$LOG_ROOT/${workers[1]}.log" 2>&1 &
pids+=("$!")
run_method_group "${GPUS[2]}" adaptive:0.50 adaptive:0.75 fixed:4 >"$LOG_ROOT/${workers[2]}.log" 2>&1 &
pids+=("$!")

for index in 0 1 2; do
    printf '%s\n' "${pids[$index]}" >"$LOG_ROOT/${workers[$index]}.pid"
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
Dream core scheduler comparison
started: $(date '+%Y-%m-%d %H:%M:%S %Z')
host: $(hostname)
tasks per benchmark: $TASKS
methods: fixed:1 fixed:2 fixed:4 fixed:8 adaptive:0.15 adaptive:0.30 adaptive:0.50 adaptive:0.75
timing repeats: 1
decision traces: off
max candidates: $MAX_CANDIDATES
microbatch: gsm8k=$MICROBATCH_GSM8K schema_json=$MICROBATCH_SCHEMA_JSON mbpp_plus=$MICROBATCH_MBPP_PLUS
fixed sampling: T=$FIXED_TEMPERATURE top_p=$FIXED_TOP_P conf=$FIXED_CONFIDENCE
adaptive sampling: T=$ADAPTIVE_TEMPERATURE conf=$ADAPTIVE_CONFIDENCE
seed: fixed per (seed, method, task)
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

# EvalPlus is the source of MBPP+ pass@1. Do not recompute a finished score on
# resume; records.scored.jsonl is the input used by the summary.
for method_dir in "$OUTPUT_ROOT/mbpp_plus"/*; do
    if [ -f "$method_dir/evalplus_samples.jsonl" ] && \
       [ ! -f "$method_dir/records.scored.jsonl" ]; then
        "$PYTHON_BIN" scripts/score_mbpp_plus.py \
            --parallel "$EVALPLUS_WORKERS" "$method_dir"
    fi
done

"$PYTHON_BIN" scripts/summarize_results.py \
    --output-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/comparison"

echo "Complete: $OUTPUT_ROOT/comparison/summary.json"
