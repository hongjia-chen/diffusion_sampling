#!/usr/bin/env bash
# Matched-sampling fixed arm: re-run fixed:1/2/4/8 under the adaptive arm's
# decoding recipe so the two arms can be compared.
#
# In outputs/v2 the fixed arm ran the Dream/ADJUST instruct recipe
# (alg=entropy, T=0.1, top_p=0.9) while the adaptive arm ran greedy argmax with
# max-log-prob confidence. Every fixed-vs-adaptive difference in that run
# therefore mixed a scheduler effect with a decoding-recipe effect, which is why
# its README called the arms non-comparable. This run removes the confound by
# putting the fixed arm on the adaptive arm's sampling.
#
# Only the fixed arm is regenerated. The adaptive arm was already greedy, so it
# is imported from outputs/v2 after assemble_matched_root.py verifies the two
# agree on every other setting.
#
# Estimated makespan at 100 tasks on three GPUs: ~4.1h (upper bound; greedy
# skips the nucleus sort that the v2 fixed arm paid for).
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
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data}"
SOURCE_ROOT="${SOURCE_ROOT:-$PROJECT_DIR/outputs/v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/v3_matched}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/v3_matched}"
MODEL_ID="${MODEL_ID:-Dream-org/Dream-v0-Instruct-7B}"
DTYPE="${DTYPE:-bfloat16}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
EVALPLUS_WORKERS="${EVALPLUS_WORKERS:-4}"

# Everything below must match outputs/v2 or the arms stop being paired.
SEED="${SEED:-271828}"
MAX_CANDIDATES="${MAX_CANDIDATES:-64}"
MICROBATCH_GSM8K="${MICROBATCH_GSM8K:-8}"
MICROBATCH_SCHEMA_JSON="${MICROBATCH_SCHEMA_JSON:-4}"
MICROBATCH_MBPP_PLUS="${MICROBATCH_MBPP_PLUS:-4}"
# timing_repeats stays at 1. Raising it would tighten the new latencies but make
# them incomparable with the adaptive latencies already recorded in v2.
TIMING_REPEATS="${TIMING_REPEATS:-1}"
# Which fixed-k settings to regenerate. Narrow it for a cheap end-to-end check:
# KS=8 TASKS=1 OUTPUT_ROOT=... bash run_matched_fixed.sh
KS="${KS:-1 2 4 8}"

# The point of this run: the fixed arm adopts the adaptive arm's sampling.
# top_p is deliberately unset -- greedy argmax has nothing to filter.
FIXED_TEMPERATURE="${FIXED_TEMPERATURE:-0.0}"
FIXED_CONFIDENCE="${FIXED_CONFIDENCE:-logp}"

declare -A CANVAS=([mbpp_plus]=512 [gsm8k]=256 [schema_json]=256)
declare -A MICROBATCH=(
    [mbpp_plus]="$MICROBATCH_MBPP_PLUS"
    [gsm8k]="$MICROBATCH_GSM8K"
    [schema_json]="$MICROBATCH_SCHEMA_JSON"
)

if ! [ "$TASKS" -ge 1 ] 2>/dev/null; then
    echo "ERROR: TASKS must be a positive integer, got '$TASKS'" >&2
    exit 2
fi
for dataset in mbpp_plus gsm8k schema_json; do
    if [ ! -s "$DATA_ROOT/$dataset/test.jsonl" ]; then
        echo "ERROR: missing $DATA_ROOT/$dataset/test.jsonl" >&2
        exit 2
    fi
done
if [ ! -d "$SOURCE_ROOT" ]; then
    echo "ERROR: SOURCE_ROOT $SOURCE_ROOT does not exist; nothing to import the adaptive arm from." >&2
    exit 2
fi

# Free-GPU discovery. This is a shared host, so pick cards that are actually
# idle rather than assuming 0 1 2. Override with GPU_IDS="0 1" to pin.
FREE_MEMORY_MIB="${FREE_MEMORY_MIB:-1024}"
if [ -z "${GPU_IDS:-}" ]; then
    GPU_IDS="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', *' -v limit="$FREE_MEMORY_MIB" '$2 < limit {print $1}' | tr '\n' ' ')"
    GPU_IDS="${GPU_IDS% }"
fi
read -r -a GPUS <<< "$GPU_IDS"
if [ "${#GPUS[@]}" -lt 1 ]; then
    echo "ERROR: no GPU has less than ${FREE_MEMORY_MIB} MiB in use." >&2
    echo "Set GPU_IDS explicitly to run anyway." >&2
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv >&2
    exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

# Resuming into records written under different sampling would silently blend
# the confounded run with the matched one -- exactly the failure this run fixes.
for metadata in "$OUTPUT_ROOT"/*/fixed_*/metadata.json; do
    [ -e "$metadata" ] || continue
    if ! grep -q '"temperature": 0.0' "$metadata" || ! grep -q '"confidence_metric": "logp"' "$metadata"; then
        echo "ERROR: $metadata was written under different sampling." >&2
        echo "Point OUTPUT_ROOT elsewhere or delete it; do not resume into it." >&2
        exit 2
    fi
done

common=(
    --model "$MODEL_ID"
    --device cuda:0
    --dtype "$DTYPE"
    --seed "$SEED"
    --timing-repeats "$TIMING_REPEATS"
    --trace-limit 0
    --require-lean-logits
    --max-candidates "$MAX_CANDIDATES"
    --fixed-temperature "$FIXED_TEMPERATURE"
    --fixed-confidence "$FIXED_CONFIDENCE"
    --sample-size "$TASKS"
)
if [ "$LOCAL_FILES_ONLY" = "1" ]; then
    common+=(--local-files-only)
fi

PLAN_FILE="$LOG_ROOT/plan.txt"
# shellcheck disable=SC2086
"$PYTHON_BIN" scripts/pack_work.py --workers "${#GPUS[@]}" --tasks "$TASKS" --ks $KS --plan \
    >"$PLAN_FILE" 2>"$LOG_ROOT/plan_estimate.txt"

run_worker() {
    local worker_index="$1" physical_gpu="$2"
    local benchmark methods
    while read -r worker benchmark methods; do
        [ "$worker" = "$worker_index" ] || continue
        echo "[GPU $physical_gpu] $benchmark (canvas ${CANVAS[$benchmark]}, microbatch ${MICROBATCH[$benchmark]}): $methods"
        # shellcheck disable=SC2086
        EXPERIMENT_PHYSICAL_GPU_ID="$physical_gpu" \
        CUDA_VISIBLE_DEVICES="$physical_gpu" \
        "$PYTHON_BIN" -u scripts/run_benchmark.py \
            --dataset "$DATA_ROOT/$benchmark/test.jsonl" \
            --output-root "$OUTPUT_ROOT/$benchmark" \
            --continuation-length "${CANVAS[$benchmark]}" \
            --verifier-microbatch-size "${MICROBATCH[$benchmark]}" \
            --methods $methods \
            "${common[@]}"
    done <"$PLAN_FILE"
}

rm -f "$LOG_ROOT"/*.pid
pids=()
for index in "${!GPUS[@]}"; do
    run_worker "$index" "${GPUS[$index]}" >"$LOG_ROOT/worker_${GPUS[$index]}.log" 2>&1 &
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
Dream scheduler comparison -- matched-sampling fixed arm
started: $(date '+%Y-%m-%d %H:%M:%S %Z')
host: $(hostname)
purpose: remove the sampling confound between the fixed and adaptive arms of
         outputs/v2 by re-running the fixed arm under the adaptive recipe
tasks per benchmark: $TASKS
methods regenerated: $(for k in $KS; do printf 'fixed:%s ' "$k"; done)
adaptive arm: imported from $SOURCE_ROOT (already greedy; unchanged)
fixed sampling: T=$FIXED_TEMPERATURE top_p=none confidence=$FIXED_CONFIDENCE
adaptive sampling: T=0.0 top_p=none confidence=logp
max candidates: $MAX_CANDIDATES
microbatch: gsm8k=$MICROBATCH_GSM8K schema_json=$MICROBATCH_SCHEMA_JSON mbpp_plus=$MICROBATCH_MBPP_PLUS
seed: $SEED, drawn per (seed, method, task)
timing repeats: $TIMING_REPEATS
decision traces: off
GPUs: ${GPUS[*]}
output: $OUTPUT_ROOT
$(cat "$LOG_ROOT/plan_estimate.txt")
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

echo "== scoring MBPP+ =="
for method_dir in "$OUTPUT_ROOT/mbpp_plus"/*; do
    if [ -f "$method_dir/evalplus_samples.jsonl" ] && \
       [ ! -f "$method_dir/records.scored.jsonl" ]; then
        "$PYTHON_BIN" scripts/score_mbpp_plus.py \
            --parallel "$EVALPLUS_WORKERS" "$method_dir"
    fi
done

echo "== importing the adaptive arm =="
"$PYTHON_BIN" scripts/assemble_matched_root.py --source "$SOURCE_ROOT" --dest "$OUTPUT_ROOT"

echo "== re-scoring schema_json =="
"$PYTHON_BIN" scripts/rescore_schema_json.py "$OUTPUT_ROOT"

"$PYTHON_BIN" scripts/summarize_results.py \
    --output-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/comparison"

echo "Complete: $OUTPUT_ROOT/comparison/summary.json"
