#!/usr/bin/env bash
# Extends the finished core sweep with additional adaptive delta thresholds on
# the *same* frozen 100-example subset per benchmark. Sample identity comes from
# --sample-size + --seed alone (scripts/run_benchmark.py select_sample), so the
# subset is bit-identical to the core run as long as TASKS, SEED, and the
# data/*/test.jsonl files are unchanged.
#
# Records land in the core OUTPUT_ROOT next to the original eight methods, in
# fresh per-delta directories. Existing methods are never touched: run_method
# resumes completed example_ids and --overwrite is not passed.
#
# Two workers, paired cheap-with-expensive so they finish together:
#   GPU A: adaptive:0.10 then adaptive:1.50   (~11h + ~4h)
#   GPU B: adaptive:0.20 then adaptive:1.00   (~9h  + ~4.5h)
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

# These four must match the core run exactly or the subset changes.
TASKS="${TASKS:-100}"
SEED="${SEED:-12345}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/core}"

GPU_IDS="${GPU_IDS:-0 1}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/core-ext}"
MODEL_ID="${MODEL_ID:-Dream-org/Dream-v0-Instruct-7B}"
DTYPE="${DTYPE:-bfloat16}"
VERIFIER_MICROBATCH_SIZE="${VERIFIER_MICROBATCH_SIZE:-2}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
EVALPLUS_WORKERS="${EVALPLUS_WORKERS:-4}"

if ! [ "$TASKS" -ge 1 ] 2>/dev/null; then
    echo "ERROR: TASKS must be a positive integer, got '$TASKS'" >&2
    exit 2
fi
read -r -a GPUS <<< "$GPU_IDS"
if [ "${#GPUS[@]}" -ne 2 ]; then
    echo "ERROR: GPU_IDS must contain exactly two GPU IDs, got '$GPU_IDS'." >&2
    exit 2
fi
for dataset in mbpp_plus gsm8k schema_json; do
    if [ ! -s "$DATA_ROOT/$dataset/test.jsonl" ]; then
        echo "ERROR: missing $DATA_ROOT/$dataset/test.jsonl" >&2
        exit 2
    fi
done

# The method directory name is f"{delta:.2f}" with '.'->'p', so two deltas that
# agree to two decimals would silently share a directory and resume onto each
# other's records. Refuse a delta that is not exact at 2dp, and refuse one whose
# directory already exists in the core run.
NEW_DELTAS=(0.10 0.20 1.00 1.50)
for delta in "${NEW_DELTAS[@]}"; do
    label="adaptive_delta_$("$PYTHON_BIN" -c "print(f'{float(\"$delta\"):.2f}'.replace('.','p'))")"
    if [ "$("$PYTHON_BIN" -c "print(float(\"$delta\") == round(float(\"$delta\"), 2))")" != "True" ]; then
        echo "ERROR: delta $delta is not exact at two decimals; it would collide." >&2
        exit 2
    fi
    for benchmark in mbpp_plus gsm8k schema_json; do
        if [ -e "$OUTPUT_ROOT/$benchmark/$label" ]; then
            echo "ERROR: $OUTPUT_ROOT/$benchmark/$label already exists." >&2
            echo "       delta $delta collides with an existing method directory." >&2
            exit 2
        fi
    done
done

# Dream-7B in bfloat16 peaked at 14.52 GiB of the A4000's 15.60 GiB during the
# core run -- 1.08 GiB of headroom. One model per GPU is all that fits, and a
# co-tenant of any size makes model.to(device) fail. On this host that failure
# is not a clean torch OOM: the allocator's out-of-memory path calls nvmlInit,
# NVML is broken (kernel module 595.71.05 vs userspace libnvidia-ml 595.84), and
# the process dies on an INTERNAL ASSERT that names memory only in passing.
#
# So gate on free memory before committing to a load. Fail fast by default;
# set GPU_WAIT_SECONDS to wait out a transient co-tenant on this shared server.
REQUIRED_FREE_GIB="${REQUIRED_FREE_GIB:-14.8}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-0}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-120}"

# Prints free GiB on the given physical GPU; exit 0 if it clears the threshold,
# 1 if short, 2 if CUDA could not be probed at all. Uses a throwaway subprocess
# so the ~300 MiB probe context is released before the real worker starts.
gpu_has_room() {
    CUDA_VISIBLE_DEVICES="$1" REQUIRED_FREE_GIB="$REQUIRED_FREE_GIB" \
    "$PYTHON_BIN" - 2>/dev/null <<'PY'
import os
import sys

import torch

need = float(os.environ["REQUIRED_FREE_GIB"])
try:
    free, _total = torch.cuda.mem_get_info(0)
except Exception:
    sys.exit(2)
gib = free / 1024 ** 3
print(f"{gib:.2f}")
sys.exit(0 if gib >= need else 1)
PY
}

for gpu in "${GPUS[@]}"; do
    waited=0
    while :; do
        status=0
        free_gib="$(gpu_has_room "$gpu")" || status=$?
        if [ "$status" -eq 0 ]; then
            echo "GPU $gpu: ${free_gib} GiB free, need ${REQUIRED_FREE_GIB} GiB -- ok"
            break
        fi
        if [ "$status" -eq 2 ]; then
            echo "ERROR: could not probe GPU $gpu. CUDA itself is unavailable." >&2
            echo "       Check the driver: NVRM $(sed -n 's/^NVRM version:.*x86_64 *\([0-9.]*\).*/\1/p' /proc/driver/nvidia/version 2>/dev/null) vs userspace libnvidia-ml." >&2
            exit 2
        fi
        if [ "$waited" -ge "$GPU_WAIT_SECONDS" ]; then
            echo "ERROR: GPU $gpu has only ${free_gib} GiB free; need ${REQUIRED_FREE_GIB} GiB." >&2
            echo "       Dream-7B bf16 peaks at 14.52 GiB, so this GPU has a co-tenant." >&2
            echo "       Another user's job is the usual cause on this shared host --" >&2
            echo "       nvidia-smi cannot confirm it while NVML is broken. Options:" >&2
            echo "         - pick free GPUs:  GPU_IDS=\"2 3\" bash $(basename "${BASH_SOURCE[0]}")" >&2
            echo "         - wait it out:     GPU_WAIT_SECONDS=7200 bash $(basename "${BASH_SOURCE[0]}")" >&2
            exit 2
        fi
        echo "GPU $gpu: ${free_gib} GiB free, need ${REQUIRED_FREE_GIB} GiB -- waiting" \
             "(${waited}s/${GPU_WAIT_SECONDS}s)"
        sleep "$GPU_POLL_SECONDS"
        waited=$(( waited + GPU_POLL_SECONDS ))
    done
done

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

common=(
    --model "$MODEL_ID"
    --device cuda:0
    --dtype "$DTYPE"
    --timing-repeats 1
    --trace-limit 0
    --verifier-microbatch-size "$VERIFIER_MICROBATCH_SIZE"
    --sample-size "$TASKS"
    --seed "$SEED"
)
if [ "$LOCAL_FILES_ONLY" = "1" ]; then
    common+=(--local-files-only)
fi

# EXPERIMENT_PHYSICAL_GPU_ID doubles as the run_metadata_worker_*.json filename
# suffix. Using the bare GPU id would overwrite the core run's
# run_metadata_worker_{0,1,2}.json and lose its record of which methods that
# worker ran, so this run writes run_metadata_worker_ext<gpu>.json instead.
# CUDA_VISIBLE_DEVICES still carries the true physical GPU.
run_method_group() {
    local physical_gpu="$1"
    shift
    local methods=("$@")
    local benchmark length
    while read -r benchmark length; do
        echo "[GPU $physical_gpu] $benchmark: ${methods[*]}"
        EXPERIMENT_PHYSICAL_GPU_ID="ext$physical_gpu" \
        CUDA_VISIBLE_DEVICES="$physical_gpu" \
        "$PYTHON_BIN" -u scripts/run_benchmark.py \
            --dataset "$DATA_ROOT/$benchmark/test.jsonl" \
            --output-root "$OUTPUT_ROOT/$benchmark" \
            --continuation-length "$length" \
            --methods "${methods[@]}" \
            "${common[@]}"
    done <<'DATASETS'
mbpp_plus 512
gsm8k 256
schema_json 256
DATASETS
}

workers=(delta_0p10_and_1p50 delta_0p20_and_1p00)
rm -f "$LOG_ROOT"/*.pid

run_method_group "${GPUS[0]}" adaptive:0.10 adaptive:1.50 >"$LOG_ROOT/${workers[0]}.log" 2>&1 &
pids=("$!")
run_method_group "${GPUS[1]}" adaptive:0.20 adaptive:1.00 >"$LOG_ROOT/${workers[1]}.log" 2>&1 &
pids+=("$!")

for index in 0 1; do
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
Dream scheduler comparison -- adaptive delta extension
started: $(date '+%Y-%m-%d %H:%M:%S %Z')
host: $(hostname)
tasks per benchmark: $TASKS
seed: $SEED
new methods: ${NEW_DELTAS[*]}
extends: adaptive 0.15 0.30 0.50 0.75 + fixed 1 2 4 8
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

# Same guard as run.sh: EvalPlus is the source of MBPP+ pass@1, and a finished
# score is never recomputed. This only scores the new delta directories.
for method_dir in "$OUTPUT_ROOT/mbpp_plus"/*; do
    if [ -f "$method_dir/evalplus_samples.jsonl" ] && \
       [ ! -f "$method_dir/records.scored.jsonl" ]; then
        "$PYTHON_BIN" scripts/score_mbpp_plus.py \
            --parallel "$EVALPLUS_WORKERS" "$method_dir"
    fi
done

# Resummarize the whole output root so the comparison covers the original eight
# methods and the new deltas in one table.
"$PYTHON_BIN" scripts/summarize_results.py \
    --output-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/comparison"

echo "Complete: $OUTPUT_ROOT/comparison/summary.json"
