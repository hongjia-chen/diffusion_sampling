#!/usr/bin/env bash
# Four-GPU, schedule-parallel Dream time-versus-quality experiment.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ulimit -v unlimited
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-$SCRIPT_DIR/data/owt_p64_l128.jsonl}"
DATASET_METADATA="${DATASET_METADATA:-$SCRIPT_DIR/data/owt_p64_l128.metadata.json}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
NUM_EXAMPLES="${NUM_EXAMPLES:-500}"
PREFIX_LENGTH="${PREFIX_LENGTH:-64}"
CONTINUATION_LENGTH="${CONTINUATION_LENGTH:-128}"
TIMING_REPEATS="${TIMING_REPEATS:-1}"
TRAJECTORY_EXAMPLE_IDS="${TRAJECTORY_EXAMPLE_IDS:-0 1 2}"
DETERMINISM_CHECK_EXAMPLES="${DETERMINISM_CHECK_EXAMPLES:-3}"
MODEL_ID="${MODEL_ID:-Dream-org/Dream-v0-Base-7B}"
MODEL_REVISION="${MODEL_REVISION:-}"
DTYPE="${DTYPE:-bfloat16}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
MAUVE_BATCH_SIZE="${MAUVE_BATCH_SIZE:-8}"
DATE_TAG="${DATE_TAG:-0729}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/${DATE_TAG}_time_vs_quality_7B_n${NUM_EXAMPLES}}"
SUBSET_DIR="$OUTPUT_ROOT/data"
SUBSET_DATASET="$SUBSET_DIR/owt_n${NUM_EXAMPLES}_p${PREFIX_LENGTH}_l${CONTINUATION_LENGTH}.jsonl"
SUBSET_METADATA="$SUBSET_DIR/owt_n${NUM_EXAMPLES}_p${PREFIX_LENGTH}_l${CONTINUATION_LENGTH}.metadata.json"
SCHEDULE_ROOT="$OUTPUT_ROOT/schedule_runs"
JOB_LOG_DIR="$OUTPUT_ROOT/job_logs"
MERGED_ROOT="$OUTPUT_ROOT/merged"
FIXED_RUN="$MERGED_ROOT/fixed"
ADAPTIVE_RUN="$MERGED_ROOT/adaptive"
COMPARISON_DIR="$OUTPUT_ROOT/comparison"
TRAJECTORY_DIR="$COMPARISON_DIR/trajectories"

REQUIRED_FILES=(
    dream_adapter.py
    adaptive_scheduler.py
    build_dataset_subset.py
    merge_schedule_runs.py
    run_dream_topk.py
    audit_dream_topk.py
    compute_dream_mauve.py
    summarize_dream_topk.py
    run_dream_adaptive.py
    audit_dream_adaptive.py
    compute_dream_adaptive_mauve.py
    summarize_dream_adaptive.py
    extract_dream_trajectories.py
    plot_time_vs_quality.py
)
for path in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$path" ]; then
        echo "ERROR: required file is missing: $SCRIPT_DIR/$path" >&2
        exit 1
    fi
done
for path in "$DATASET" "$DATASET_METADATA"; do
    if [ ! -f "$path" ]; then
        echo "ERROR: source dataset artifact does not exist: $path" >&2
        exit 1
    fi
done
if [ "$TIMING_REPEATS" != "1" ]; then
    echo "ERROR: overnight design requires TIMING_REPEATS=1." >&2
    exit 1
fi

read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [ "${#GPU_ARRAY[@]}" -ne 4 ]; then
    echo "ERROR: GPU_IDS must contain exactly four physical GPU IDs." >&2
    exit 1
fi
read -r -a TRAJECTORY_ARGS <<< "$TRAJECTORY_EXAMPLE_IDS"
GPU_CSV="$(IFS=,; echo "${GPU_ARRAY[*]}")"

"$PYTHON_BIN" -m py_compile "${REQUIRED_FILES[@]}"
for script in "${REQUIRED_FILES[@]}"; do
    case "$script" in
        *.py) "$PYTHON_BIN" "$script" --help >/dev/null ;;
    esac
done

CUDA_VISIBLE_DEVICES="$GPU_CSV" "$PYTHON_BIN" - <<'PY'
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    sys.exit(
        f"ERROR: expected four visible CUDA GPUs, found {torch.cuda.device_count()}."
    )
for index in range(4):
    p = torch.cuda.get_device_properties(index)
    print(f"[OK] logical GPU {index}: {p.name} ({p.total_memory / 1024**3:.1f} GiB)")
PY

mkdir -p "$SUBSET_DIR" "$SCHEDULE_ROOT" "$JOB_LOG_DIR" "$COMPARISON_DIR"
subset_args=()
if [ "$FORCE_REBUILD" = "1" ]; then
    subset_args+=(--overwrite)
fi
"$PYTHON_BIN" -u build_dataset_subset.py \
    --source "$DATASET" \
    --source-metadata "$DATASET_METADATA" \
    --output "$SUBSET_DATASET" \
    --metadata-output "$SUBSET_METADATA" \
    --num-examples "$NUM_EXAMPLES" \
    "${subset_args[@]}"

metadata_complete() {
    local run_dir="$1"
    [ -f "$run_dir/metadata.json" ] && \
        "$PYTHON_BIN" -c \
        'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["status"] == "complete" else 1)' \
        "$run_dir/metadata.json"
}

model_args=(--model "$MODEL_ID" --device cuda:0 --dtype "$DTYPE")
if [ -n "$MODEL_REVISION" ]; then
    model_args+=(--model-revision "$MODEL_REVISION")
fi
if [ "$LOCAL_FILES_ONLY" = "1" ]; then
    model_args+=(--local-files-only)
fi

run_fixed_job() {
    local gpu="$1"
    local k="$2"
    local run_dir="$SCHEDULE_ROOT/fixed_k_${k}"
    local log="$JOB_LOG_DIR/fixed_k_${k}.log"
    echo "[queue] GPU $gpu starting fixed k=$k (log: $log)"
    {
        if metadata_complete "$run_dir" && [ "$FORCE_REBUILD" != "1" ]; then
            echo "[reuse] complete generation: $run_dir"
        else
            command=(
                "$PYTHON_BIN" -u run_dream_topk.py
                --dataset "$SUBSET_DATASET"
                --dataset-metadata "$SUBSET_METADATA"
                --output-dir "$run_dir"
                --prefix-length "$PREFIX_LENGTH"
                --continuation-length "$CONTINUATION_LENGTH"
                --k-values "$k"
                --batch-size 1
                --timing-repeats 1
                --warmup-batches 1
                --trajectory-example-ids "${TRAJECTORY_ARGS[@]}"
                --determinism-check-examples "$DETERMINISM_CHECK_EXAMPLES"
                "${model_args[@]}"
            )
            if [ "$FORCE_REBUILD" = "1" ]; then
                command+=(--overwrite)
            fi
            EXPERIMENT_PHYSICAL_GPU_ID="$gpu" \
            CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}"
        fi
        "$PYTHON_BIN" -u audit_dream_topk.py "$run_dir" \
            --dataset "$SUBSET_DATASET" \
            --dataset-metadata "$SUBSET_METADATA" \
            --prefix-length "$PREFIX_LENGTH" \
            --continuation-length "$CONTINUATION_LENGTH" \
            --k-values "$k"
        mauve_args=()
        if [ -f "$run_dir/mauve.jsonl" ]; then
            mauve_args+=(--resume)
        fi
        EXPERIMENT_PHYSICAL_GPU_ID="$gpu" \
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u compute_dream_mauve.py \
            "$run_dir" \
            --dataset "$SUBSET_DATASET" \
            --prefix-length "$PREFIX_LENGTH" \
            --continuation-length "$CONTINUATION_LENGTH" \
            --k-values "$k" \
            --device-id 0 \
            --batch-size "$MAUVE_BATCH_SIZE" \
            "${mauve_args[@]}" \
            --hard-exit-after-success
    } > "$log" 2>&1
    echo "[queue] GPU $gpu completed fixed k=$k"
}

run_adaptive_job() {
    local gpu="$1"
    local multiplier="$2"
    local label="$3"
    local run_dir="$SCHEDULE_ROOT/adaptive_${label}"
    local log="$JOB_LOG_DIR/adaptive_${label}.log"
    echo "[queue] GPU $gpu starting adaptive log($multiplier) (log: $log)"
    {
        if metadata_complete "$run_dir" && [ "$FORCE_REBUILD" != "1" ]; then
            echo "[reuse] complete generation: $run_dir"
        else
            command=(
                "$PYTHON_BIN" -u run_dream_adaptive.py
                --dataset "$SUBSET_DATASET"
                --dataset-metadata "$SUBSET_METADATA"
                --output-dir "$run_dir"
                --prefix-length "$PREFIX_LENGTH"
                --continuation-length "$CONTINUATION_LENGTH"
                --tau-multipliers "$multiplier"
                --max-candidates 8
                --batch-size 1
                --conditional-microbatch-size 1
                --timing-repeats 1
                --warmup-examples 1
                --trajectory-example-ids "${TRAJECTORY_ARGS[@]}"
                --determinism-check-examples "$DETERMINISM_CHECK_EXAMPLES"
                "${model_args[@]}"
            )
            if [ "$FORCE_REBUILD" = "1" ]; then
                command+=(--overwrite)
            fi
            EXPERIMENT_PHYSICAL_GPU_ID="$gpu" \
            CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}"
        fi
        "$PYTHON_BIN" -u audit_dream_adaptive.py "$run_dir" \
            --dataset "$SUBSET_DATASET" \
            --dataset-metadata "$SUBSET_METADATA" \
            --prefix-length "$PREFIX_LENGTH" \
            --continuation-length "$CONTINUATION_LENGTH"
        mauve_args=()
        if [ -f "$run_dir/adaptive_mauve.jsonl" ]; then
            mauve_args+=(--resume)
        fi
        EXPERIMENT_PHYSICAL_GPU_ID="$gpu" \
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u \
            compute_dream_adaptive_mauve.py "$run_dir" \
            --dataset "$SUBSET_DATASET" \
            --device-id 0 \
            --batch-size "$MAUVE_BATCH_SIZE" \
            "${mauve_args[@]}" \
            --hard-exit-after-success
    } > "$log" 2>&1
    echo "[queue] GPU $gpu completed adaptive log($multiplier)"
}

worker_zero() {
    run_adaptive_job "${GPU_ARRAY[0]}" 1.01 tau_log_1p01
    run_fixed_job "${GPU_ARRAY[0]}" 5
}
worker_one() {
    run_adaptive_job "${GPU_ARRAY[1]}" 1.10 tau_log_1p10
    run_fixed_job "${GPU_ARRAY[1]}" 6
    run_fixed_job "${GPU_ARRAY[1]}" 8
}
worker_two() {
    run_adaptive_job "${GPU_ARRAY[2]}" 1.25 tau_log_1p25
    run_fixed_job "${GPU_ARRAY[2]}" 7
}
worker_three() {
    run_fixed_job "${GPU_ARRAY[3]}" 1
    run_fixed_job "${GPU_ARRAY[3]}" 2
    run_fixed_job "${GPU_ARRAY[3]}" 3
    run_fixed_job "${GPU_ARRAY[3]}" 4
}

echo "[parallel] launching eleven schedules across GPUs: ${GPU_ARRAY[*]}"
worker_zero &
PIDS=("$!")
worker_one &
PIDS+=("$!")
worker_two &
PIDS+=("$!")
worker_three &
PIDS+=("$!")

worker_failure=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        worker_failure=1
    fi
done
if [ "$worker_failure" -ne 0 ]; then
    echo "ERROR: one or more GPU workers failed. Inspect $JOB_LOG_DIR." >&2
    exit 1
fi

fixed_runs=()
for k in 1 2 3 4 5 6 7 8; do
    fixed_runs+=("$SCHEDULE_ROOT/fixed_k_${k}")
done
adaptive_runs=(
    "$SCHEDULE_ROOT/adaptive_tau_log_1p01"
    "$SCHEDULE_ROOT/adaptive_tau_log_1p10"
    "$SCHEDULE_ROOT/adaptive_tau_log_1p25"
)

echo "[merge] combining independently timed schedule artifacts"
"$PYTHON_BIN" -u merge_schedule_runs.py \
    --fixed-runs "${fixed_runs[@]}" \
    --adaptive-runs "${adaptive_runs[@]}" \
    --output-root "$MERGED_ROOT" \
    --overwrite

echo "[merge] auditing reconstructed combined runs"
"$PYTHON_BIN" -u audit_dream_topk.py "$FIXED_RUN" \
    --dataset "$SUBSET_DATASET" \
    --dataset-metadata "$SUBSET_METADATA" \
    --prefix-length "$PREFIX_LENGTH" \
    --continuation-length "$CONTINUATION_LENGTH" \
    --k-values 1 2 3 4 5 6 7 8
"$PYTHON_BIN" -u audit_dream_adaptive.py "$ADAPTIVE_RUN" \
    --dataset "$SUBSET_DATASET" \
    --dataset-metadata "$SUBSET_METADATA" \
    --prefix-length "$PREFIX_LENGTH" \
    --continuation-length "$CONTINUATION_LENGTH"

"$PYTHON_BIN" -u summarize_dream_topk.py "$FIXED_RUN"
"$PYTHON_BIN" -u summarize_dream_adaptive.py "$ADAPTIVE_RUN"

"$PYTHON_BIN" -u extract_dream_trajectories.py \
    --fixed-run "$FIXED_RUN" \
    --adaptive-run "$ADAPTIVE_RUN" \
    --dataset "$SUBSET_DATASET" \
    --example-ids "${TRAJECTORY_ARGS[@]}" \
    --output-dir "$TRAJECTORY_DIR"

"$PYTHON_BIN" -u plot_time_vs_quality.py \
    --fixed-summary "$FIXED_RUN/summary.csv" \
    --adaptive-summary "$ADAPTIVE_RUN/adaptive_summary.csv" \
    --output-dir "$COMPARISON_DIR"

echo "[done] four-GPU experiment: $OUTPUT_ROOT"
echo "  plot:         $COMPARISON_DIR/time_vs_quality.png"
echo "  trajectories: $TRAJECTORY_DIR/trajectory_all.txt"
echo "  worker logs:  $JOB_LOG_DIR"
