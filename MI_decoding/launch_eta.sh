#!/usr/bin/env bash
# The η study (ETA_STUDY.md): SDAR-8B-b16 decodes each prompt at fixed:1, and
# at one random step per block the 8B and SDAR-1.7B-b16 read that step's probe
# rows; measure_eta.py records D^P, D^V and η per rank. One replica, one card
# pair, datasets one after another. Resumable: rerunning skips finished tasks.
#
#   bash launch_eta.sh                                   # held-out GSM8K / MATH / MBPP, 150 each
#   LIMIT=10 OUTPUT_ROOT=outputs/eta_pilot bash launch_eta.sh
#   DATASETS="../data/opencompass/gsm8k.jsonl" LIMIT=5 \
#       CHECK_ROOT=outputs/opencompass/SDAR-8B-Chat-b16-ec2 bash launch_eta.sh   # decode == baseline?
#   DATASETS="math_train" PAIR=2,3 bash launch_eta.sh    # a second replica on other cards
#   DATASETS="gsm8k_train mbpp_heldout:4" bash launch_eta.sh   # name:N measures N steps per block there
#   PLACEMENT=split bash launch_eta.sh                   # 8B over the pair, 1.7B on the second (16 GB cards)
#   DRY_RUN=1 bash launch_eta.sh
#
# DATASETS entries are names in DATA_DIR (default ../data/heldout) or paths to
# .jsonl files, each optionally suffixed :N to measure N steps per block on that
# dataset instead of MEASURE_PER_BLOCK (the default puts MBPP's short answers at 4). CHECK_ROOT/<stem>/fixed_k_1/records.jsonl, when it exists, is
# the fixed:1 baseline each completion is compared with.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
HARNESS_DIR="$(cd -- "$PROJECT_DIR/../harness" && pwd)"
export PYTHONPATH="$HARNESS_DIR:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
# The baselines ran with deterministic cuBLAS; so must the decode compared with them.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/diffusion/bin/python}"
PAIR="${PAIR:-0,1}"
PLACEMENT="${PLACEMENT:-separate}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/../data/heldout}"
DATASETS="${DATASETS:-gsm8k_train math_train mbpp_heldout:4}"
PROPOSER_MODEL="${PROPOSER_MODEL:-$PROJECT_DIR/../models/SDAR-8B-Chat-b16-local}"
REFERENCE_MODEL="${REFERENCE_MODEL:-$PROJECT_DIR/../models/SDAR-1.7B-Chat-b16-local}"
BLOCK="${BLOCK:-16}"
CANVAS="${CANVAS:-4096}"
GROUP_ROWS="${GROUP_ROWS:-16}"
MEASURE_PER_BLOCK="${MEASURE_PER_BLOCK:-1}"
MAX_MEASURED="${MAX_MEASURED:-64}"
FLOOR_TASKS="${FLOOR_TASKS:-10}"
FLOOR_GROUP_ROWS="${FLOOR_GROUP_ROWS:-4}"
SEED="${SEED:-271828}"
LIMIT="${LIMIT:-}"
CHECK_ROOT="${CHECK_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/eta/heldout_b16}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/eta/$(basename -- "$OUTPUT_ROOT")}"
SDAR_OVERLAY="${SDAR_OVERLAY:-$HARNESS_DIR/vendor_env/tf452}"

case "$PLACEMENT" in
    separate) DEVICE_ARGS=(--proposer-devices cuda:0 --reference-device cuda:1) ;;
    split) DEVICE_ARGS=(--proposer-devices cuda:0 cuda:1 --reference-device cuda:1) ;;
    *) echo "ERROR: PLACEMENT must be separate or split" >&2; exit 2 ;;
esac

dataset_file() {
    local name="${1%%:*}"
    if [ -f "$name" ]; then echo "$name"; else echo "$DATA_DIR/$name.jsonl"; fi
}
per_block() {
    case "$1" in *:*) echo "${1##*:}" ;; *) echo "$MEASURE_PER_BLOCK" ;; esac
}

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
echo "== preconditions =="
for path in "$PROPOSER_MODEL" "$REFERENCE_MODEL" "$SDAR_OVERLAY"; do
    [ -d "$path" ] || { echo "ERROR: missing $path" >&2; exit 2; }
done
for name in $DATASETS; do
    [ -s "$(dataset_file "$name")" ] || { echo "ERROR: missing $(dataset_file "$name")" >&2; exit 2; }
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
η study: SDAR-8B-b$BLOCK decodes at fixed:1; 8B and 1.7B read each measured step's probe rows
started: $(date '+%Y-%m-%d %H:%M:%S %Z')   host: $(hostname)   cards: $PAIR   placement: $PLACEMENT
datasets: $(for n in $DATASETS; do printf '%s(%s/block) ' "$(dataset_file "$n")" "$(per_block "$n")"; done)
limit: ${LIMIT:-all}   canvas: $CANVAS   block: $BLOCK   group rows: $GROUP_ROWS   measured steps per block: $MEASURE_PER_BLOCK (max $MAX_MEASURED per task)
self-floor: first $FLOOR_TASKS tasks per dataset, grouping $FLOOR_GROUP_ROWS   seed: $SEED   check root: ${CHECK_ROOT:-none}
proposer: $PROPOSER_MODEL   reference: $REFERENCE_MODEL   fp16, whole vocabulary
output: $OUTPUT_ROOT
INFO
cat "$LOG_ROOT/RUNINFO.txt"
[ "${DRY_RUN:-0}" = "1" ] && exit 0

for name in $DATASETS; do
    file="$(dataset_file "$name")"
    stem="$(basename -- "$file" .jsonl)"
    extra=()
    [ -n "$LIMIT" ] && extra+=(--limit "$LIMIT")
    if [ -n "$CHECK_ROOT" ] && [ -f "$CHECK_ROOT/$stem/fixed_k_1/records.jsonl" ]; then
        extra+=(--check-against "$CHECK_ROOT/$stem/fixed_k_1/records.jsonl")
    fi
    echo "== $stem ($(date '+%H:%M:%S')) =="
    CUDA_VISIBLE_DEVICES="$PAIR" PYTHONPATH="$SDAR_OVERLAY:$PYTHONPATH" \
    "$PYTHON_BIN" -u "$PROJECT_DIR/measure_eta.py" \
        --dataset "$file" --output-dir "$OUTPUT_ROOT/$stem" \
        --proposer "$PROPOSER_MODEL" --reference "$REFERENCE_MODEL" "${DEVICE_ARGS[@]}" \
        --block-length "$BLOCK" --continuation-length "$CANVAS" --dtype float16 --vocab-limit 151936 \
        --group-rows "$GROUP_ROWS" --measure-per-block "$(per_block "$name")" --max-measured "$MAX_MEASURED" \
        --floor-tasks "$FLOOR_TASKS" --floor-group-rows "$FLOOR_GROUP_ROWS" --seed "$SEED" \
        --local-files-only "${extra[@]}" \
        >"$LOG_ROOT/$stem.log" 2>&1
done
echo "Done ($(date '+%H:%M:%S'))"
