#!/usr/bin/env bash
# SDAR-8B-Chat-b64 fixed-k frontier on exactly exp05's SDAR-1.7B tasks.
#
# Everything except the checkpoint is exp05's B1 recipe, so the only variable
# is model scale (IMPLEMENTATION_AUDIT.md "Interpretation for exp06_vgb"):
#   * methods fixed:{1,2,4,8}; greedy T=0, logp confidence, max_candidates = block = 64
#   * 100 tasks per benchmark sampled with seed 271828 from the same data files
#   * canvases GSM8K 256, HumanEval+ 512, MBPP+ 256, MATH-500 1024; early stop at the first
#     settled stop token {151645 <|im_end|>, 151643 <|endoftext|>}
#   * logits read over IDs < 151665, BF16, one timed repeat, no traces
#
# What differs is placement. 16.4 GB of BF16 weights do not fit one 16 GB
# A4000, so each replica is pipeline-split over one PIX-connected pair
# (sdar_harness.model.shard_decoder: 18 layers per card) and two replicas run
# in parallel, one per pair. Timing synchronizes BOTH cards of a replica
# before starting and after finishing each task, so tokens/s is wall-clock
# for the whole model. Splitting computes the same function: SDAR-1.7B split
# over two cards reproduced the single-card exp05 records token for token.
#
# A split model runs its stages one after another, so 8B tokens/s includes one
# card-to-card hop per forward and cannot use the second card's compute in
# parallel. It is the latency of this deployment, not of an 8B model on one
# larger card.
#
# MATH-500's 1.7B reference is ported from exp04 (same data file, same 100
# ordered task IDs): reference/sdar_1p7b_chat_b64_math500. Its scores carry
# over; its timings do not, because exp04 decoded the full canvas with no early
# stop. REFERENCE_MATH=1 adds a phase 2 that reruns 1.7B MATH under exactly
# this recipe, one arm per card, when comparable MATH timings are needed.
#
#   bash launch_sdar8b.sh                     # full run
#   TASKS=5 OUTPUT_ROOT=outputs/smoke8b LOG_ROOT=logs/smoke8b bash launch_sdar8b.sh
#   DRY_RUN=1 bash launch_sdar8b.sh           # print the plan only
#   bash scripts/check_run.sh sdar8b          # progress, from another shell
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
# The shared harness (package, runner and scoring scripts) lives one level up.
HARNESS_DIR="$(cd -- "$PROJECT_DIR/../harness" && pwd)"
export PYTHONPATH="$HARNESS_DIR:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "${HOME}/miniconda3/envs/diffusion/bin/python" ]; then
        PYTHON_BIN="${HOME}/miniconda3/envs/diffusion/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

TASKS="${TASKS:-100}"
SEED="${SEED:-271828}"
# One replica per entry; each entry is the comma-separated pair it spans.
REPLICAS="${REPLICAS:-0,1 2,3}"
BENCHMARKS="${BENCHMARKS:-gsm8k humaneval_plus mbpp_plus math_500}"
METHODS=(${METHODS:-fixed:1 fixed:2 fixed:4 fixed:8})
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/../data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/sdar8b}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/sdar8b}"
SDAR_REPO="JetLM/SDAR-8B-Chat-b64"
SDAR_MODEL="${SDAR_MODEL:-$PROJECT_DIR/../models/SDAR-8B-Chat-b64-local}"
# Same transformers 4.52.4 overlay exp05 ran under.
SDAR_OVERLAY="${SDAR_OVERLAY:-$HARNESS_DIR/vendor_env/tf452}"
# 1.7B seconds-per-task from exp04. Only the RATIOS matter here: they balance
# the two replicas, and 8B scales every arm by roughly the same factor.
COSTS="${COSTS:-$PROJECT_DIR/reference/exp06_vgb_pack_costs.json}"
REFERENCE_MATH="${REFERENCE_MATH:-0}"
REF_MODEL="${REF_MODEL:-$PROJECT_DIR/../models/SDAR-1.7B-Chat-b64-local}"
REF_OUTPUT_ROOT="${REF_OUTPUT_ROOT:-$PROJECT_DIR/outputs/sdar1p7b_math500}"
EVALPLUS_WORKERS="${EVALPLUS_WORKERS:-4}"
PROFILE_STAGES="${PROFILE_STAGES:-0}"
# Any card already using more than this is shared, and its timings would be
# contended. ALLOW_BUSY=1 runs anyway and says so in RUNINFO.txt.
MAX_USED_MIB="${MAX_USED_MIB:-1000}"
ALLOW_BUSY="${ALLOW_BUSY:-0}"
BLOCK=64
MICROBATCH=4
VOCAB_LIMIT=151665

canvas_for() {
    case "$1" in
        gsm8k) echo 256 ;;
        humaneval_plus) echo 512 ;;
        mbpp_plus) echo 256 ;;
        # exp04's SDAR MATH canvas. 16% of 1.7B k=1 answers still hit it.
        math_500) echo 1024 ;;
        *) echo "ERROR: no canvas for $1" >&2; exit 2 ;;
    esac
}

read -r -a PAIRS <<< "$REPLICAS"
read -r -a BENCH_LIST <<< "$BENCHMARKS"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
echo "$$" >"$LOG_ROOT/launcher.pid"

echo "== preconditions =="
if [ ! -d "$SDAR_MODEL" ]; then
    echo "building patched local checkpoint at $SDAR_MODEL"
    "$PYTHON_BIN" -c "from huggingface_hub import snapshot_download; snapshot_download('$SDAR_REPO')"
    "$PYTHON_BIN" "$HARNESS_DIR/scripts/prepare_sdar_local.py" "$SDAR_REPO" \
        --output-root "$(dirname -- "$SDAR_MODEL")"
fi
if [ "$REFERENCE_MATH" = "1" ]; then
    [ -d "$REF_MODEL" ] || { echo "ERROR: no 1.7B checkpoint at $REF_MODEL" >&2; exit 2; }
fi
[ -d "$SDAR_OVERLAY" ] || { echo "ERROR: no transformers overlay at $SDAR_OVERLAY" >&2; exit 2; }
for bench in "${BENCH_LIST[@]}"; do
    [ -s "$DATA_ROOT/sample100/$bench.jsonl" ] || { echo "ERROR: missing $DATA_ROOT/sample100/$bench.jsonl" >&2; exit 2; }
    canvas_for "$bench" >/dev/null
done
busy=""
for pair in "${PAIRS[@]}"; do
    IFS=, read -r -a cards <<< "$pair"
    [ "${#cards[@]}" -ge 2 ] || { echo "ERROR: replica '$pair' must name at least two cards" >&2; exit 2; }
    for gpu in "${cards[@]}"; do
        used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
        if [ "$used" -gt "$MAX_USED_MIB" ]; then
            busy="$busy GPU$gpu=${used}MiB"
        fi
    done
done
if [ -n "$busy" ]; then
    if [ "$ALLOW_BUSY" = "1" ]; then
        echo "WARNING: shared GPUs ($busy); timings will be contended."
    else
        echo "ERROR: GPUs already in use:$busy. Timings would be contended." >&2
        echo "       Wait for them to free up, or rerun with ALLOW_BUSY=1." >&2
        exit 2
    fi
fi

PLAN="$LOG_ROOT/plan.txt"
"$PYTHON_BIN" "$HARNESS_DIR/scripts/pack_work.py" --workers "${#PAIRS[@]}" --costs "$COSTS" \
    --tasks "$TASKS" --plan --benchmarks "${BENCH_LIST[@]}" --methods "${METHODS[@]}" >"$PLAN"
echo "== plan (1.7B costs, for balancing only) =="
cat "$PLAN"

cat >"$LOG_ROOT/RUNINFO.txt" <<INFO
exp06_vgb SDAR-8B-Chat-b64 fixed-k frontier
started: $(date '+%Y-%m-%d %H:%M:%S %Z')   host: $(hostname)
methods: ${METHODS[*]}
benchmarks: ${BENCH_LIST[*]} (canvas $(for b in "${BENCH_LIST[@]}"; do printf '%s=%s ' "$b" "$(canvas_for "$b")"; done))
tasks: $TASKS   seed: $SEED   block: $BLOCK   microbatch: $MICROBATCH   vocab_limit: $VOCAB_LIMIT
stop tokens: checkpoint generation_config (151643, 151645)   early stop: on
recipe: greedy T=0, logp confidence, max_candidates=$BLOCK
model: $SDAR_MODEL
replicas: ${PAIRS[*]} (one pipeline-split replica per pair; all cards synced around every timed task)
shared GPUs at launch: ${busy:-none}
output: $OUTPUT_ROOT
INFO
cat "$LOG_ROOT/RUNINFO.txt"

# Arguments shared by both phases: exp05's B1 recipe, verbatim.
recipe_args() {
    local bench="$1"
    printf '%s\n' \
        --dataset "$DATA_ROOT/sample100/$bench.jsonl" \
        --continuation-length "$(canvas_for "$bench")" \
        --verifier-microbatch-size "$MICROBATCH" \
        --family sdar --block-length "$BLOCK" --dtype bfloat16 --seed "$SEED" \
        --timing-repeats 1 --trace-limit 0 --require-lean-logits \
        --profile-stages "$PROFILE_STAGES" \
        --max-candidates "$BLOCK" --sample-size "$TASKS" --local-files-only \
        --vocab-limit "$VOCAB_LIMIT" \
        --fixed-temperature 0.0 --fixed-confidence logp
}

run_worker() {
    local worker="$1" pair="$2"
    local count devices=()
    IFS=, read -r -a cards <<< "$pair"
    for count in "${!cards[@]}"; do devices+=("cuda:$count"); done
    while read -r index bench methods; do
        [ "$index" = "$worker" ] || continue
        echo "[GPUs $pair] $bench (canvas $(canvas_for "$bench")): $methods"
        if [ "${DRY_RUN:-0}" = "1" ]; then continue; fi
        mapfile -t recipe < <(recipe_args "$bench")
        # shellcheck disable=SC2086
        EXPERIMENT_PHYSICAL_GPU_ID="${pair//,/+}" CUDA_VISIBLE_DEVICES="$pair" \
        PYTHONPATH="$SDAR_OVERLAY:$PYTHONPATH" \
        "$PYTHON_BIN" -u "$HARNESS_DIR/scripts/run_benchmark.py" "${recipe[@]}" \
            --output-root "$OUTPUT_ROOT/$bench" \
            --methods $methods \
            --model "$SDAR_MODEL" --devices "${devices[@]}"
    done <"$PLAN"
}

# Phase 2: one 1.7B arm per physical card, exactly as exp05 ran its baselines.
run_reference_math() {
    local gpu="$1" method="$2"
    echo "[GPU $gpu] 1.7B math_500 (canvas $(canvas_for math_500)): $method"
    if [ "${DRY_RUN:-0}" = "1" ]; then return; fi
    mapfile -t recipe < <(recipe_args math_500)
    EXPERIMENT_PHYSICAL_GPU_ID="$gpu" CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$SDAR_OVERLAY:$PYTHONPATH" \
    "$PYTHON_BIN" -u "$HARNESS_DIR/scripts/run_benchmark.py" "${recipe[@]}" \
        --output-root "$REF_OUTPUT_ROOT/math_500" \
        --methods "$method" \
        --model "$REF_MODEL" --device cuda:0
}

pids=()
for i in "${!PAIRS[@]}"; do
    name="replica${i}_gpu${PAIRS[$i]//,/_}"
    run_worker "$i" "${PAIRS[$i]}" >"$LOG_ROOT/$name.log" 2>&1 &
    pids+=("$!")
    echo "$!" >"$LOG_ROOT/$name.pid"
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" = "0" ] || { echo "ERROR: a replica failed; see $LOG_ROOT" >&2; exit 1; }

REF_SUMMARY=("$PROJECT_DIR/reference/sdar_1p7b_chat_b64_math500/comparison/summary.exp06_vgb.json")
if [ "$REFERENCE_MATH" = "1" ] && [[ " ${BENCH_LIST[*]} " == *" math_500 "* ]]; then
    echo "== phase 2: SDAR-1.7B MATH-500 reference =="
    read -r -a ALL_CARDS <<< "$(echo "${PAIRS[*]}" | tr ',' ' ')"
    pids=()
    for i in "${!METHODS[@]}"; do
        gpu="${ALL_CARDS[$(( i % ${#ALL_CARDS[@]} ))]}"
        name="ref_math_${METHODS[$i]/:/_}_gpu$gpu"
        # Arms that outnumber cards queue behind the arm already on that card.
        if [ "$i" -ge "${#ALL_CARDS[@]}" ]; then wait "${pids[$(( i - ${#ALL_CARDS[@]} ))]}" || failed=1; fi
        run_reference_math "$gpu" "${METHODS[$i]}" >"$LOG_ROOT/$name.log" 2>&1 &
        pids+=("$!")
        echo "$!" >"$LOG_ROOT/$name.pid"
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [ "$failed" = "0" ] || { echo "ERROR: a 1.7B MATH arm failed; see $LOG_ROOT" >&2; exit 1; }
    if [ "${DRY_RUN:-0}" != "1" ]; then
        "$PYTHON_BIN" "$HARNESS_DIR/scripts/summarize_results.py" --output-root "$REF_OUTPUT_ROOT" \
            --output-dir "$REF_OUTPUT_ROOT/comparison"
        # The rerun replaces the port: same scores rules, comparable timings.
        REF_SUMMARY=("$REF_OUTPUT_ROOT/comparison/summary.json")
    fi
fi
if [ "${DRY_RUN:-0}" = "1" ]; then exit 0; fi

echo "== scoring =="
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
if [ "$TASKS" = "100" ]; then
    "$PYTHON_BIN" scripts/compare_scale.py "$OUTPUT_ROOT/comparison/summary.json" \
        --reference "$PROJECT_DIR/reference/sdar_1p7b_chat_b64/comparison/summary.json" \
        "${REF_SUMMARY[@]}" | tee "$LOG_ROOT/compare_scale.txt"
fi
echo "Done: $OUTPUT_ROOT/comparison/summary.json"
