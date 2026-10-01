#!/usr/bin/env bash
# SDAR-only baselines under the exp05 rules (DESIGN.md "Baselines to re-run").
#
#   PHASE=b1   SDAR fixed:{1,2,4,8}                         -- the proposer's frontier
#   PHASE=b2   SDAR-verified adaptive:{0.10,...,10.00}  -- DROPPED 2026-09-19, kept runnable
#
# exp05 rules, all on by default in ../harness/scripts/run_benchmark.py or set below:
#   * early stop at the first stop token once everything before it is committed
#   * stop set = SDAR's generation_config eos_token_id = {151645 <|im_end|>, 151643 <|endoftext|>}
#   * logits read over the shared SDAR/Dream vocabulary, IDs < 151665
#   * canvases GSM8K 256, HumanEval+ 512, MBPP+ 256 (MBPP+ was 512 in exp04)
#
# Recipe is exp04's SDAR recipe unchanged: greedy (T=0), logp confidence,
# max_candidates = block = 64, verifier groups of 4, seed 271828, 100 tasks.
# SDAR is block-causal, so none of the new rules can change a token before the
# stop: B1 must reproduce exp04's completions (compare with
# scripts/compare_exp04.py once it finishes).
#
# One process per GPU, so every timed arm has the card to itself.
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

PHASE="${PHASE:-b1}"
TASKS="${TASKS:-100}"
SEED="${SEED:-271828}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
BENCHMARKS="${BENCHMARKS:-gsm8k humaneval_plus mbpp_plus}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/../data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/baseline/sdar}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/baseline_$PHASE}"
SDAR_MODEL="${SDAR_MODEL:-$PROJECT_DIR/../models/SDAR-1.7B-Chat-b64-local}"
SDAR_OVERLAY="${SDAR_OVERLAY:-$HARNESS_DIR/vendor_env/tf452}"
COSTS="${COSTS:-$PROJECT_DIR/reference/exp04_sdar_costs.json}"
EVALPLUS_WORKERS="${EVALPLUS_WORKERS:-4}"
BLOCK=64
MICROBATCH=4
VOCAB_LIMIT=151665

canvas_for() {
    case "$1" in
        gsm8k) echo 256 ;;
        humaneval_plus) echo 512 ;;
        mbpp_plus) echo 256 ;;
        *) echo "ERROR: no exp05 canvas for $1" >&2; exit 2 ;;
    esac
}

case "$PHASE" in
    b1) METHODS=(fixed:1 fixed:2 fixed:4 fixed:8) ;;
    # exp04's five deltas (for the reproduction check against exp04) plus
    # 0.30, 3.00 and 10.00, so B2 covers every point of the cross arms' grid.
    b2) METHODS=(adaptive:0.10 adaptive:0.20 adaptive:0.30 adaptive:0.50 adaptive:1.00
                 adaptive:1.50 adaptive:3.00 adaptive:10.00) ;;
    *) echo "ERROR: PHASE must be b1 or b2, got '$PHASE'" >&2; exit 2 ;;
esac

read -r -a GPUS <<< "$GPU_IDS"
read -r -a BENCH_LIST <<< "$BENCHMARKS"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

echo "== preconditions =="
[ -d "$SDAR_MODEL" ] || { echo "ERROR: no checkpoint at $SDAR_MODEL" >&2; exit 2; }
[ -d "$SDAR_OVERLAY" ] || { echo "ERROR: no transformers overlay at $SDAR_OVERLAY" >&2; exit 2; }
for bench in "${BENCH_LIST[@]}"; do
    [ -s "$DATA_ROOT/$bench/test.jsonl" ] || { echo "ERROR: missing $DATA_ROOT/$bench/test.jsonl" >&2; exit 2; }
    canvas_for "$bench" >/dev/null
done
for gpu in "${GPUS[@]}"; do
    used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
    if [ "$used" -gt 1000 ]; then
        echo "ERROR: GPU $gpu already has ${used} MiB in use; timings would be contended." >&2
        exit 2
    fi
done

PLAN="$LOG_ROOT/plan.txt"
"$PYTHON_BIN" "$HARNESS_DIR/scripts/pack_work.py" --workers "${#GPUS[@]}" --costs "$COSTS" \
    --tasks "$TASKS" --plan --benchmarks "${BENCH_LIST[@]}" --methods "${METHODS[@]}" >"$PLAN"
echo "== plan (costs are exp04 upper bounds) =="
cat "$PLAN"

cat >"$LOG_ROOT/RUNINFO.txt" <<INFO
exp05 SDAR baselines, phase $PHASE
started: $(date '+%Y-%m-%d %H:%M:%S %Z')   host: $(hostname)
methods: ${METHODS[*]}
benchmarks: ${BENCH_LIST[*]} (canvas $(for b in "${BENCH_LIST[@]}"; do printf '%s=%s ' "$b" "$(canvas_for "$b")"; done))
tasks: $TASKS   seed: $SEED   block: $BLOCK   microbatch: $MICROBATCH   vocab_limit: $VOCAB_LIMIT
stop tokens: checkpoint generation_config (151643, 151645)   early stop: on
recipe: greedy T=0, logp confidence, max_candidates=$BLOCK
model: $SDAR_MODEL
GPUs: ${GPUS[*]} (one process per card)
output: $OUTPUT_ROOT
INFO
cat "$LOG_ROOT/RUNINFO.txt"

run_worker() {
    local worker="$1" gpu="$2"
    while read -r index bench methods; do
        [ "$index" = "$worker" ] || continue
        echo "[GPU $gpu] $bench (canvas $(canvas_for "$bench")): $methods"
        if [ "${DRY_RUN:-0}" = "1" ]; then continue; fi
        # shellcheck disable=SC2086
        EXPERIMENT_PHYSICAL_GPU_ID="$gpu" CUDA_VISIBLE_DEVICES="$gpu" \
        PYTHONPATH="$SDAR_OVERLAY:$PYTHONPATH" \
        "$PYTHON_BIN" -u "$HARNESS_DIR/scripts/run_benchmark.py" \
            --dataset "$DATA_ROOT/$bench/test.jsonl" \
            --output-root "$OUTPUT_ROOT/$bench" \
            --continuation-length "$(canvas_for "$bench")" \
            --verifier-microbatch-size "$MICROBATCH" \
            --methods $methods \
            --model "$SDAR_MODEL" --family sdar --block-length "$BLOCK" \
            --device cuda:0 --dtype bfloat16 --seed "$SEED" \
            --timing-repeats 1 --trace-limit 0 --require-lean-logits \
            --max-candidates "$BLOCK" --sample-size "$TASKS" --local-files-only \
            --vocab-limit "$VOCAB_LIMIT" \
            --fixed-temperature 0.0 --fixed-confidence logp \
            --adaptive-temperature 0.0 --adaptive-confidence logp
    done <"$PLAN"
}

pids=()
for i in "${!GPUS[@]}"; do
    run_worker "$i" "${GPUS[$i]}" >"$LOG_ROOT/gpu${GPUS[$i]}.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" = "0" ] || { echo "ERROR: a worker failed; see $LOG_ROOT" >&2; exit 1; }
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
"$PYTHON_BIN" scripts/compare_exp04.py "$OUTPUT_ROOT" \
    "$PROJECT_DIR/../exp04_autospeculation/outputs/baseline/sdar" | tee "$LOG_ROOT/compare_exp04.txt"
echo "Done: $OUTPUT_ROOT/comparison/summary.json"
