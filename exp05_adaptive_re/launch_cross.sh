#!/usr/bin/env bash
# exp05 main arms: SDAR proposes, Dream-7B verifies (DESIGN.md F1-F4).
#
#   xadaptive-remask:<delta>   Dream-only acceptance ratio, NO veto, one remasking
#                              pass per block (F3) -- the one exp05 variant
#   xadaptive:<delta>          the same without remasking (ablation; not run by default)
#
# delta grid 0.1 / 0.3 / 1 / 3 / 10, log-spaced (decided 2026-09-19). The
# smoke-test traces put the delta needed to accept one more token at
# p25 0.71 / p50 1.35 / p75 3.7 / p90 549, so exp04's 0.1-1.5 grid would have
# missed most of the range where delta acts.
#
# Rules shared with the SDAR baselines (launch_sdar_baselines.sh): early stop,
# stop set {151645, 151643}, shared vocabulary IDs < 151665, canvases GSM8K 256
# / HumanEval+ 512 / MBPP+ 256, greedy logp proposer, block = max_candidates = 64,
# Dream rows in groups of 4 with early exit, seed 271828, 100 tasks.
#
# GPU layout (LAYOUT):
#   exclusive (default)  one worker per PCIe-switch pair: SDAR on one card and
#                        Dream on the other (GPU0+1, GPU2+3). Every card is
#                        used by one process, so timings are uncontended and
#                        comparable with the SDAR baselines. 2 workers.
#   shared               Dream alone on GPU0, GPU1, GPU2; the three SDAR copies
#                        (~4.3 GiB each) share GPU3. 3 workers, but SDAR's
#                        share of the latency is contended: fine for pilots and
#                        ablations, NOT for headline tokens/s.
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
SEED="${SEED:-271828}"
LAYOUT="${LAYOUT:-exclusive}"
BENCHMARKS="${BENCHMARKS:-gsm8k humaneval_plus mbpp_plus}"
DELTAS="${DELTAS:-0.10 0.30 1.00 3.00 10.00}"
VARIANTS="${VARIANTS:-xadaptive-remask}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/../data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/cross}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/cross}"
SDAR_MODEL="${SDAR_MODEL:-$PROJECT_DIR/../models/SDAR-1.7B-Chat-b64-local}"
DREAM_MODEL="${DREAM_MODEL:-Dream-org/Dream-v0-Instruct-7B}"
SDAR_OVERLAY="${SDAR_OVERLAY:-$PROJECT_DIR/vendor_env/tf452}"
TRACE_LIMIT="${TRACE_LIMIT:-5}"
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

# worker -> "sdar_gpu dream_gpu"
case "$LAYOUT" in
    exclusive) WORKERS=("0 1" "2 3") ;;
    shared) WORKERS=("3 0" "3 1" "3 2") ;;
    *) echo "ERROR: LAYOUT must be exclusive or shared" >&2; exit 2 ;;
esac

read -r -a BENCH_LIST <<< "$BENCHMARKS"
METHODS=()
for variant in $VARIANTS; do
    for delta in $DELTAS; do METHODS+=("$variant:$delta"); done
done
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

echo "== preconditions =="
[ -d "$SDAR_MODEL" ] || { echo "ERROR: no checkpoint at $SDAR_MODEL" >&2; exit 2; }
[ -d "$SDAR_OVERLAY" ] || { echo "ERROR: no transformers overlay at $SDAR_OVERLAY" >&2; exit 2; }
for bench in "${BENCH_LIST[@]}"; do
    [ -s "$DATA_ROOT/$bench/test.jsonl" ] || { echo "ERROR: missing $DATA_ROOT/$bench/test.jsonl" >&2; exit 2; }
    canvas_for "$bench" >/dev/null
done
for gpu in 0 1 2 3; do
    used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
    if [ "$used" -gt 1000 ]; then
        echo "ERROR: GPU $gpu already has ${used} MiB in use." >&2
        exit 2
    fi
done

# Units are (benchmark, method); dealt round-robin over workers, heaviest
# benchmark first. There is no measured cost table for these arms yet -- run a
# pilot (TASKS=4) and read s/task off its summary before a 100-task sweep.
UNITS=()
for bench in humaneval_plus gsm8k mbpp_plus; do
    [[ " ${BENCH_LIST[*]} " == *" $bench "* ]] || continue
    for method in "${METHODS[@]}"; do UNITS+=("$bench $method"); done
done

cat >"$LOG_ROOT/RUNINFO.txt" <<INFO
exp05 cross-model arms (SDAR proposes, Dream verifies)
started: $(date '+%Y-%m-%d %H:%M:%S %Z')   host: $(hostname)
layout: $LAYOUT   workers (sdar_gpu dream_gpu): $(printf '[%s] ' "${WORKERS[@]}")
methods: ${METHODS[*]}
benchmarks: ${BENCH_LIST[*]} (canvas $(for b in "${BENCH_LIST[@]}"; do printf '%s=%s ' "$b" "$(canvas_for "$b")"; done))
tasks: $TASKS   seed: $SEED   block: $BLOCK   dream group rows: $MICROBATCH   vocab_limit: $VOCAB_LIMIT
stop tokens: SDAR generation_config (151643, 151645)   early stop: on
proposer: $SDAR_MODEL (greedy, logp)
verifier: $DREAM_MODEL (veto off)
output: $OUTPUT_ROOT
INFO
cat "$LOG_ROOT/RUNINFO.txt"

run_worker() {
    local worker="$1" sdar_gpu dream_gpu i
    read -r sdar_gpu dream_gpu <<< "${WORKERS[$worker]}"
    for (( i = worker; i < ${#UNITS[@]}; i += ${#WORKERS[@]} )); do
        read -r bench method <<< "${UNITS[$i]}"
        echo "[SDAR GPU $sdar_gpu / Dream GPU $dream_gpu] $bench (canvas $(canvas_for "$bench")): $method"
        if [ "${DRY_RUN:-0}" = "1" ]; then continue; fi
        # cuda:0 is SDAR's card and cuda:1 Dream's, whatever their physical IDs.
        EXPERIMENT_PHYSICAL_GPU_ID="${sdar_gpu}+${dream_gpu}" \
        CUDA_VISIBLE_DEVICES="$sdar_gpu,$dream_gpu" \
        PYTHONPATH="$SDAR_OVERLAY:$PYTHONPATH" \
        "$PYTHON_BIN" -u scripts/run_benchmark.py \
            --dataset "$DATA_ROOT/$bench/test.jsonl" \
            --output-root "$OUTPUT_ROOT/$bench" \
            --continuation-length "$(canvas_for "$bench")" \
            --verifier-microbatch-size "$MICROBATCH" \
            --methods "$method" \
            --model "$SDAR_MODEL" --family sdar --block-length "$BLOCK" \
            --verifier-model "$DREAM_MODEL" --verifier-family dream \
            --device cuda:0 --verifier-device cuda:1 \
            --dtype bfloat16 --seed "$SEED" \
            --timing-repeats 1 --trace-limit "$TRACE_LIMIT" --require-lean-logits \
            --max-candidates "$BLOCK" --sample-size "$TASKS" --local-files-only \
            --vocab-limit "$VOCAB_LIMIT" \
            --adaptive-temperature 0.0 --adaptive-confidence logp
    done
}

pids=()
for worker in "${!WORKERS[@]}"; do
    run_worker "$worker" >"$LOG_ROOT/worker${worker}.log" 2>&1 &
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
            "$PYTHON_BIN" scripts/score_evalplus.py --benchmark "$bench" \
                --parallel "$EVALPLUS_WORKERS" "$method_dir"
        fi
    done
done
"$PYTHON_BIN" scripts/summarize_results.py --output-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/comparison"
echo "Done: $OUTPUT_ROOT/comparison/summary.json"
