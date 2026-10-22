#!/usr/bin/env bash
# The MI decoding rule (`xadaptive:δ`), ported from exp06's launch_cross8b.sh
# with remasking off by default. One SDAR checkpoint proposes (greedy, logp
# confidence, up to `block` candidates); the other scores them with the
# verifier-only ratio log p_V(y_j | c, y_<j) - log p_V(y_j | c), and the
# longest prefix whose |cumulative ratio| stays within log(1 + δ) is
# committed, at least one token, no veto. GROUP_ROWS verifier rows per call
# (default 4). Same tasks, canvases, stop rules, warmups and timing as
# launch_baselines.sh.
#
# ROLES=8b_proposes (default): SDAR-8B proposes and SDAR-1.7B verifies -- every
# committed token is an 8B greedy proposal; the 1.7B only decides how many
# commit per step (exp06 DESIGN §10a/§10b). ROLES=1p7b_proposes swaps them.
#
#   bash launch_mi.sh                         # refuses if another user's job is on the cards
#   TASKS=200 bash launch_mi.sh                # screening subset (default: full sets)
#   TASKS=250 bash launch_mi.sh                # dev sets: GSM8K/MATH 250 (../data/sample250), HE/MBPP all
#   TASKS=100 BENCHMARKS="gsm8k math_500 mbpp_plus humaneval_plus" bash launch_mi.sh   # legacy exp04-06 samples and prompts
#   TASKS=5 OUTPUT_ROOT=outputs/mi_pilot LOG_ROOT=logs/mi_pilot bash launch_mi.sh
#   DELTAS="0.10 1.00" bash launch_mi.sh
#   PROFILE_STAGES=0 bash launch_mi.sh          # no per-stage latency breakdown
#   VARIANTS="xadaptive xadaptive-remask" bash launch_mi.sh   # exp06's remask arm too
#   VARIANTS="xadaptive xadaptive-plus1" bash launch_mi.sh     # prefix plus one later candidate (cross.py)
#   VARIANTS="xadaptive-dyn95" bash launch_mi.sh               # dynamic 0.95 first, MI rule on the rest (cross.py)
#   VARIANTS="xadaptive-dyn95-band80" bash launch_mi.sh        # 0.95 set minus its last untested, MI rule on that last + the band above 0.80
#   BIG_MODEL=... SMALL_MODEL=... BLOCK=16 bash launch_mi.sh  # other block sizes
#   GROUP_ROWS=8 bash launch_mi.sh             # verifier rows per call (default 4)
#   PAIR=2,3 bash launch_mi.sh
#   PLACEMENT=separate bash launch_mi.sh       # 8B alone on the first card, 1.7B on the second
#   SKIP_SCORING=1 bash launch_mi.sh           # generate only (replicas sharing an OUTPUT_ROOT)
#   SKIP_GENERATION=1 bash launch_mi.sh        # score and summarize only
#   DRY_RUN=1 bash launch_mi.sh
#
# One replica, one card pair. PLACEMENT=split (default): 8B is pipeline-split
# over the pair and 1.7B shares the second card, whichever role each plays.
# PLACEMENT=separate: each model on its own card, for cards that hold the 8B
# alone (A10G 22.5 GiB, EC2 2026-10-17). Resumable: rerunning skips finished tasks.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
# The shared harness (package, runner and scoring scripts) lives one level up.
HARNESS_DIR="$(cd -- "$PROJECT_DIR/../harness" && pwd)"
export PYTHONPATH="$HARNESS_DIR:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/diffusion/bin/python}"
PAIR="${PAIR:-0,1}"
TASKS="${TASKS:-all}"
SEED="${SEED:-271828}"
BENCHMARKS="${BENCHMARKS:-gsm8k math_500 humaneval mbpp_sanitized}"
# Spans ~1.5 to ~7 tokens/step (exp05/06), the range of the dynamic θ grid.
DELTAS="${DELTAS:-0.03 0.10 0.30 1.00 3.00 10.00}"
# Remask arms first: they are the experiment; the counterfactual follows.
VARIANTS="${VARIANTS:-xadaptive}"
TRACE_LIMIT="${TRACE_LIMIT:-5}"
# Per-stage latency (prefill, propose, verify, decide, stop check, commit, and
# the untimed rest) for the first N tasks of each arm, from one extra profiled
# decode per task; the headline latencies stay unprofiled. 0 turns it off.
PROFILE_STAGES="${PROFILE_STAGES:-10}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/../data}"
SMALL_MODEL="${SMALL_MODEL:-$PROJECT_DIR/../models/SDAR-1.7B-Chat-b64-local}"
BIG_MODEL="${BIG_MODEL:-$PROJECT_DIR/../models/SDAR-8B-Chat-b64-local}"
# Which model proposes. With PLACEMENT=split the 8B takes both cards and the
# 1.7B the second; with PLACEMENT=separate the 8B takes the first card alone.
ROLES="${ROLES:-8b_proposes}"
PLACEMENT="${PLACEMENT:-split}"
# One card goes through --device/--verifier-device, as launch_baselines.sh does
# for one card, so the 8B runs the same code path as its single-card baselines.
case "$PLACEMENT" in
    split) BIG_PROPOSER=(--devices cuda:0 cuda:1); BIG_VERIFIER=(--verifier-devices cuda:0 cuda:1) ;;
    separate) BIG_PROPOSER=(--device cuda:0); BIG_VERIFIER=(--verifier-device cuda:0) ;;
    *) echo "ERROR: PLACEMENT must be split or separate" >&2; exit 2 ;;
esac
case "$ROLES" in
    1p7b_proposes)
        PROPOSER_MODEL="$SMALL_MODEL"; PROPOSER_ARGS=(--device cuda:1)
        VERIFIER_MODEL="$BIG_MODEL"; VERIFIER_ARGS=("${BIG_VERIFIER[@]}")
        DEFAULT_NAME=1p7b_proposes ;;
    8b_proposes)
        PROPOSER_MODEL="$BIG_MODEL"; PROPOSER_ARGS=("${BIG_PROPOSER[@]}")
        VERIFIER_MODEL="$SMALL_MODEL"; VERIFIER_ARGS=(--verifier-device cuda:1)
        DEFAULT_NAME=8b_proposes ;;
    *) echo "ERROR: ROLES must be 1p7b_proposes or 8b_proposes" >&2; exit 2 ;;
esac
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/mi/$DEFAULT_NAME}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/mi/$DEFAULT_NAME}"
SDAR_OVERLAY="${SDAR_OVERLAY:-$HARNESS_DIR/vendor_env/tf452}"
EVALPLUS_WORKERS="${EVALPLUS_WORKERS:-4}"
# Both checkpoints must be trained at this block; the harness checks their names.
BLOCK="${BLOCK:-64}"
# Verifier rows per call (dream_verify's group). 4 for the B=64 run (4 x 64 =
# 256 tokens per call); bench_verifier_rows.py measures the cost per call vs
# rows and context. It moves fp16 numerics, so keep it fixed across arms.
# Only --verifier-group-rows reaches the MI rule (before 2026-10-15 this went
# to --verifier-microbatch-size alone and the group was always 4).
GROUP_ROWS="${GROUP_ROWS:-4}"
# SDAR's eval config: fp16 weights, logits over the whole vocabulary (151936
# = the lm_head width, so the limit is a no-op; only the mask token is barred).
DTYPE="${DTYPE:-float16}"
VOCAB_LIMIT=151936

# Same as launch_baselines.sh: longer than exp05/06, see there.
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

METHODS=()
for variant in $VARIANTS; do
    for delta in $DELTAS; do METHODS+=("$variant:$delta"); done
done

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
echo "== preconditions =="
for path in "$PROPOSER_MODEL" "$VERIFIER_MODEL" "$SDAR_OVERLAY"; do
    [ -d "$path" ] || { echo "ERROR: missing $path" >&2; exit 2; }
done
for bench in $BENCHMARKS; do
    [ -s "$(data_for "$bench")" ] || { echo "ERROR: missing $(data_for "$bench")" >&2; exit 2; }
    canvas_for "$bench" >/dev/null
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
MI_decoding: MI rule (xadaptive), roles $ROLES
started: $(date '+%Y-%m-%d %H:%M:%S %Z')   host: $(hostname)   cards: $PAIR   placement: $PLACEMENT   contention guard: ${CONTENTION_GUARD:-wait}
methods: ${METHODS[*]}
benchmarks: $BENCHMARKS (canvas $(for b in $BENCHMARKS; do printf '%s=%s ' "$b" "$(canvas_for "$b")"; done))
tasks: $TASKS   seed: $SEED   block: $BLOCK   verifier group rows: $GROUP_ROWS   vocab_limit: $VOCAB_LIMIT   dtype: $DTYPE   protocol: OpenCompass
traced tasks per arm: $TRACE_LIMIT   stage-profiled tasks per arm: $PROFILE_STAGES
data: $(dirname -- "$(data_for gsm8k)")
proposer: $PROPOSER_MODEL (${PROPOSER_ARGS[*]}, greedy, logp)   verifier: $VERIFIER_MODEL (${VERIFIER_ARGS[*]}, veto off)
output: $OUTPUT_ROOT
INFO
cat "$LOG_ROOT/RUNINFO.txt"
[ "${DRY_RUN:-0}" = "1" ] && exit 0

# SKIP_GENERATION=1 / SKIP_SCORING=1: replicas that share one OUTPUT_ROOT
# generate with SKIP_SCORING=1 and are scored once at the end. Scoring while
# another replica still writes would grade a partial records.jsonl.
[ "${SKIP_GENERATION:-0}" = "1" ] && BENCH_TO_RUN="" || BENCH_TO_RUN="$BENCHMARKS"
for bench in $BENCH_TO_RUN; do
    echo "== $bench ($(date '+%H:%M:%S')) =="
    EXPERIMENT_PHYSICAL_GPU_ID="${PAIR//,/+}" CUDA_VISIBLE_DEVICES="$PAIR" \
    PYTHONPATH="$SDAR_OVERLAY:$PYTHONPATH" \
    "$PYTHON_BIN" -u "$HARNESS_DIR/scripts/run_benchmark.py" \
        --dataset "$(data_for "$bench")" \
        --output-root "$OUTPUT_ROOT/$bench" \
        --continuation-length "$(canvas_for "$bench")" \
        --verifier-microbatch-size "$GROUP_ROWS" --verifier-group-rows "$GROUP_ROWS" \
        --model "$PROPOSER_MODEL" --family sdar --block-length "$BLOCK" "${PROPOSER_ARGS[@]}" \
        --verifier-model "$VERIFIER_MODEL" --verifier-family sdar "${VERIFIER_ARGS[@]}" \
        --dtype "$DTYPE" --seed "$SEED" \
        --timing-repeats 1 --warmup-runs 3 --trace-limit "$TRACE_LIMIT" --require-lean-logits \
        --profile-stages "$PROFILE_STAGES" \
        --max-candidates "$BLOCK" "${SAMPLE_ARGS[@]}" --local-files-only \
        --vocab-limit "$VOCAB_LIMIT" \
        --adaptive-temperature 0.0 --adaptive-confidence logp \
        --contention-guard "${CONTENTION_GUARD:-wait}" \
        --methods "${METHODS[@]}" \
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
