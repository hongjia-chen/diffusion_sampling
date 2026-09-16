#!/usr/bin/env bash
# The adaptive delta sweep and Algorithm 3, on SDAR-1.7B-Chat-b64.
#
# This is phase 3. Phase 2 (dream_SDAR_baseline_launch.sh) measured the fixed:k
# frontier for the same checkpoint on the same 100 tasks; this adds the two
# arms that frontier exists to be compared against:
#
#   adaptive:delta   the same nu as Algorithm 3, accepted by threshold rather
#                    than by coin. Confidence-ordered, greedy -- byte-identical
#                    scheduler and recipe to the fixed:k arms, so a difference
#                    between them is the acceptance rule and nothing else.
#   autospec:rho     Algorithm 3. Forced to T=1.0 with no top-p and a uniform
#                    random sigma (DESIGN.md D1, D2), so it is NOT recipe-matched
#                    to the other two. That confound is stated, not engineered
#                    around; PHASE=t1ref below prices half of it.
#
# Results land in the SAME output root as the phase-2 fixed:k arms, so
# scripts/summarize_results.py produces one table with all three schedulers.
# Existing fixed_k_* directories are never touched.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
# Generation is bit-reproducible only with this set; nothing in the code
# enforces it, so it is exported here and not left to the caller.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "${HOME}/miniconda3/envs/diffusion/bin/python" ]; then
        PYTHON_BIN="${HOME}/miniconda3/envs/diffusion/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

TASKS="${TASKS:-100}"
GPU_IDS="${GPU_IDS:-0 1}"
SEED="${SEED:-271828}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/baseline/sdar}"
# MATH-500 is the one benchmark whose fixed:k frontier does not live in
# $OUTPUT_ROOT. At canvas 512 it truncated 46% of fixed:1 tasks and 88% of
# fixed:8, so the quality axis was measuring the canvas; the canvas-1024 re-run
# put fixed:1 at 0.43 against 0.36 and dropped maximum_length_rate to 0.16.
# That re-run went to its own root, because changing the canvas changes the
# latency and the two are not interchangeable -- so the MATH-500 sweep follows
# it there, and lands beside the frontier it is actually comparable to.
OUTPUT_ROOT_MATH_500="${OUTPUT_ROOT_MATH_500:-$PROJECT_DIR/outputs/baseline/sdar_math1024}"
# Pairing metadata can live separately from the new results. This is what lets
# an autospec-only corrective run preserve the preliminary capped outputs while
# still adopting the exact baseline canvases and validating seed/config parity.
PAIRING_ROOT="${PAIRING_ROOT:-$PROJECT_DIR/outputs/baseline/sdar}"
PAIRING_ROOT_MATH_500="${PAIRING_ROOT_MATH_500:-$PROJECT_DIR/outputs/baseline/sdar_math1024}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/sweep}"
EVALPLUS_WORKERS="${EVALPLUS_WORKERS:-4}"
COSTS="${COSTS:-$PROJECT_DIR/reference/sdar_b64_costs.json}"
PHASE="${PHASE:-sweep}"           # sweep | autospec | t1ref | both
BENCHMARKS="${BENCHMARKS:-gsm8k math_500 humaneval_plus mbpp_plus}"

# The block size IS the commitment cap under SDAR: no scheduler can commit more
# than one block, because a block is all there is. 64 both matches the prior
# Dream run's cap and is the largest block SDAR ships (DESIGN.md D8). Keep
# MAX_CANDIDATES == SDAR_BLOCK_LENGTH.
#
# D8 predicted b64 would leave the delta sweep uncensored, from Dream's
# commitment widths. On a 4-task pilot that holds for GSM8K (commit 4.8 at
# delta 0.20 rising to 7.7 at delta 1.00, 2-4% of steps at the cap) and does
# NOT hold for MBPP+ (commit 18.1 -> 31.7 over delta 0.10 -> 1.50, with 21% to
# 43% of steps committing the whole block). Read `at_cap_rate` per benchmark
# before reading the top of the sweep as a delta effect on the code arms.
MAX_CANDIDATES="${MAX_CANDIDATES:-64}"
SDAR_BLOCK_LENGTH="${SDAR_BLOCK_LENGTH:-64}"

SDAR_MODEL="${SDAR_MODEL:-$PROJECT_DIR/models/SDAR-1.7B-Chat-b64-local}"
SDAR_OVERLAY="${SDAR_OVERLAY:-$PROJECT_DIR/vendor_env/tf452}"

# ---- the sweep --------------------------------------------------------------
# Five deltas, a subset of the eight the Dream ablation swept (0.10 0.15 0.20
# 0.30 0.50 0.75 1.00 1.50), so every point here pairs with a measured Dream
# point on the same 100 GSM8K/MBPP+ tasks.
DELTAS="${DELTAS:-0.10 0.20 0.50 1.00 1.50}"
# rho is the one knob the paper leaves free (D6), and on this checkpoint it is
# not a second-order one. It sets how fast the rejection loop's batch grows, so
# it trades oracle ROWS for oracle ROUNDS -- and rounds are the D4 headline. At
# the paper's default it is 1/log2(64) = 1/6, which means the loop draws
# candidates nearly one at a time: a 4-task GSM8K pilot spent 140 fallback
# draws per task and 377 physical rounds, against 284 for plain sequential
# ancestral sampling in the same order. rho = 1.0 doubles the batch each round,
# which is the cheapest way to find out how much of that 377 is the rejection
# loop's serialization rather than the model's coupling.
#
# Not in the default sweep, because it doubles the Algorithm 3 budget: autospec
# is 4.0 of the 10.8 GPU-h below, and a second rho arm adds another 4.0. Add it
# deliberately -- RHOS="auto 1.0" -- if the round counts turn out to be the
# result worth explaining.
RHOS="${RHOS:-auto}"

# Verifier microbatch. In exact arithmetic this is a pure speed knob -- the
# accepted prefix does not depend on it, because _score_all_prefixes exits at
# the first violating chunk and a truncated error curve still contains that
# violation. In bf16 it is NOT: a 4-row and an 8-row batched matmul reduce in
# different orders, which moves the log-factors in the last bits, and a
# candidate sitting within that of tau flips. Measured on GSM8K at delta 0.50,
# the same 4 tasks took 46.2 steps at microbatch 8 and 44.2 at microbatch 4 --
# one flipped acceptance early and the trajectories diverge from there.
#
# So this is a setting, not a tuning parameter: pick it once, hold it across
# every arm, and record it. It is picked here, before any score is looked at.
#
# 4, measured. The cost curve is U-shaped and the minimum is not where the
# Dream reasoning put it -- on SDAR a batch-1 block forward is memory-bound on
# weights, so cutting the forward count by scoring more candidates per chunk
# buys nothing and wastes rows. adaptive:0.50, 4 tasks, seconds per task:
#
#   microbatch      2      4      8     16     32
#   GSM8K        8.17   6.90   7.78   9.34  13.34   (rows  346/390/484/680/1071)
#   MBPP+       12.19   9.31   9.10   9.23  10.04
#
# 2 loses to per-forward overhead, 16 and up lose to wasted rows. 4 is the
# minimum on GSM8K (and on HumanEval+ and MATH-500, which share its narrow
# commitment widths) and within 2% of the minimum on MBPP+.
MICROBATCH="${MICROBATCH:-4}"

# Algorithm 3's physical batching budget, and under block diffusion it is
# INERT. Measured: the widest pooled request the recursion ever builds is
# exactly B rows -- the root block's conditional probe, one row per in-block
# position. Over three GSM8K tasks, 701 requests, the width histogram tops out
# at 64 (12 requests) with 355 of them a single row. The rejection loop never
# pools past that either: at the paper's rho = 1/log2(64) = 1/6 the batch
# schedule draws one candidate at a time.
#
# So any value >= B behaves identically, and B is the value that says so. It
# is NOT the canvas: `parse_method` defaults this to `continuation_length`,
# which is D5's full-canvas-dLLM reasoning ("size the canvas so the root block
# is one physical round"). Under SDAR the root block is the BLOCK, not the
# canvas, so a canvas-derived budget is both meaningless and different per
# benchmark. Below B it would start chunking the root block and inflate the
# round count with an artifact of the setting.
#
# This is also why `oracle_rounds_physical` equalled the theoretical count in
# every pilot: at b64 nothing is ever chunked, so the two were identically
# equal and the theoretical counter has been dropped.
AUTOSPEC_MAX_BATCH_ROWS="${AUTOSPEC_MAX_BATCH_ROWS:-$SDAR_BLOCK_LENGTH}"

# Canvas per benchmark. NOT a free parameter: it must equal the canvas the
# fixed:k arms in $OUTPUT_ROOT were measured at, or the sweep is compared
# against a frontier that had a different amount of room. So the default is
# READ OUT of the baseline's own metadata rather than written down here --
# whichever canvas MATH-500 ends up on, the sweep follows it automatically.
# Setting one of these explicitly is an override, and it still has to agree
# with the baseline; the check below refuses otherwise.
#
# The reason MATH-500's canvas is live at all: at 512 the phase-2 arms hit the
# canvas limit on 46% (fixed:1) to 88% (fixed:8) of tasks, with 45-93% of
# completions carrying no \boxed{} at all. That measures the canvas, not the
# scheduler, so the sweep should not be run at 512.
LENGTH_GSM8K="${LENGTH_GSM8K:-}"
LENGTH_MATH_500="${LENGTH_MATH_500:-}"
LENGTH_MBPP_PLUS="${LENGTH_MBPP_PLUS:-}"
LENGTH_HUMANEVAL_PLUS="${LENGTH_HUMANEVAL_PLUS:-}"

read -r -a GPUS <<< "$GPU_IDS"
read -r -a BENCH_LIST <<< "$BENCHMARKS"
if [ "${#GPUS[@]}" -lt 1 ]; then echo "ERROR: GPU_IDS is empty." >&2; exit 2; fi
if ! [ "$TASKS" -ge 1 ] 2>/dev/null; then
    echo "ERROR: TASKS must be a positive integer, got '$TASKS'" >&2; exit 2
fi
if [ "$AUTOSPEC_MAX_BATCH_ROWS" -lt "$SDAR_BLOCK_LENGTH" ]; then
    echo "ERROR: AUTOSPEC_MAX_BATCH_ROWS=$AUTOSPEC_MAX_BATCH_ROWS is below the" >&2
    echo "       block length $SDAR_BLOCK_LENGTH, so the root block's probe would be" >&2
    echo "       chunked and oracle_rounds_physical would count the setting" >&2
    echo "       rather than the algorithm. Refusing." >&2
    exit 2
fi
if [ "$MAX_CANDIDATES" -ne "$SDAR_BLOCK_LENGTH" ]; then
    echo "WARNING: MAX_CANDIDATES=$MAX_CANDIDATES != block $SDAR_BLOCK_LENGTH." >&2
    echo "         The scheduler clamps to the block anyway; a smaller value" >&2
    echo "         would censor the top of the delta sweep." >&2
fi
mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT_MATH_500" "$LOG_ROOT"

declare -A CANVAS=()

canvas_override() {
    case "$1" in
        gsm8k) echo "$LENGTH_GSM8K" ;;
        math_500) echo "$LENGTH_MATH_500" ;;
        mbpp_plus) echo "$LENGTH_MBPP_PLUS" ;;
        humaneval_plus) echo "$LENGTH_HUMANEVAL_PLUS" ;;
        *) echo "ERROR: unknown benchmark $1" >&2; exit 2 ;;
    esac
}

canvas_for() {
    if [ -z "${CANVAS[$1]+set}" ]; then
        echo "ERROR: canvas for $1 was never resolved." >&2; exit 2
    fi
    echo "${CANVAS[$1]}"
}

root_for() {
    case "$1" in
        math_500) echo "$OUTPUT_ROOT_MATH_500" ;;
        *) echo "$OUTPUT_ROOT" ;;
    esac
}

pairing_root_for() {
    case "$1" in
        math_500) echo "$PAIRING_ROOT_MATH_500" ;;
        *) echo "$PAIRING_ROOT" ;;
    esac
}

# Every distinct root a selected benchmark writes to, deduplicated -- what has
# to be scored and summarised at the end.
roots_in_use() {
    local bench
    for bench in "${BENCH_LIST[@]}"; do root_for "$bench"; done | sort -u
}

# ------------------------------------------------------------------ checks
echo "== preconditions =="
[ -d "$SDAR_MODEL" ] || {
    echo "ERROR: no patched checkpoint at $SDAR_MODEL." >&2
    echo "       Build it: $PYTHON_BIN scripts/prepare_sdar_local.py JetLM/SDAR-1.7B-Chat-b64" >&2
    exit 2
}
[ -d "$SDAR_OVERLAY" ] || {
    echo "ERROR: missing transformers overlay at $SDAR_OVERLAY. Build it with:" >&2
    echo "  pip install --target $SDAR_OVERLAY --no-deps transformers==4.52.4 tokenizers==0.21.1" >&2
    exit 2
}
PYTHONPATH="$SDAR_OVERLAY:$PYTHONPATH" "$PYTHON_BIN" -c "
from autospec.model import FAMILIES
from autospec.blockwise import generate_blockwise
from autospec.autospeculation import run_autospec_block
raise SystemExit(0 if 'sdar' in FAMILIES else 1)" || {
    echo "ERROR: the sdar family or the block loop is missing; see DESIGN.md D8." >&2
    exit 2
}
for bench in "${BENCH_LIST[@]}"; do
    [ -s "$DATA_ROOT/$bench/test.jsonl" ] || {
        echo "ERROR: missing $DATA_ROOT/$bench/test.jsonl -- run scripts/prepare_all.sh" >&2
        exit 2
    }
done

# The sweep is only meaningful beside a fixed:k frontier measured on the same
# tasks at the same canvas. Refuse to run against a mismatched one rather than
# produce a table whose rows are not comparable.
echo "== pairing with the phase-2 fixed:k arms =="
for bench in "${BENCH_LIST[@]}"; do
    want="$(canvas_override "$bench")"
    meta="$(pairing_root_for "$bench")/$bench/fixed_k_1/metadata.json"
    if [ ! -f "$meta" ]; then
        echo "  $bench: no fixed_k_1 baseline in this root -- the sweep will have" \
             "nothing to be compared against. Set PAIRING_ROOT, or accept it." >&2
        if [ "${ALLOW_UNPAIRED:-0}" != "1" ]; then
            echo "  Refusing. Re-run with ALLOW_UNPAIRED=1 to proceed anyway." >&2
            exit 2
        fi
        if [ -z "$want" ]; then
            echo "  Refusing: with no baseline to read the canvas from, it must be" >&2
            echo "  given explicitly (LENGTH_${bench^^}=...)." >&2
            exit 2
        fi
        CANVAS[$bench]="$want"
        continue
    fi
    have="$("$PYTHON_BIN" -c "
import json,sys
m=json.load(open(sys.argv[1]))
print(m.get('continuation_length'), m.get('block_length'),
      m.get('temperature'), m.get('confidence_metric'),
      m.get('max_candidates'), m.get('base_seed'))" "$meta")"
    read -r have_len have_block have_t have_conf have_cap have_seed <<< "$have"
    fail=0
    if [ -z "$want" ]; then
        want="$have_len"           # adopt the baseline's canvas
    fi
    CANVAS[$bench]="$want"
    [ "$have_len" = "$want" ] || { echo "  $bench: canvas $have_len in the baseline, $want requested" >&2; fail=1; }
    [ "$have_block" = "$SDAR_BLOCK_LENGTH" ] || { echo "  $bench: block $have_block in the baseline, $SDAR_BLOCK_LENGTH requested" >&2; fail=1; }
    [ "$have_seed" = "$SEED" ] || { echo "  $bench: seed $have_seed in the baseline, $SEED requested" >&2; fail=1; }
    [ "$have_cap" = "$MAX_CANDIDATES" ] || { echo "  $bench: max_candidates $have_cap in the baseline, $MAX_CANDIDATES requested" >&2; fail=1; }
    if [ "$fail" = "1" ]; then
        echo "  Refusing: the sweep would not be comparable to the frontier it lands beside." >&2
        exit 2
    fi
    echo "  $bench: canvas $have_len, block $have_block, seed $have_seed," \
         "recipe T=$have_t/$have_conf -- ok  ($(pairing_root_for "$bench"))"
done

# ------------------------------------------------------------------ packing
# Longest-processing-time packing over MEASURED seconds per task. There is no
# default table: these are SDAR-1.7B-b64 on this host, from a 4-task
# calibration, and mean nothing for another checkpoint.
METHODS=()
if [ "$PHASE" != "autospec" ]; then
    for d in $DELTAS; do METHODS+=("adaptive:$d"); done
fi
for r in $RHOS; do METHODS+=("autospec:$r"); done

[ -f "$COSTS" ] || {
    echo "ERROR: no measured cost table at $COSTS." >&2
    echo "       Build one from a short calibration run, then re-launch:" >&2
    echo "         see scripts/pack_work.py's docstring." >&2
    exit 2
}
PLAN="$LOG_ROOT/plan.txt"
"$PYTHON_BIN" scripts/pack_work.py \
    --workers "${#GPUS[@]}" --costs "$COSTS" --tasks "$TASKS" --plan \
    --benchmarks "${BENCH_LIST[@]}" --methods "${METHODS[@]}" >"$PLAN"
echo "== plan =="
cat "$PLAN"

# ------------------------------------------------------------------ memory
require_free_memory() {
    local need_mib="$1" usable=() gpu free
    for gpu in "${GPUS[@]}"; do
        free="$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0)"
        if [ "${free:-0}" -ge "$need_mib" ]; then
            usable+=("$gpu"); echo "  GPU $gpu: ${free} MiB free -- ok"
        else
            echo "  GPU $gpu: ${free} MiB free -- SKIPPED, needs ${need_mib} MiB"
        fi
    done
    if [ "${#usable[@]}" -eq 0 ]; then
        echo "ERROR: no GPU has ${need_mib} MiB free. Wait, or set GPU_IDS." >&2; exit 2
    fi
    if [ "${#usable[@]}" -ne "${#GPUS[@]}" ]; then
        echo "ERROR: the plan above was packed for ${#GPUS[@]} workers but only" >&2
        echo "       ${#usable[@]} GPUs are free. Re-launch with GPU_IDS='${usable[*]}'." >&2
        exit 2
    fi
    echo "  using GPUs: ${GPUS[*]}"
}
echo "== memory =="
# Measured per-process peaks: the adaptive arms sit at 4.0-4.3 GiB on every
# canvas, autospec at 5.6 (canvas 512) to 7.0 GiB (canvas 1024). Autospec's is
# the term that grows with the canvas, because its 64-row block probe
# materialises the expanded KV cache per layer at `B x (prompt + canvas)`,
# where the adaptive arms only ever carry `MICROBATCH` rows. Note this is a
# function of the BLOCK, not of AUTOSPEC_MAX_BATCH_ROWS, which never binds.
# 9 GiB covers the worst of those with room for a long prompt.
require_free_memory "${SDAR_FREE_MIB:-9000}"

# ------------------------------------------------------------------ run
run_worker() {
    local worker="$1" gpu="$2"
    while read -r index bench methods; do
        [ "$index" = "$worker" ] || continue
        local length root
        length="$(canvas_for "$bench")"; root="$(root_for "$bench")"
        echo "[GPU $gpu] $bench (canvas $length) -> $root: $methods"
        if [ "${DRY_RUN:-0}" = "1" ]; then continue; fi
        # shellcheck disable=SC2086
        EXPERIMENT_PHYSICAL_GPU_ID="$gpu" CUDA_VISIBLE_DEVICES="$gpu" \
        PYTHONPATH="$SDAR_OVERLAY:$PYTHONPATH" \
        "$PYTHON_BIN" -u scripts/run_benchmark.py \
            --dataset "$DATA_ROOT/$bench/test.jsonl" \
            --output-root "$root/$bench" \
            --continuation-length "$length" \
            --verifier-microbatch-size "$MICROBATCH" \
            --autospec-max-batch-rows "$AUTOSPEC_MAX_BATCH_ROWS" \
            --methods $methods \
            --model "$SDAR_MODEL" --family sdar --block-length "$SDAR_BLOCK_LENGTH" \
            --device cuda:0 --dtype bfloat16 --seed "$SEED" \
            --timing-repeats 1 --trace-limit 0 --require-lean-logits \
            --max-candidates "$MAX_CANDIDATES" --sample-size "$TASKS" \
            --local-files-only \
            --adaptive-temperature 0.0 --adaptive-confidence logp
    done <"$PLAN"
}

launch_plan() {
    local tag="$1" pids=() index
    rm -f "$LOG_ROOT/$tag"_*.pid
    for index in "${!GPUS[@]}"; do
        run_worker "$index" "${GPUS[$index]}" \
            >"$LOG_ROOT/${tag}_gpu${GPUS[$index]}.log" 2>&1 &
        pids+=("$!")
        printf '%s\n' "$!" >"$LOG_ROOT/${tag}_gpu${GPUS[$index]}.pid"
    done
    printf '%s\n' "$$" >"$LOG_ROOT/launcher.pid"
    local failed=0 pid
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    rm -f "$LOG_ROOT/$tag"_*.pid
    return "$failed"
}

score_and_summarize() {
    local root="$1"
    if [ "${DRY_RUN:-0}" = "1" ]; then echo "  (dry run: skipping scoring of $root)"; return; fi
    for bench in mbpp_plus humaneval_plus; do
        [ -d "$root/$bench" ] || continue
        for method_dir in "$root/$bench"/*; do
            if [ -f "$method_dir/evalplus_samples.jsonl" ] && \
               [ ! -f "$method_dir/records.scored.jsonl" ]; then
                "$PYTHON_BIN" scripts/score_evalplus.py --benchmark "$bench" \
                    --parallel "$EVALPLUS_WORKERS" "$method_dir"
            fi
        done
    done
    "$PYTHON_BIN" scripts/summarize_results.py \
        --output-root "$root" --output-dir "$root/comparison"
}

cat >"$LOG_ROOT/RUNINFO.txt" <<INFO
Adaptive delta sweep + Algorithm 3, SDAR-1.7B-Chat-b64
started: $(date '+%Y-%m-%d %H:%M:%S %Z')
host: $(hostname)
phase: $PHASE
tasks per benchmark: $TASKS   seed: $SEED
benchmarks: $BENCHMARKS
deltas: $DELTAS
rhos: $RHOS
adaptive recipe: greedy T=0.0, confidence logp, confidence-ordered
                 -- identical to the phase-2 fixed:k arms
autospec recipe: T=1.0, no top-p, uniform random sigma (DESIGN.md D1/D2)
                 -- NOT recipe-matched; see the confound note in this script
max candidates: $MAX_CANDIDATES   block: $SDAR_BLOCK_LENGTH
verifier microbatch: $MICROBATCH   autospec batch rows: $AUTOSPEC_MAX_BATCH_ROWS
canvas (adopted from the fixed:k baseline unless overridden):
$(for b in "${BENCH_LIST[@]}"; do printf '  %s=%s\n' "$b" "${CANVAS[$b]}"; done)
model: $SDAR_MODEL
GPUs: $GPU_IDS
output roots:
$(for b in "${BENCH_LIST[@]}"; do printf '  %-16s %s\n' "$b" "$(root_for "$b")"; done)
pairing roots:
$(for b in "${BENCH_LIST[@]}"; do printf '  %-16s %s\n' "$b" "$(pairing_root_for "$b")"; done)
INFO
cat "$LOG_ROOT/RUNINFO.txt"

if [ "$PHASE" = "sweep" ] || [ "$PHASE" = "autospec" ] || [ "$PHASE" = "both" ]; then
    echo
    echo "== $PHASE: ${#METHODS[@]} methods x ${#BENCH_LIST[@]} benchmarks =="
    launch_plan "$PHASE" || { echo "ERROR: a worker failed; inspect $LOG_ROOT" >&2; exit 1; }
    while IFS= read -r root; do
        score_and_summarize "$root"
        echo "Sweep complete: $root/comparison/summary.json"
    done < <(roots_in_use)
fi

# ------------------------------------------------------------------ T=1 ref
# DESIGN.md D7, half of it. Algorithm 3 is forced to T=1.0 while every other
# arm is greedy, so an autospec-vs-fixed quality gap mixes the sampler with the
# decoding recipe. fixed:1 at T=1.0 prices the temperature on its own. It does
# NOT price the other half -- the position order -- because fixed:k always
# ranks by confidence and the harness has no random-sigma fixed arm. That
# confound stays open and stated.
#
# Its own root, so nothing reads the two as one run.
if [ "$PHASE" = "t1ref" ] || [ "$PHASE" = "both" ]; then
    echo
    echo "== T=1.0 reference: fixed:1 and fixed:4, sampled =="
    T1_ROOT="${T1_ROOT:-$PROJECT_DIR/outputs/sdar_t1_reference}"
    mkdir -p "$T1_ROOT"
    cat >"$T1_ROOT/IMPORTED_FROM.txt" <<NOTE
Temperature reference for DESIGN.md D7. fixed:1 and fixed:4 at T=1.0, no top-p,
confidence-ordered, on the same $TASKS tasks and canvases as $OUTPUT_ROOT.
Exists so an autospec-vs-fixed gap can be split into temperature and sampler.
This is NOT part of the greedy frontier and must not be merged into it.
NOTE
    # One subshell per GPU, benchmarks assigned round-robin and run SEQUENTIALLY
    # inside it. Spawning one per benchmark would put two workers on a card
    # whenever there are more benchmarks than GPUs, and each loads its own copy
    # of the weights.
    t1_pids=()
    for slot in "${!GPUS[@]}"; do
        gpu="${GPUS[$slot]}"
        (
            for index in "${!BENCH_LIST[@]}"; do
                [ $(( index % ${#GPUS[@]} )) -eq "$slot" ] || continue
                bench="${BENCH_LIST[$index]}"
                length="$(canvas_for "$bench")"
                echo "[GPU $gpu] $bench (canvas $length): fixed:1 fixed:4 at T=1.0"
                if [ "${DRY_RUN:-0}" = "1" ]; then continue; fi
                EXPERIMENT_PHYSICAL_GPU_ID="$gpu" CUDA_VISIBLE_DEVICES="$gpu" \
                PYTHONPATH="$SDAR_OVERLAY:$PYTHONPATH" \
                "$PYTHON_BIN" -u scripts/run_benchmark.py \
                    --dataset "$DATA_ROOT/$bench/test.jsonl" \
                    --output-root "$T1_ROOT/$bench" \
                    --continuation-length "$length" \
                    --verifier-microbatch-size "$MICROBATCH" \
                    --methods fixed:1 fixed:4 \
                    --model "$SDAR_MODEL" --family sdar --block-length "$SDAR_BLOCK_LENGTH" \
                    --device cuda:0 --dtype bfloat16 --seed "$SEED" \
                    --timing-repeats 1 --trace-limit 0 --require-lean-logits \
                    --max-candidates "$MAX_CANDIDATES" --sample-size "$TASKS" \
                    --local-files-only \
                    --fixed-temperature 1.0 --fixed-confidence logp
            done
        ) >"$LOG_ROOT/t1ref_gpu${gpu}.log" 2>&1 &
        t1_pids+=("$!")
    done
    t1_failed=0
    for pid in "${t1_pids[@]}"; do wait "$pid" || t1_failed=1; done
    [ "$t1_failed" = "0" ] || { echo "ERROR: a T=1 worker failed; inspect $LOG_ROOT" >&2; exit 1; }
    score_and_summarize "$T1_ROOT"
    echo "T=1 reference complete: $T1_ROOT/comparison/summary.json"
fi

echo
echo "Done. Roots:"
roots_in_use | sed 's/^/  /' 
