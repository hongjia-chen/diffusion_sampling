#!/usr/bin/env bash
# Baseline run: the fixed-k frontier for both models.
#
#   Phase 1  Dream-7B-Instruct on the two NEW benchmarks (HumanEval+, MATH-500),
#            fixed:{1,2,4,8} under the Dream/ADJUST instruct recipe. Extends the
#            ported GSM8K/MBPP+ reference frontier to all four benchmarks.
#
#   Phase 2  SDAR-1.7B-Chat-b64 on ALL FOUR benchmarks, fixed:{1,2,4,8} under
#            SDAR's own published recipe. The block-loop harness it needs
#            (DESIGN.md D8) now exists -- autospec/blockwise.py, with the
#            "sdar" family in autospec/model.py -- so this phase RUNS. The
#            guard below is kept: if either piece is missing the phase skips
#            rather than producing fluent nonsense from a wrong attention mask.
#
# Phase 1 is ~15.5 GPU-h, ~4.3h on four GPUs.
# Phase 2 is ~3.5 GPU-h, ~1.1h on four GPUs (one benchmark per card), measured
# from a 4-task smoke: per task, summed over fixed:1/2/4/8, gsm8k 19.4s and
# ~36s each for the three canvas-512 benchmarks. Peak GPU memory is ~4.0 GiB.
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
GPU_IDS="${GPU_IDS:-0 1 2 3}"
SEED="${SEED:-271828}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/baseline}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/baseline}"
EVALPLUS_WORKERS="${EVALPLUS_WORKERS:-4}"
# Where the extended reference frontier is written. Defaults to the canonical
# path, so a SMOKE TEST MUST OVERRIDE IT -- otherwise a two-task run silently
# replaces the frontier every real comparison is measured against.
FRONTIER_OUT="${FRONTIER_OUT:-$PROJECT_DIR/reference/dream7b_frontier.json}"
MAX_CANDIDATES="${MAX_CANDIDATES:-64}"
# The GSM8K/MBPP+ half of the frontier, ported from the Dream ablation. NOT the
# ablation's own outputs/v2: those were scored before the GSM8K extractor was
# fixed to skip a trailing non-numeric match (`The answer is $\boxed{15}$.`),
# which moves gsm8k fixed_k_8 from 0.15 to 0.16 and leaves all 35 other arms
# bit-identical. This root is that run re-scored with the current scorer, so
# both halves of the frontier are measured by the same code as the SDAR arms.
PORTED_SUMMARY="${PORTED_SUMMARY:-$PROJECT_DIR/outputs/baseline/dream_v2_rescored/comparison/summary.json}"
KS="${KS:-1 2 4 8}"
PHASE="${PHASE:-both}"           # dream | sdar | both

DREAM_MODEL="${DREAM_MODEL:-Dream-org/Dream-v0-Instruct-7B}"
# SDAR needs two things this environment cannot give it directly:
#   * flash_attn, which its modeling file imports unguarded and which cannot be
#     built here (no nvcc; torch 2.13+cu130 has no wheel); and
#   * transformers >= 4.52.4, which it declares, against the 4.46.2 that Dream
#     declares and that phase 1 runs on.
# So it loads from a local patched copy (scripts/prepare_sdar_local.py: two
# minimal patches, both falling back to the authors' own code) with a 20 MB
# transformers overlay on PYTHONPATH. The base environment stays at 4.46.2.
SDAR_REPO="${SDAR_REPO:-JetLM/SDAR-1.7B-Chat-b64}"
SDAR_MODEL="${SDAR_MODEL:-$PROJECT_DIR/models/SDAR-1.7B-Chat-b64-local}"
SDAR_OVERLAY="${SDAR_OVERLAY:-$PROJECT_DIR/vendor_env/tf452}"
# MUST match the checkpoint. SDAR ships one block length per repo (-b8 ... -b64)
# and a mismatch produces an attention pattern the weights never saw.
SDAR_BLOCK_LENGTH="${SDAR_BLOCK_LENGTH:-64}"

# Canvas per benchmark. GSM8K 256 and MBPP+ 512 are fixed by the port: changing
# either invalidates the ported latency numbers they are compared against.
LENGTH_GSM8K="${LENGTH_GSM8K:-256}"
LENGTH_MBPP_PLUS="${LENGTH_MBPP_PLUS:-512}"
LENGTH_HUMANEVAL_PLUS="${LENGTH_HUMANEVAL_PLUS:-512}"
LENGTH_MATH_500="${LENGTH_MATH_500:-512}"
# fixed:k never runs a verifier, so this is unused by every method here. It is
# passed only because run_benchmark.py requires a value.
MICROBATCH="${MICROBATCH:-4}"

read -r -a GPUS <<< "$GPU_IDS"
read -r -a K_LIST <<< "$KS"
if [ "${#GPUS[@]}" -lt 1 ]; then echo "ERROR: GPU_IDS is empty." >&2; exit 2; fi
if ! [ "$TASKS" -ge 1 ] 2>/dev/null; then
    echo "ERROR: TASKS must be a positive integer, got '$TASKS'" >&2; exit 2
fi
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

# The run itself passes --local-files-only so a worker never stalls on the
# network mid-job. That only works if the weights are already there, so fetch
# them up front. Idempotent: a cached repo is a no-op.
fetch_model() {
    local repo="$1"
    if [ "${SKIP_FETCH:-0}" = "1" ]; then return; fi
    "$PYTHON_BIN" - "$repo" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
try:
    path = snapshot_download(repo, local_files_only=True)
    print(f"  {repo}: already cached")
except Exception:
    print(f"  {repo}: downloading (first use)...")
    path = snapshot_download(repo)
    print(f"  {repo}: fetched to {path}")
PY
}

require_dataset() {
    if [ ! -s "$DATA_ROOT/$1/test.jsonl" ]; then
        echo "ERROR: missing $DATA_ROOT/$1/test.jsonl -- run scripts/prepare_all.sh" >&2
        exit 2
    fi
}

# The ported task pairing is the whole basis of the Dream comparison. If those
# files have been regenerated the sha256 will not match what the Dream run
# recorded, and every quality number would be compared against a frontier
# measured on a different 100 tasks.
verify_ported() {
    local name="$1"
    local marker="$DATA_ROOT/$name/PORTED_FROM.txt"
    [ -f "$marker" ] || return 0
    local recorded actual
    recorded="$(sed -n 's/^dataset_sha256 //p' "$marker")"
    actual="$("$PYTHON_BIN" - "$DATA_ROOT/$name/test.jsonl" <<'PY'
import hashlib, sys
d = hashlib.sha256()
with open(sys.argv[1], "rb") as h:
    for c in iter(lambda: h.read(1 << 20), b""):
        d.update(c)
print(d.hexdigest())
PY
)"
    if [ "$recorded" != "$actual" ]; then
        echo "ERROR: $name/test.jsonl no longer matches the Dream run it was ported from." >&2
        echo "       recorded $recorded" >&2
        echo "       actual   $actual" >&2
        echo "       The task pairing is broken; re-port it before running." >&2
        exit 2
    fi
    echo "  $name: ported task pairing verified (${actual:0:16})"
}

launch() {
    # launch <tag> <model> <family> <recipe> <assignment...>
    #
    # Assignments are distributed round-robin over GPU_IDS and every assignment
    # landing on the same GPU is run SEQUENTIALLY inside that GPU's single
    # subshell. Spawning one subshell per assignment would put several workers
    # on one card whenever there are more assignments than GPUs, and each one
    # loads a full copy of the weights -- with Dream-7B at ~14.2 GiB that is an
    # immediate OOM, not a slowdown.
    local tag="$1" model="$2" family="$3" recipe_csv="$4"; shift 4
    local root="$OUTPUT_ROOT/$tag"
    local -a recipe; IFS=' ' read -r -a recipe <<< "$recipe_csv"
    local -a assignments=("$@")
    local extra_flags="" worker_pythonpath="$PYTHONPATH"
    if [ "$family" = "sdar" ]; then
        extra_flags="--block-length $SDAR_BLOCK_LENGTH"
        worker_pythonpath="$SDAR_OVERLAY:$PYTHONPATH"
    fi

    # Bucket assignments per GPU, joining with a newline.
    local -a buckets=()
    local i
    for (( i = 0; i < ${#GPUS[@]}; i++ )); do buckets[i]=""; done
    for (( i = 0; i < ${#assignments[@]}; i++ )); do
        local slot=$(( i % ${#GPUS[@]} ))
        buckets[slot]="${buckets[slot]}${assignments[i]}"$'\n'
    done

    local pids=() gpu
    rm -f "$LOG_ROOT/$tag"_*.pid
    for (( i = 0; i < ${#GPUS[@]}; i++ )); do
        [ -n "${buckets[i]}" ] || continue
        gpu="${GPUS[i]}"
        (
            while IFS= read -r assignment; do
                [ -n "$assignment" ] || continue
                IFS='|' read -r -a units <<< "$assignment"
                for unit in "${units[@]}"; do
                    bench="${unit%%:*}"; rest="${unit#*:}"
                    length="${rest%%:*}"; methods="${rest#*:}"
                    echo "[GPU $gpu] $tag $bench (canvas $length): $methods"
                    if [ "${DRY_RUN:-0}" = "1" ]; then
                        # shellcheck disable=SC2086
                        echo "      CUDA_VISIBLE_DEVICES=$gpu run_benchmark.py" \
                            "--dataset $DATA_ROOT/$bench/test.jsonl" \
                            "--output-root $root/$bench --continuation-length $length" \
                            "--methods $methods --model $model --family $family $extra_flags" \
                            "PYTHONPATH=$worker_pythonpath" \
                            "--seed $SEED --sample-size $TASKS ${recipe[*]}"
                        continue
                    fi
                    # shellcheck disable=SC2086
                    EXPERIMENT_PHYSICAL_GPU_ID="$gpu" CUDA_VISIBLE_DEVICES="$gpu" \
                    PYTHONPATH="$worker_pythonpath" \
                    "$PYTHON_BIN" -u scripts/run_benchmark.py \
                        --dataset "$DATA_ROOT/$bench/test.jsonl" \
                        --output-root "$root/$bench" \
                        --continuation-length "$length" \
                        --verifier-microbatch-size "$MICROBATCH" \
                        --methods $methods \
                        --model "$model" --family "$family" $extra_flags \
                        --device cuda:0 --dtype bfloat16 --seed "$SEED" \
                        --timing-repeats 1 --trace-limit 0 --require-lean-logits \
                        --max-candidates "$MAX_CANDIDATES" --sample-size "$TASKS" \
                        --local-files-only "${recipe[@]}"
                done
            done <<< "${buckets[i]}"
        ) >"$LOG_ROOT/${tag}_gpu${gpu}.log" 2>&1 &
        pids+=("$!")
        printf '%s\n' "$!" >"$LOG_ROOT/${tag}_gpu${gpu}.pid"
    done

    local failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    rm -f "$LOG_ROOT/$tag"_*.pid
    return "$failed"
}

# The host is shared. Dream-7B needs ~14.2 GiB of weights alone, so a card with
# a few GiB already in use will OOM during `model.to(device)` -- after the
# launcher has reported success on every earlier check. Refuse up front.
require_free_memory() {
    local need_mib="$1"; shift
    local usable=() gpu free
    for gpu in "${GPUS[@]}"; do
        free="$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0)"
        if [ "${free:-0}" -ge "$need_mib" ]; then
            usable+=("$gpu")
            echo "  GPU $gpu: ${free} MiB free -- ok"
        else
            echo "  GPU $gpu: ${free} MiB free -- SKIPPED, needs ${need_mib} MiB"
        fi
    done
    if [ "${#usable[@]}" -eq 0 ]; then
        echo "ERROR: no GPU has ${need_mib} MiB free. Wait, or set GPU_IDS." >&2
        exit 2
    fi
    GPUS=("${usable[@]}")
    echo "  using GPUs: ${GPUS[*]}"
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

METHODS="$(printf 'fixed:%s ' "${K_LIST[@]}")"

cat >"$LOG_ROOT/RUNINFO.txt" <<INFO
Baseline fixed-k frontier
started: $(date '+%Y-%m-%d %H:%M:%S %Z')
host: $(hostname)
tasks per benchmark: $TASKS   seed: $SEED
methods: $METHODS   (phase 1 is always fixed:1/2/4/8; KS applies to phase 2)
max candidates: $MAX_CANDIDATES
GPUs: $GPU_IDS
phase: $PHASE
dream model: $DREAM_MODEL   recipe: entropy, T=0.1, top_p=0.9
sdar model:  $SDAR_MODEL  block $SDAR_BLOCK_LENGTH
             recipe: greedy, logp = SDAR's published greedy static low-confidence remasking
canvas: gsm8k=$LENGTH_GSM8K mbpp_plus=$LENGTH_MBPP_PLUS humaneval_plus=$LENGTH_HUMANEVAL_PLUS math_500=$LENGTH_MATH_500
output: $OUTPUT_ROOT
INFO
cat "$LOG_ROOT/RUNINFO.txt"

echo
echo "== dataset checks =="
for b in gsm8k mbpp_plus humaneval_plus math_500; do require_dataset "$b"; done
verify_ported gsm8k
verify_ported mbpp_plus
echo "  humaneval_plus, math_500: freshly frozen"

# ----------------------------------------------------------------- phase 1
if [ "$PHASE" = "dream" ] || [ "$PHASE" = "both" ]; then
    echo
    echo "== phase 1: Dream-7B on HumanEval+ and MATH-500 =="
    # Phase 1's assignment below is a longest-processing-time packing over the
    # MEASURED per-method costs of exactly fixed:1/2/4/8. Any other KS would be
    # silently mispacked, so refuse rather than run an unbalanced job.
    if [ "$KS" != "1 2 4 8" ]; then
        echo "ERROR: phase 1's packing is tuned for KS='1 2 4 8', got '$KS'." >&2
        echo "       Use PHASE=sdar to vary KS, or repack phase 1." >&2
        exit 2
    fi
    fetch_model "$DREAM_MODEL"
    require_free_memory "${DREAM_FREE_MIB:-15000}"
    echo "   recipe: entropy / T=0.1 / top_p=0.9 -- the Dream/ADJUST instruct"
    echo "   recipe the ported GSM8K and MBPP+ frontier was measured under."
    # Longest-processing-time packing over measured per-method costs scaled from
    # MBPP+ (fixed:1/2/4/8 = 133.8/69.2/34.0/16.7 s/task at canvas 512).
    # Estimated loads: 152.9 / 143.3 / 133.4 / 132.1 s/task -> balanced to 14%.
    launch dream "$DREAM_MODEL" dream \
        "--fixed-temperature 0.1 --fixed-top-p 0.9 --fixed-confidence entropy" \
        "math_500:$LENGTH_MATH_500:fixed:1" \
        "humaneval_plus:$LENGTH_HUMANEVAL_PLUS:fixed:1" \
        "math_500:$LENGTH_MATH_500:fixed:2|humaneval_plus:$LENGTH_HUMANEVAL_PLUS:fixed:4 fixed:8" \
        "humaneval_plus:$LENGTH_HUMANEVAL_PLUS:fixed:2|math_500:$LENGTH_MATH_500:fixed:4 fixed:8" \
        || { echo "ERROR: a Dream worker failed; inspect $LOG_ROOT" >&2; exit 1; }
    score_and_summarize "$OUTPUT_ROOT/dream"

    # Extend the reference frontier to all four benchmarks. Refuse when the run
    # is too small to mean anything -- the frontier is what every quality claim
    # is measured against, and a smoke test must not become it.
    if [ "$TASKS" -lt 50 ] && [ "$FRONTIER_OUT" = "$PROJECT_DIR/reference/dream7b_frontier.json" ]; then
        echo "  Skipping frontier update: TASKS=$TASKS is a smoke run."
        echo "  Set FRONTIER_OUT to write it somewhere else."
    else
    [ "${DRY_RUN:-0}" = "1" ] || "$PYTHON_BIN" scripts/reference_frontier.py \
        "$PORTED_SUMMARY" \
        "$OUTPUT_ROOT/dream/comparison/summary.json" \
        --temperature 0.1 \
        --benchmarks gsm8k mbpp_plus humaneval_plus math_500 \
        --output "$FRONTIER_OUT"
    fi
    echo "Phase 1 complete: $OUTPUT_ROOT/dream/comparison/summary.json"
fi

# ----------------------------------------------------------------- phase 2
if [ "$PHASE" = "sdar" ] || [ "$PHASE" = "both" ]; then
    echo
    echo "== phase 2: SDAR-1.7B-Chat-b64 on all four benchmarks =="
    fetch_model "$SDAR_REPO"
    if [ ! -d "$SDAR_MODEL" ]; then
        "$PYTHON_BIN" scripts/prepare_sdar_local.py "$SDAR_REPO"
    fi
    if [ ! -d "$SDAR_OVERLAY" ]; then
        echo "ERROR: missing transformers overlay at $SDAR_OVERLAY. Build it with:" >&2
        echo "  pip install --target $SDAR_OVERLAY --no-deps transformers==4.52.4 tokenizers==0.21.1" >&2
        exit 2
    fi
    require_free_memory "${SDAR_FREE_MIB:-6000}"
    if ! PYTHONPATH="$SDAR_OVERLAY:$PYTHONPATH" "$PYTHON_BIN" -c "
from autospec.model import FAMILIES
from autospec.blockwise import generate_blockwise
raise SystemExit(0 if 'sdar' in FAMILIES else 1)" 2>/dev/null; then
        cat >&2 <<'BLOCKED'
  SKIPPED. autospec/model.py has no "sdar" family yet, so this phase cannot run.

  SDAR needs two things the harness does not have (DESIGN.md D8):
    1. AutoModelForCausalLM -- AutoModel resolves to SDARModel, the backbone,
       which has no lm_head.
    2. A block-causal attention mask, built by the caller. modeling_sdar.py has
       _update_causal_mask commented out and passes is_causal=False, so the
       attention pattern is entirely whatever mask we supply. An all-ones mask
       gives full bidirectional attention over the whole canvas, which SDAR was
       never trained with: it would still emit fluent text while its conditional
       marginals stopped being mutually consistent.

  Running it anyway would produce plausible-looking numbers that mean nothing,
  which is why this exits rather than falling back.
BLOCKED
        [ "$PHASE" = "sdar" ] && exit 3
    else
        # SDAR's published protocol: greedy static low-confidence remasking,
        # which is exactly fixed:k with confidence=logp at temperature 0.
        launch sdar "$SDAR_MODEL" sdar \
            "--fixed-temperature 0.0 --fixed-confidence logp" \
            "gsm8k:$LENGTH_GSM8K:$METHODS" \
            "math_500:$LENGTH_MATH_500:$METHODS" \
            "humaneval_plus:$LENGTH_HUMANEVAL_PLUS:$METHODS" \
            "mbpp_plus:$LENGTH_MBPP_PLUS:$METHODS" \
            || { echo "ERROR: an SDAR worker failed; inspect $LOG_ROOT" >&2; exit 1; }
        score_and_summarize "$OUTPUT_ROOT/sdar"
        echo "Phase 2 complete: $OUTPUT_ROOT/sdar/comparison/summary.json"
    fi
fi

echo
echo "Done. Roots under $OUTPUT_ROOT"
