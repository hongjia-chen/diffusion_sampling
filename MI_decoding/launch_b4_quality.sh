#!/usr/bin/env bash
# B=4 quality anchors: SDAR's published block-4 checkpoints (SDAR-1.7B-Chat,
# SDAR-8B-Chat) decoded at B=4 under SDAR's own OpenCompass protocol, on
# GSM8K (1,319), MATH-500 (500), HumanEval (164) and sanitized MBPP (257).
#
#   fixed:1        static, 1 token/step: the paper's Table 2 setting, the quality target
#   fixed:2, :4    static, 2 / 4 tokens per step (4 = one whole block per step)
#   dynamic:0.95   SDAR low_confidence_dynamic, θ from SDAR's eval config
#   dynamic:0.90   θ from SDAR's README / JetEngine
#
# The two models run at the same time, each through launch_baselines.sh (same
# recipe as every MI_decoding baseline): 1.7B on one card, 8B pipeline-split
# over a pair -- three cards in all, the most the host's sharing policy allows.
# Builds ../models/SDAR-8B-Chat-local first if it is missing (16 GB download).
#
#   bash launch_b4_quality.sh                       # 1.7B on card 1, 8B on cards 2,3
#   SMALL_CARD=0 BIG_PAIR=2,3 bash launch_b4_quality.sh
#   TASKS=100 BENCHMARKS="gsm8k math_500 mbpp_plus humaneval_plus" bash launch_b4_quality.sh   # legacy exp04-06 samples and prompts
#   ONLY=8b bash launch_b4_quality.sh               # one model only (1p7b | 8b)
#   DRY_RUN=1 bash launch_b4_quality.sh
#
# Outputs: outputs/opencompass/SDAR-{1.7B,8B}-Chat/<bench>/<method>/, logs under
# logs/opencompass/. Resumable: rerunning skips finished tasks.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
HARNESS_DIR="$(cd -- "$PROJECT_DIR/../harness" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/diffusion/bin/python}"
MODELS_DIR="$(cd -- "$PROJECT_DIR/../models" && pwd)"
SMALL_CARD="${SMALL_CARD:-1}"
BIG_PAIR="${BIG_PAIR:-2,3}"
ONLY="${ONLY:-both}"
export TASKS="${TASKS:-all}"
export BENCHMARKS="${BENCHMARKS:-gsm8k math_500 humaneval mbpp_sanitized}"
export METHODS="${METHODS:-fixed:1 fixed:2 fixed:4 dynamic:0.95 dynamic:0.90}"

SMALL_MODEL="$MODELS_DIR/SDAR-1.7B-Chat-local"
BIG_MODEL="$MODELS_DIR/SDAR-8B-Chat-local"
BIG_REPO="JetLM/SDAR-8B-Chat"

case "$ONLY" in both|1p7b|8b) ;; *) echo "ERROR: ONLY must be both, 1p7b or 8b" >&2; exit 2 ;; esac
IFS=, read -r -a big_cards <<< "$BIG_PAIR"
if [ "$ONLY" = "both" ] && [[ " ${big_cards[*]} " == *" $SMALL_CARD "* ]]; then
    echo "ERROR: SMALL_CARD $SMALL_CARD is also in BIG_PAIR $BIG_PAIR" >&2
    exit 2
fi

if [ "$ONLY" != "1p7b" ] && [ ! -d "$BIG_MODEL" ]; then
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "would download $BIG_REPO (16 GB) and build $BIG_MODEL"
    else
        echo "== building $BIG_MODEL ($(date '+%H:%M:%S')) =="
        "$PYTHON_BIN" -c "from huggingface_hub import snapshot_download; snapshot_download('$BIG_REPO')"
        "$PYTHON_BIN" "$HARNESS_DIR/scripts/prepare_sdar_local.py" "$BIG_REPO" --output-root "$MODELS_DIR"
    fi
fi

pids=()
if [ "$ONLY" != "8b" ]; then
    MODEL="$SMALL_MODEL" CARDS="$SMALL_CARD" bash launch_baselines.sh &
    pids+=($!)
fi
if [ "$ONLY" != "1p7b" ]; then
    if [ "${DRY_RUN:-0}" = "1" ] && [ ! -d "$BIG_MODEL" ]; then
        echo "(8B plan skipped in dry run: checkpoint not built yet)"
    else
        MODEL="$BIG_MODEL" CARDS="$BIG_PAIR" bash launch_baselines.sh &
        pids+=($!)
    fi
fi
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
