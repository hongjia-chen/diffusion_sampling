#!/usr/bin/env bash
# One checkpoint's baselines as (benchmark x method) cells spread over several
# cards: each card pulls the next cell from a shared queue, in the order CELLS
# lists them (longest first balances the cards), and the run is scored once
# when every cell is done. Each cell is an ordinary launch_baselines.sh run on
# one card with SKIP_SCORING=1, so the protocol (prompts, canvas, warmups,
# timing, contention guard) is unchanged; the cells share OUTPUT_ROOT and each
# method writes its own directory. Per-task timings stay single-card: a cell
# owns its card, and the guard only watches the card it times on.
#
# Billing is per instance-hour, so on a multi-GPU instance this costs about
# 1/N of running the same cells one after another on one card.
#
#   MODEL=../models/SDAR-8B-Chat-b16-local RUN_NAME=SDAR-8B-Chat-b16-ec2 CARDS=0,1,2,3 \
#     CELLS="gsm8k:fixed:8 math_500:fixed:8 gsm8k:fixed:1" bash run_cells.sh
#   DRY_RUN=1 ... bash run_cells.sh      # print the queue and dry-run the first cell
#
# Resumable: rerunning rebuilds the queue and every cell skips finished tasks.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
: "${MODEL:?set MODEL}" "${RUN_NAME:?set RUN_NAME}" "${CELLS:?set CELLS (bench:method ...)}"
CARDS="${CARDS:-0,1,2,3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/opencompass/$RUN_NAME}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/opencompass/$RUN_NAME}"
QUEUE="$LOG_ROOT/queue.txt"
LOCK="$LOG_ROOT/queue.lock"
CELL_LOG="$LOG_ROOT/cells.log"
IFS=, read -r -a CARD_LIST <<< "$CARDS"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT/cells"
tr ' ' '\n' <<< "$CELLS" | sed '/^$/d' > "$QUEUE"
echo "== $(wc -l < "$QUEUE") cells on cards $CARDS, output $OUTPUT_ROOT"
cat -n "$QUEUE"

cell_tag() { local bench="${1%%:*}" method="${1#*:}"; echo "${bench}__${method//[:.]/_}"; }
run_cell() {  # card cell [extra env...]
    local card="$1" cell="$2"; shift 2
    env "$@" MODEL="$MODEL" RUN_NAME="$RUN_NAME" CARDS="$card" \
        BENCHMARKS="${cell%%:*}" METHODS="${cell#*:}" \
        OUTPUT_ROOT="$OUTPUT_ROOT" LOG_ROOT="$LOG_ROOT/cells/$(cell_tag "$cell")" \
        SKIP_SCORING=1 bash "$PROJECT_DIR/launch_baselines.sh"
}

if [ "${DRY_RUN:-0}" = "1" ]; then
    run_cell "${CARD_LIST[0]}" "$(head -n1 "$QUEUE")" DRY_RUN=1
    exit 0
fi

# Pop the queue's first line under a lock shared by the card workers.
next_cell() {
    (
        flock 9
        head -n1 "$QUEUE"
        tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
    ) 9>"$LOCK"
}
worker() {
    local card="$1" cell status
    while cell="$(next_cell)"; [ -n "$cell" ]; do
        echo "$(date -u '+%m-%d %H:%M:%S') card $card start $cell" >> "$CELL_LOG"
        if run_cell "$card" "$cell" > "$LOG_ROOT/cells/$(cell_tag "$cell").out" 2>&1; then
            status=ok
        else
            status="FAILED($?)"
        fi
        echo "$(date -u '+%m-%d %H:%M:%S') card $card end   $cell $status" >> "$CELL_LOG"
    done
}

for card in "${CARD_LIST[@]}"; do worker "$card" & done
wait

if grep -q FAILED "$CELL_LOG"; then
    echo "ERROR: failed cells (see $CELL_LOG); not scoring. Rerun to resume them." >&2
    grep FAILED "$CELL_LOG" >&2
    exit 1
fi
echo "== all cells done ($(date -u '+%m-%d %H:%M:%S')); scoring"
env MODEL="$MODEL" RUN_NAME="$RUN_NAME" CARDS="${CARD_LIST[0]}" METHODS="$(tr ' ' '\n' <<< "$CELLS" | sed -n 's/^[^:]*://p' | sort -u | paste -sd' ')" \
    OUTPUT_ROOT="$OUTPUT_ROOT" LOG_ROOT="$LOG_ROOT" SKIP_GENERATION=1 \
    bash "$PROJECT_DIR/launch_baselines.sh"
