#!/usr/bin/env bash
## Waits for an idle GPU pair, then starts launch_sdar8b.sh on every idle pair.
##
## A pair is idle when both cards use at most MAX_USED_MIB and at most
## MAX_UTIL_PCT utilization on every poll for QUIET_POLLS polls in a row, so a
## card between two of another user's jobs is not mistaken for free.
##
##   setsid nohup bash scripts/wait_and_launch.sh > logs/sdar8b_wait.log 2>&1 &
##   kill "$(cat logs/sdar8b_wait.pid)"      # give up waiting
set -uo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CANDIDATES="${CANDIDATES:-0,1 2,3}"
MAX_USED_MIB="${MAX_USED_MIB:-1000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-5}"
POLL_SECONDS="${POLL_SECONDS:-60}"
QUIET_POLLS="${QUIET_POLLS:-5}"

mkdir -p "$PROJECT_DIR/logs"
echo "$$" >"$PROJECT_DIR/logs/sdar8b_wait.pid"
read -r -a PAIRS <<< "$CANDIDATES"
declare -A quiet
for pair in "${PAIRS[@]}"; do quiet[$pair]=0; done

card_idle() {
    local used util
    IFS=', ' read -r used util <<< "$(nvidia-smi --id="$1" \
        --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
    [ "$used" -le "$MAX_USED_MIB" ] && [ "$util" -le "$MAX_UTIL_PCT" ]
}

while true; do
    ready=()
    for pair in "${PAIRS[@]}"; do
        idle=1
        IFS=, read -r -a cards <<< "$pair"
        for gpu in "${cards[@]}"; do card_idle "$gpu" || idle=0; done
        if [ "$idle" = "1" ]; then quiet[$pair]=$(( quiet[$pair] + 1 )); else quiet[$pair]=0; fi
        [ "${quiet[$pair]}" -ge "$QUIET_POLLS" ] && ready+=("$pair")
    done
    echo "$(date '+%F %T') quiet polls: $(for p in "${PAIRS[@]}"; do printf '(%s)=%s ' "$p" "${quiet[$p]}"; done)"
    if [ "${#ready[@]}" -gt 0 ]; then
        echo "$(date '+%F %T') launching on: ${ready[*]}"
        REPLICAS="${ready[*]}" exec bash "$PROJECT_DIR/launch_sdar8b.sh"
    fi
    sleep "$POLL_SECONDS"
done
