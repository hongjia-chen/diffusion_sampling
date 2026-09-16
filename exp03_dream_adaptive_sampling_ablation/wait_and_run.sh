#!/usr/bin/env bash
# Wait for the target GPUs to go completely idle, then launch run_all.sh.
#
# Polls rather than pre-empts: nothing already running is signalled or killed.
# Two independent conditions must hold, and must hold for several consecutive
# polls, so the run does not start in a gap between phases of another job or
# while one is restarting from a checkpoint.
#
#   1. no compute processes on any target GPU
#   2. used memory below IDLE_MIB on every target GPU
#
# Usage:
#   nohup bash wait_and_run.sh > logs/v2/queue.out 2>&1 &
#   tail -f logs/v2/queue.out
# Cancel with: kill "$(cat logs/v2/queue.pid)"
set -uo pipefail   # deliberately not -e: a transient nvidia-smi failure must
                   # not abandon the wait

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

GPU_IDS="${GPU_IDS:-0 1 2}"
POLL_SECONDS="${POLL_SECONDS:-120}"
IDLE_MIB="${IDLE_MIB:-1000}"
STABLE_CHECKS="${STABLE_CHECKS:-3}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/v2}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-168}"

read -r -a GPUS <<< "$GPU_IDS"
mkdir -p "$LOG_ROOT"
printf '%s\n' "$$" >"$LOG_ROOT/queue.pid"
trap 'rm -f "$LOG_ROOT/queue.pid"' EXIT

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

log "queued: waiting for GPUs ${GPUS[*]} to go idle"
log "  idle means no compute processes and < ${IDLE_MIB} MiB used,"
log "  sustained for ${STABLE_CHECKS} consecutive polls ${POLL_SECONDS}s apart"
log "current occupants (these are left alone):"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
    | while read -r line; do
          pid="${line%%,*}"
          log "  pid $pid  $(ps -o args= -p "$pid" 2>/dev/null | cut -c1-90)"
      done

streak=0
deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
while :; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        log "ABORT: still busy after ${MAX_WAIT_HOURS}h; not launching"
        exit 1
    fi

    busy=""
    for gpu in "${GPUS[@]}"; do
        procs="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
                 | grep -c '[0-9]')"
        used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)"
        # An unreadable GPU counts as busy; never launch on an unknown state.
        if ! [ "$used" -ge 0 ] 2>/dev/null; then
            busy="$busy gpu$gpu:unreadable"
        elif [ "$procs" -gt 0 ] || [ "$used" -ge "$IDLE_MIB" ]; then
            busy="$busy gpu$gpu:${procs}proc/${used}MiB"
        fi
    done

    if [ -z "$busy" ]; then
        streak=$(( streak + 1 ))
        log "all target GPUs idle ($streak/$STABLE_CHECKS)"
        if [ "$streak" -ge "$STABLE_CHECKS" ]; then
            break
        fi
    else
        if [ "$streak" -gt 0 ]; then
            log "no longer idle, resetting streak:$busy"
        fi
        streak=0
        log "waiting -$busy"
    fi
    sleep "$POLL_SECONDS"
done

log "GPUs ${GPUS[*]} idle and stable; launching run_all.sh"
GPU_IDS="$GPU_IDS" exec bash run_all.sh
