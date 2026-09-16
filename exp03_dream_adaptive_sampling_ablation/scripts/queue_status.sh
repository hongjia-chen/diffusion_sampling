#!/usr/bin/env bash
# One-shot status for the queued run: is it still waiting, or has it started?
#   bash scripts/queue_status.sh
set -uo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${LOG_ROOT:-$DIR/logs/v2}"

alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }

qpid="$(cat "$LOG_ROOT/queue.pid" 2>/dev/null || true)"
lpid="$(cat "$LOG_ROOT/launcher.pid" 2>/dev/null || true)"

if alive "$lpid"; then
    echo "STATE: running  (run_all.sh started, launcher pid $lpid, up $(ps -o etime= -p "$lpid" | tr -d ' '))"
elif alive "$qpid"; then
    echo "STATE: queued   (waiting for GPUs, pid $qpid, up $(ps -o etime= -p "$qpid" | tr -d ' '))"
elif [ -s "$LOG_ROOT/queue.out" ]; then
    echo "STATE: finished or stopped  (no live pid)"
else
    echo "STATE: nothing queued"
fi

echo
echo "--- GPUs"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null \
    | sed 's/^/  /'

echo
echo "--- last queue log lines"
tail -n 4 "$LOG_ROOT/queue.out" 2>/dev/null | sed 's/^/  /' || echo "  (no queue.out)"

progress="$(grep -h '^\[' "$LOG_ROOT"/worker_*.log 2>/dev/null | tail -n 3)"
if [ -n "$progress" ]; then
    echo
    echo "--- latest task completions"
    printf '%s\n' "$progress" | sed 's/^/  /'
    echo
    echo "--- records written so far"
    find "$DIR/outputs/v2" -name records.jsonl -printf '%h\n' 2>/dev/null \
        | sed "s|$DIR/outputs/v2/||" | sort | while read -r m; do
              printf '  %-34s %s\n' "$m" "$(wc -l < "$DIR/outputs/v2/$m/records.jsonl")"
          done
fi
