#!/usr/bin/env bash
## Reports whether the three ablation workers are alive and advancing.
##
## Reads only /proc and the run's own output files. Nothing here calls CUDA,
## nvidia-smi, or NVML, so it keeps reporting when the driver and the kernel
## module disagree and the usual GPU tooling refuses to load.
##
##   bash scripts/check_run.sh smoke
##   bash scripts/check_run.sh final
##   watch -n 30 bash scripts/check_run.sh smoke

# Deliberately no `set -e`: a status tool must finish its report even when an
# individual probe fails.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

RUN_LABEL="${1:-smoke}"
case "$RUN_LABEL" in
    -h|--help)
        echo "usage: bash scripts/check_run.sh [smoke|barebones|final|<label>]" >&2
        exit 0
        ;;
esac

LOG_ROOT="${LOG_ROOT:-$PROJECT_DIR/logs/$RUN_LABEL}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/$RUN_LABEL}"

if [ ! -d "$LOG_ROOT" ]; then
    echo "No log directory at $LOG_ROOT -- has the $RUN_LABEL run been started?" >&2
    exit 1
fi

shopt -s nullglob

rule() { printf '%s\n' "----------------------------------------------------------------------"; }

# Seconds since a file was last written, or "-" when it does not exist.
age_seconds() {
    local path="$1" mtime now
    [ -e "$path" ] || { printf '%s' "-"; return; }
    mtime="$(stat -c %Y "$path" 2>/dev/null)" || { printf '%s' "-"; return; }
    now="$(date +%s)"
    printf '%s' "$(( now - mtime ))"
}

human_age() {
    local seconds="$1"
    [ "$seconds" = "-" ] && { printf '%s' "-"; return; }
    if   [ "$seconds" -lt 60 ]   ; then printf '%ss' "$seconds"
    elif [ "$seconds" -lt 3600 ] ; then printf '%sm%ss' "$(( seconds / 60 ))" "$(( seconds % 60 ))"
    else                                printf '%sh%sm' "$(( seconds / 3600 ))" "$(( seconds % 3600 / 60 ))"
    fi
}

# True when the pid is live *and* still the benchmark process, so a recycled
# pid from a previous run is not reported as healthy.
is_our_worker() {
    local pid="$1"
    [ -d "/proc/$pid" ] || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "run_benchmark.py"
}

printf '\n'
printf 'run          %s\n' "$RUN_LABEL"
printf 'logs         %s\n' "$LOG_ROOT"
printf 'outputs      %s\n' "$OUTPUT_ROOT"
printf 'checked at   %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
printf '\n'

# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------
rule
printf '%-22s %-8s %-9s %-10s %-8s %-7s %s\n' WORKER PID STATE CPU-TIME RSS GPU-FD LOG-AGE
rule

any_alive=0
any_worker=0
for pid_file in "$LOG_ROOT"/*.pid; do
    worker="$(basename "$pid_file" .pid)"
    [ "$worker" = "launcher" ] && continue
    any_worker=1
    pid="$(head -n 1 "$pid_file" 2>/dev/null | tr -d '[:space:]')"
    log_file="$LOG_ROOT/$worker.log"
    log_age="$(human_age "$(age_seconds "$log_file")")"

    if [ -z "$pid" ]; then
        printf '%-22s %-8s %-9s %-10s %-8s %-7s %s\n' "$worker" "?" "no-pid" "-" "-" "-" "$log_age"
        continue
    fi

    state="dead"; cpu_time="-"; rss="-"; gpu_fd="-"
    # The launcher backgrounds a subshell; the python process is its child.
    worker_pid="$pid"
    if ! is_our_worker "$worker_pid"; then
        for child in $(pgrep -P "$pid" 2>/dev/null); do
            if is_our_worker "$child"; then worker_pid="$child"; break; fi
        done
    fi

    if is_our_worker "$worker_pid"; then
        state="running"; any_alive=1
        read -r cpu_time rss <<<"$(ps -p "$worker_pid" -o time=,rss= 2>/dev/null | tr -s ' ')"
        [ -n "${rss:-}" ] && rss="$(( rss / 1024 ))M"
        # An open handle on /dev/nvidia* means the process still holds the GPU.
        # This is readable straight from /proc and needs no NVML.
        if grep -qs '/dev/nvidia' "/proc/$worker_pid/maps"; then gpu_fd="yes"; else gpu_fd="no"; fi
    elif [ -d "/proc/$pid" ]; then
        state="starting"
    fi

    printf '%-22s %-8s %-9s %-10s %-8s %-7s %s\n' \
        "$worker" "$worker_pid" "$state" "${cpu_time:--}" "${rss:--}" "$gpu_fd" "$log_age"
done

if [ "$any_worker" -eq 0 ]; then
    printf '  (no .pid files -- this run predates pid tracking, or never started)\n'
fi
rule
printf '\n'
printf 'CPU-TIME rising between checks means the worker is doing work.\n'
printf 'LOG-AGE is time since the log was last written; a stalled log with\n'
printf 'rising CPU-TIME is normal mid-task (one task can take minutes).\n'
printf '\n'

# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------
rule
printf '%-40s %s\n' "METHOD" "TASKS DONE"
rule
found_records=0
for benchmark_dir in "$OUTPUT_ROOT"/*/; do
    benchmark="$(basename "$benchmark_dir")"
    [ "$benchmark" = "comparison" ] && continue

    total="?"
    for meta in "$benchmark_dir"run_metadata_worker_*.json; do
        parsed="$(grep -o '"number_of_tasks"[[:space:]]*:[[:space:]]*[0-9]*' "$meta" 2>/dev/null | grep -o '[0-9]*$' | head -n 1)"
        [ -n "$parsed" ] && { total="$parsed"; break; }
    done

    for method_dir in "$benchmark_dir"*/; do
        method="$(basename "$method_dir")"
        [ -f "$method_dir/metadata.json" ] || continue
        found_records=1
        done_count=0
        [ -f "$method_dir/records.jsonl" ] && done_count="$(wc -l < "$method_dir/records.jsonl" 2>/dev/null | tr -d ' ')"
        status="$(grep -o '"status"[[:space:]]*:[[:space:]]*"[a-z]*"' "$method_dir/metadata.json" 2>/dev/null | grep -o '"[a-z]*"$' | tr -d '"')"
        printf '%-40s %s / %s  %s\n' "$benchmark/$method" "$done_count" "$total" "${status:-}"
    done
done
[ "$found_records" -eq 0 ] && printf '  (no method directories yet -- model is probably still loading)\n'
rule
printf '\n'

# ---------------------------------------------------------------------------
# Trouble
# ---------------------------------------------------------------------------
problems=0
for log_file in "$LOG_ROOT"/*.log; do
    if grep -qE 'Traceback|OutOfMemoryError|INTERNAL ASSERT|Error:' "$log_file" 2>/dev/null; then
        [ "$problems" -eq 0 ] && { rule; printf 'PROBLEMS FOUND\n'; rule; }
        problems=1
        printf '\n%s:\n' "$(basename "$log_file")"
        grep -nE 'Traceback|OutOfMemoryError|INTERNAL ASSERT|Error:|memory allocation failed' "$log_file" 2>/dev/null | tail -n 5 | sed 's/^/  /'
    fi
done
[ "$problems" -eq 1 ] && { printf '\n'; rule; printf '\n'; }

# Any of your processes still holding a GPU that is not one of these workers --
# usually an orphan from an interrupted run, and the reason a fresh run OOMs.
strays=""
for maps in /proc/[0-9]*/maps; do
    grep -qs '/dev/nvidia' "$maps" || continue
    stray_pid="${maps%/maps}"; stray_pid="${stray_pid#/proc/}"
    grep -qs "^$stray_pid\$" <(for f in "$LOG_ROOT"/*.pid; do cat "$f" 2>/dev/null; pgrep -P "$(cat "$f" 2>/dev/null)" 2>/dev/null; done) && continue
    strays="$strays $stray_pid"
done
if [ -n "$strays" ]; then
    printf 'OTHER PROCESSES OF YOURS HOLDING A GPU (possible orphans from an\n'
    printf 'interrupted run -- these keep memory reserved and will cause OOM):\n'
    # shellcheck disable=SC2086
    ps -o pid,etime,rss,cmd -p $strays 2>/dev/null | sed 's/^/  /'
    printf '\n  stop one with:  kill <pid>\n\n'
fi

launcher_pid="$(head -n 1 "$LOG_ROOT/launcher.pid" 2>/dev/null | tr -d '[:space:]')"
if [ "$any_alive" -eq 1 ]; then
    printf 'VERDICT: at least one worker is running.\n'
    if [ -n "$launcher_pid" ] && [ -d "/proc/$launcher_pid" ]; then
        printf '\nStop the whole run cleanly (releases the GPUs):\n'
        printf '  kill %s\n' "$launcher_pid"
    fi
else
    printf 'VERDICT: no live workers. Check the logs above, or the run finished.\n'
fi
printf '\n'
