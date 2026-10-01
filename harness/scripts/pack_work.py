#!/usr/bin/env python3
"""Pack (benchmark, method) units across GPUs by longest-processing-time.

Emits one line per GPU worker and benchmark:

    <worker_index> <benchmark> <method> [<method> ...]

`run_benchmark.py` takes one dataset and a list of methods, so a worker's units
have to be grouped by benchmark; each emitted line is one invocation.

Costs must be MEASURED, and are read from --costs. The previous experiment
hardcoded seconds-per-task from its own fixed arm; those numbers belong to
Dream-7B on this host and mean nothing for a different checkpoint, so there is
no default table here. Produce one with a short calibration run:

    scripts/summarize_results.py --output-root outputs/calib --output-dir /tmp/c
    python3 -c 'import json,sys; print(json.dumps({
        (r["benchmark"], r["method"]): r["seconds_per_task"]
        for r in json.load(open("/tmp/c/summary.json"))}))'

The file maps "<benchmark>" -> "<method>" -> seconds per task, e.g.

    {"gsm8k": {"fixed:1": 38.8, "fixed:8": 4.9}}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--costs", type=Path, required=True,
                        help="JSON: benchmark -> method -> measured seconds per task")
    parser.add_argument("--methods", nargs="+",
                        help="restrict to these methods (default: all in --costs)")
    parser.add_argument("--benchmarks", nargs="+",
                        help="restrict to these benchmarks (default: all in --costs)")
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--plan", action="store_true",
                        help="print a human-readable makespan estimate to stderr")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    costs: dict[str, dict[str, float]] = json.loads(args.costs.read_text(encoding="utf-8"))
    benchmarks = args.benchmarks or sorted(costs)
    units: list[tuple[float, str, str]] = []
    for benchmark in benchmarks:
        if benchmark not in costs:
            parser.error(f"no measured costs for benchmark {benchmark}")
        methods = args.methods or sorted(costs[benchmark])
        for method in methods:
            if method not in costs[benchmark]:
                parser.error(f"no measured cost for {benchmark}/{method}")
            units.append((float(costs[benchmark][method]), benchmark, method))
    if not units:
        parser.error("nothing to pack")
    units.sort(reverse=True)

    loads = [0.0] * args.workers
    bins: list[list[tuple[str, str]]] = [[] for _ in range(args.workers)]
    for cost, benchmark, method in units:
        target = min(range(args.workers), key=lambda i: loads[i])
        loads[target] += cost
        bins[target].append((benchmark, method))

    for index, assigned in enumerate(bins):
        by_benchmark: dict[str, list[str]] = {}
        for benchmark, method in assigned:
            by_benchmark.setdefault(benchmark, []).append(method)
        # Cheapest benchmark first so a worker produces complete results early
        # if the run is interrupted.
        for benchmark in sorted(
            by_benchmark,
            key=lambda b: sum(costs[b][m] for m in by_benchmark[b]),
        ):
            print(f"{index} {benchmark} {' '.join(sorted(by_benchmark[benchmark]))}")

    if args.plan:
        for index, load in enumerate(loads):
            print(f"  worker {index}: {load:6.1f} s/task  ->  "
                  f"{load * args.tasks / 3600:4.1f}h", file=sys.stderr)
        spread = 100 * (max(loads) - min(loads)) / max(loads) if max(loads) else 0.0
        print(f"  makespan: ~{max(loads) * args.tasks / 3600:.1f}h "
              f"(balanced to {spread:.0f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
