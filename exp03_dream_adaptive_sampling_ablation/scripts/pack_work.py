#!/usr/bin/env python3
"""Pack (benchmark, method) units across GPUs by longest-processing-time.

Emits one line per GPU worker and benchmark:

    <worker_index> <benchmark> <method> [<method> ...]

`run_benchmark.py` takes one dataset and a list of methods, so a worker's units
have to be grouped by benchmark; each emitted line is one invocation.

Costs are measured seconds per task from the fixed arm of `outputs/v2`. They are
an upper bound here: that run used nucleus sampling, which sorts the vocabulary
and chunks the proposer more finely, while the matched run is greedy.
"""
from __future__ import annotations

import argparse

# benchmark -> fixed_k -> measured seconds per task (outputs/v2)
COSTS = {
    "mbpp_plus": {1: 133.8, 2: 69.2, 4: 34.0, 8: 16.7},
    "gsm8k": {1: 38.8, 2: 20.0, 4: 9.9, 8: 4.9},
    "schema_json": {1: 57.0, 2: 29.6, 4: 14.5, 8: 7.1},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--plan", action="store_true",
                        help="print a human-readable makespan estimate to stderr")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    units = sorted(
        ((COSTS[b][k], b, k) for b in COSTS for k in args.ks if k in COSTS[b]),
        reverse=True,
    )
    loads = [0.0] * args.workers
    bins: list[list[tuple[str, int]]] = [[] for _ in range(args.workers)]
    for cost, benchmark, k in units:
        target = min(range(args.workers), key=lambda i: loads[i])
        loads[target] += cost
        bins[target].append((benchmark, k))

    for index, assigned in enumerate(bins):
        by_benchmark: dict[str, list[int]] = {}
        for benchmark, k in assigned:
            by_benchmark.setdefault(benchmark, []).append(k)
        # Cheapest benchmark first so a worker produces complete results early
        # if the run is interrupted.
        for benchmark in sorted(by_benchmark, key=lambda b: sum(COSTS[b][k] for k in by_benchmark[b])):
            methods = " ".join(f"fixed:{k}" for k in sorted(by_benchmark[benchmark]))
            print(f"{index} {benchmark} {methods}")

    if args.plan:
        import sys
        for index, load in enumerate(loads):
            hours = load * args.tasks / 3600
            print(f"  worker {index}: {load:6.1f} s/task  ->  {hours:4.1f}h", file=sys.stderr)
        print(f"  makespan: ~{max(loads) * args.tasks / 3600:.1f}h "
              f"(balanced to {100 * (max(loads) - min(loads)) / max(loads):.0f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
