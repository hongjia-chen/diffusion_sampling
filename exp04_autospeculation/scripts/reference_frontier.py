#!/usr/bin/env python3
"""Build the Dream-7B quality/latency reference frontier.

This experiment's own comparison is SDAR-vs-SDAR: fixed-k against adaptive
against autospec, one model, one architecture, only the sampler varying. The
Dream numbers are a *reference curve*, not a control. Dream-7B has 4.5x the
parameters of SDAR-1.7B, so "beats Dream's latency" is nearly guaranteed by
parameter count and "matches Dream's quality" nearly impossible by the same
token; neither is evidence about a sampler. What the curve does support is a
system-level claim -- that a 1.7B model with a better sampler reaches the same
quality/latency Pareto region as a 7B model with a worse one -- and for that it
has to be the *best* Dream can do, or beating it means nothing.

So the frontier is the Pareto upper envelope over every Dream arm available,
pooled across output roots and decoding recipes. Pooling recipes is deliberate:
on GSM8K the recipe spread (0.82 under entropy/T=0.1 versus 0.63 under greedy)
is far larger than any scheduler effect, so a frontier built from one recipe
would understate the target by more than the entire effect under study.

Rows are tagged with the root and recipe they came from. `--temperature`
restricts to one recipe, which is how the reference curve is actually built:
the Dream/ADJUST instruct recipe (entropy, T=0.1, top_p 0.9), the same one the
new HumanEval+/MATH-500 Dream runs will use. Its `fixed:1` reproduces Dream's
published GSM8K score (0.820 measured against 81.0 published), which is the
check that the harness agrees with the model's own paper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autospec.io import write_json


def pareto(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Maximal score at minimal latency."""
    keep = []
    for a in points:
        dominated = any(
            b["score"] >= a["score"]
            and b["seconds_per_task"] <= a["seconds_per_task"]
            and (b["score"] > a["score"] or b["seconds_per_task"] < a["seconds_per_task"])
            for b in points
        )
        if not dominated:
            keep.append(a)
    return sorted(keep, key=lambda r: r["seconds_per_task"])


def load(summaries: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for path in summaries:
        root = path.parent.parent.name
        for entry in json.loads(path.read_text(encoding="utf-8")):
            row = {
                "benchmark": entry["benchmark"],
                "method": entry["method"],
                "method_type": entry.get("method_type"),
                "score": entry["primary_score"],
                "seconds_per_task": entry["seconds_per_task"],
                "tokens_per_second": entry.get("tokens_per_second"),
                "canvas_tokens_per_second": entry.get("canvas_tokens_per_second"),
                "continuation_length": entry.get("continuation_length"),
                "temperature": entry.get("temperature"),
                "top_p": entry.get("top_p"),
                "confidence_metric": entry.get("confidence_metric"),
                "model": entry.get("model"),
                "early_eos_rate": entry.get("early_eos_rate"),
                "source_root": root,
            }
            # v3_matched imported v2's adaptive arm verbatim; the same arm must
            # not appear twice and inflate the apparent number of points.
            key = (row["benchmark"], row["method"], round(row["score"], 6),
                   round(row["seconds_per_task"], 3))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", type=Path, nargs="+",
                        help="comparison/summary.json files from Dream runs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmarks", nargs="+",
                        help="restrict to these benchmarks")
    parser.add_argument("--temperature", type=float,
                        help="restrict to arms run at this temperature. The "
                             "reference frontier is the Dream/ADJUST instruct "
                             "recipe (entropy, T=0.1, top_p 0.9); mixing "
                             "recipes into one curve makes it unreadable.")
    args = parser.parse_args()
    rows = load(args.summaries)
    if args.temperature is not None:
        rows = [r for r in rows if r["temperature"] == args.temperature]
    benchmarks = args.benchmarks or sorted({row["benchmark"] for row in rows})
    result: dict[str, Any] = {
        "note": "Reference curve, not a control. See module docstring.",
        "sources": [str(path.resolve()) for path in args.summaries],
        "benchmarks": {},
    }
    for benchmark in benchmarks:
        points = [row for row in rows if row["benchmark"] == benchmark]
        if not points:
            continue
        best = max(points, key=lambda r: r["score"])
        fastest = min(points, key=lambda r: r["seconds_per_task"])
        result["benchmarks"][benchmark] = {
            "arms_considered": len(points),
            "quality_ceiling": {"score": best["score"], "method": best["method"],
                                "source_root": best["source_root"],
                                "seconds_per_task": best["seconds_per_task"]},
            "latency_floor": {"seconds_per_task": fastest["seconds_per_task"],
                              "method": fastest["method"],
                              "source_root": fastest["source_root"],
                              "score": fastest["score"]},
            "frontier": pareto(points),
        }
    write_json(args.output, result)
    for benchmark, block in result["benchmarks"].items():
        print(f"\n{benchmark}: {block['arms_considered']} arms -> "
              f"{len(block['frontier'])} on the frontier")
        print(f"  {'s/task':>8} {'score':>6}  {'method':22} {'root':12} recipe")
        for row in block["frontier"]:
            recipe = (f"T={row['temperature']} top_p={row['top_p']} "
                      f"conf={row['confidence_metric']}")
            print(f"  {row['seconds_per_task']:8.1f} {row['score']:6.3f}  "
                  f"{row['method']:22} {row['source_root']:12} {recipe}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
