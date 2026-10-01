#!/usr/bin/env python3
"""Put SDAR-8B next to the ported SDAR-1.7B frontier, after checking they match.

Refuses to print a comparison unless both sides scored the same tasks in the
same order. For the three ported benchmarks every 8B arm's ordered example_id
list must hash to the fingerprint recorded in PORTED_RESULTS.md; for anything
else (MATH-500, whose 1.7B reference is rerun here) it must hash the same as
the 1.7B arm's. A mismatch means the sample, the dataset file or the resume
logic changed, and the two columns would then be different experiments.

Usage: compare_scale.py <8B summary.json> [--reference <1.7B summary.json> ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
REFERENCE_SUMMARY = PROJECT / "reference/sdar_1p7b_chat_b64/comparison/summary.json"
# Ported from exp04. Scores carry over; timings were full-canvas (no early
# stop) and are flagged `timing_comparable: false`.
MATH_REFERENCE_SUMMARY = (
    PROJECT / "reference/sdar_1p7b_chat_b64_math500/comparison/summary.exp06_vgb.json"
)

# PORTED_RESULTS.md, "Ordered task-ID fingerprints".
FINGERPRINTS = {
    "gsm8k": "987fd1e556a9a4fce4ec097d90de2a11f3ab5acd0787c3fba2190351b6566294",
    "humaneval_plus": "d1ebc039377e8e813e79586f681222c15b7a8b1aeb83f42776e96e5475c16aa8",
    "mbpp_plus": "dac76d8097602a13c559b198366790efc74062fbb23263c24539329e2fec996e",
}


def fingerprint(records: Path) -> str:
    ids = [json.loads(line)["example_id"] for line in records.open() if line.strip()]
    # Newline-terminated, as PORTED_RESULTS.md computed them.
    return hashlib.sha256("".join(f"{value}\n" for value in ids).encode()).hexdigest()


def key(row: dict) -> tuple[str, str]:
    return row["benchmark"], row["method"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("large", type=Path, help="8B comparison/summary.json")
    parser.add_argument("--reference", type=Path, nargs="+", default=[REFERENCE_SUMMARY, MATH_REFERENCE_SUMMARY],
                        help="1.7B summary.json files; later files add benchmarks")
    args = parser.parse_args()
    large = {key(row): row for row in json.loads(args.large.read_text())}
    small = {
        key(row): row
        for path in args.reference
        for row in json.loads(path.read_text())
    }

    bad = []
    for (benchmark, method), row in sorted(large.items()):
        # The summary's records_path may be records.scored.jsonl; both keep
        # the generation order.
        got = fingerprint(Path(row["records_path"]))
        expected = FINGERPRINTS.get(benchmark)
        if expected is None and (benchmark, method) in small:
            expected = fingerprint(Path(small[benchmark, method]["records_path"]))
        if got != expected:
            bad.append(f"{benchmark}/{method}: {got}")
    if bad:
        print("TASK MISMATCH -- not comparing:\n  " + "\n  ".join(bad))
        return 1

    header = (f"{'benchmark':<15} {'arm':<10} {'1.7B score':>10} {'8B score':>9} "
              f"{'1.7B tok/s':>10} {'8B tok/s':>9} {'1.7B s/task':>11} {'8B s/task':>10} "
              f"{'8B NFE':>7} {'1.7B cap':>8} {'8B cap':>7}")
    print(header)
    print("-" * len(header))
    for name in sorted(set(large) & set(small)):
        a, b = small[name], large[name]
        timed = a.get("timing_comparable", True)
        a_rate = f"{a['tokens_per_second']:>10.2f}" if timed else f"{'n/a*':>10}"
        a_task = f"{a['seconds_per_task']:>11.2f}" if timed else f"{'n/a*':>11}"
        print(f"{name[0]:<15} {name[1]:<10} {a['primary_score']:>10.2f} {b['primary_score']:>9.2f} "
              f"{a_rate} {b['tokens_per_second']:>9.2f} "
              f"{a_task} {b['seconds_per_task']:>10.2f} "
              f"{b['mean_model_forward_passes']:>7.1f} "
              # Share of answers with no stop token inside the canvas: cut off.
              f"{a['maximum_length_rate']:>8.0%} {b['maximum_length_rate']:>7.0%}")
    if any(not small[name].get("timing_comparable", True) for name in set(large) & set(small)):
        print("\n* 1.7B timing ported from exp04 without early stop; not comparable.")
    missing = sorted(set(small) - set(large))
    if missing:
        print("\nnot yet run at 8B: " + ", ".join(f"{b}/{m}" for b, m in missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
