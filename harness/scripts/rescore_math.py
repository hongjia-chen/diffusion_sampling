#!/usr/bin/env python3
"""Re-grade finished MATH-500 records with the current scorer, in place.

`score_math_500` moved from a string normalizer to math_verify on 2026-10-03
(scoring.py). Generation is untouched: this re-reads each record's
`raw_completion`, recomputes the scoring fields, and rewrites records.jsonl
atomically. The old verdict survives as `string_match`, so every record says
which scorer graded it. Re-run summarize_results.py afterwards.

    python rescore_math.py ../MI_decoding/outputs/baselines/SDAR-8B-Chat
    python rescore_math.py <output_root> --dataset ../data/full/math_500.jsonl --dry-run

Each argument is an output root holding `math_500/<method>/records.jsonl`, or a
`math_500` directory itself. Targets come from the dataset the run metadata
names (its sha256 is checked) unless --dataset overrides it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Runnable from any directory: the package lives next to scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sdar_harness.io import sha256_file, stream_jsonl, write_jsonl
from sdar_harness.scoring import score_record


def benchmark_dir(root: Path) -> Path:
    return root if root.name == "math_500" else root / "math_500"


def dataset_for(bench_dir: Path, override: Path | None) -> Path:
    metadata = [json.loads(path.read_text()) for path in sorted(bench_dir.glob("run_metadata_worker_*.json"))]
    paths = {row["dataset"] for row in metadata}
    hashes = {row["dataset_sha256"] for row in metadata if row.get("dataset_sha256")}
    if override is not None:
        dataset = override
    elif len(paths) == 1:
        dataset = Path(paths.pop())
    else:
        raise SystemExit(f"{bench_dir}: run metadata names {sorted(paths) or 'no'} dataset(s); pass --dataset")
    if not dataset.is_file():
        raise SystemExit(f"{bench_dir}: {dataset} is gone; pass --dataset")
    # Whichever file supplies the targets must be the one the runs read.
    if len(hashes) > 1:
        raise SystemExit(f"{bench_dir}: runs read different dataset files")
    if hashes and sha256_file(dataset) != next(iter(hashes)):
        raise SystemExit(f"{bench_dir}: {dataset} is not the file the runs read (sha256 mismatch)")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--dataset", type=Path, help="MATH-500 jsonl with example_id and target")
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    for root in args.roots:
        bench_dir = benchmark_dir(root)
        method_dirs = sorted(path.parent for path in bench_dir.glob("*/records.jsonl"))
        if not method_dirs:
            raise SystemExit(f"{bench_dir}: no */records.jsonl")
        dataset = dataset_for(bench_dir, args.dataset)
        # The dataset row, not just its target: score_record grades an
        # OpenCompass-protocol row (no "Problem:" truncation) as OpenCompass does.
        rows = {str(row["example_id"]): row for row in stream_jsonl(dataset)}
        for method_dir in method_dirs:
            path = method_dir / "records.jsonl"
            records = list(stream_jsonl(path))
            before = after = gained = lost = 0
            for record in records:
                if record.get("benchmark") != "math_500":
                    raise SystemExit(f"{path}: not a MATH-500 record")
                scoring = score_record(rows[str(record["example_id"])], record["raw_completion"])
                old = bool(record.get("score"))
                before += old
                after += scoring["score"]
                gained += scoring["score"] and not old
                lost += old and not scoring["score"]
                record.update(scoring)
            if not args.dry_run:
                write_jsonl(path, records)
            n = len(records)
            print(f"{method_dir}: n={n} score {before / n:.3f} -> {after / n:.3f} "
                  f"(+{gained} -{lost}){' [dry run]' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
