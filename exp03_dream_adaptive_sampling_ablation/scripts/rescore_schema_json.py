#!/usr/bin/env python3
"""Re-score existing Schema JSON records with the current scorer.

Why this exists: `score_schema_json` originally called `json.loads` on the raw
completion, so every markdown-fenced completion was recorded as a parse error.
In `outputs/v2` that scored all four fixed methods 0.00 on content that was
correct, and it depressed the adaptive arm by an amount that grew with delta --
manufacturing a quality/speed tradeoff out of a formatting habit.

The raw completions are already on disk, so nothing here needs a GPU. Output
goes to `records.scored.jsonl`, which `summarize_results.py` prefers over
`records.jsonl` when present, so `records.jsonl` is never modified and the
original run stays reproducible.

    python scripts/rescore_schema_json.py outputs/v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dream_ablation.io import stream_jsonl, write_jsonl
from dream_ablation.scoring import score_schema_json

# Fields score_schema_json owns. Anything else on a record (timings, step
# counts, commitment histogram) is generation state and is copied through.
SCORED_FIELDS = (
    "parsed_output",
    "score",
    "failure_type",
    "json_parse_valid",
    "json_repair",
    "schema_valid",
    "field_accuracy",
    "missing_required_fields",
    "extra_fields",
    "constraint_violations",
)


def load_reference(data_root: Path) -> dict[str, dict]:
    """Targets and schemas live in the dataset, not in the records."""
    path = data_root / "schema_json" / "test.jsonl"
    if not path.is_file():
        raise SystemExit(f"Missing dataset: {path}")
    return {str(row["example_id"]): row for row in stream_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing")
    args = parser.parse_args()

    reference = load_reference(args.data_root)
    method_dirs = sorted((args.output_root / "schema_json").glob("*/records.jsonl"))
    if not method_dirs:
        raise SystemExit(f"No schema_json records below {args.output_root}")

    print(f"{'method':24s} {'n':>4} {'was':>6} {'now':>6} {'delta':>7} {'repaired':>9}")
    total_changed = 0
    for records_path in method_dirs:
        method_dir = records_path.parent
        rows = list(stream_jsonl(records_path))
        rescored = []
        before = after = repaired = changed = 0
        for row in rows:
            example = reference.get(str(row["example_id"]))
            if example is None:
                raise SystemExit(f"{row['example_id']} is not in the dataset")
            result = score_schema_json(
                row["raw_completion"], example["target"], example["schema"]
            )
            before += bool(row.get("score"))
            after += bool(result["score"])
            repaired += result["json_repair"] is not None
            changed += bool(row.get("score")) != bool(result["score"])
            rescored.append({**row, **{k: result[k] for k in SCORED_FIELDS}})
        total_changed += changed
        print(f"{method_dir.name:24s} {len(rows):4d} {before/len(rows):6.2f} "
              f"{after/len(rows):6.2f} {(after-before)/len(rows):+7.2f} "
              f"{repaired/len(rows):9.2f}")
        if not args.dry_run:
            write_jsonl(method_dir / "records.scored.jsonl", rescored)

    verb = "would change" if args.dry_run else "changed"
    print(f"\n{total_changed} task verdicts {verb}.")
    if not args.dry_run:
        print("Wrote records.scored.jsonl; re-run summarize_results.py to refresh the summary.")


if __name__ == "__main__":
    main()
