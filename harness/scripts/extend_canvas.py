#!/usr/bin/env python3
"""Prepare finished runs for a longer canvas: set aside only the tasks it can change.

A forward never reads past its own block (blockwise.py, cross.py), so a task
whose stop token landed inside the old canvas generates the same tokens under
a longer one. The tasks that must be re-decoded are the ones that hit the old
limit (`maximum_length_termination`). Also set aside, to be safe: any task whose
stop token sits in the last `block - 1` canvas positions. That block was
partly past the old canvas, and cross.py never decodes the part past it, so its
commit order can differ. (blockwise.py decodes that part either way, so for the
SDAR baselines those tasks just re-decode identically: a free determinism check.)

For each method directory this:
  * moves the affected records to `records.canvas<old>.jsonl` (kept, not deleted),
  * leaves the rest in records.jsonl,
  * sets metadata.json `continuation_length` to the new canvas and appends the
    old one to `canvas_extended_from`, so the runner's resume check accepts it,
  * deletes the EvalPlus artifacts so the launcher re-scores the method.

Then rerun the launcher with the new canvas: resume decodes only the missing tasks.

    python extend_canvas.py ../MI_decoding/outputs/baselines/SDAR-8B-Chat --to 4096 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Runnable from any directory: the package lives next to scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sdar_harness.io import stream_jsonl, write_json, write_jsonl

STALE_SCORING = (
    "evalplus_samples.jsonl",
    "evalplus_complete_samples.jsonl",
    "records.scored.jsonl",
    "evalplus_results.json",
)


def needs_rerun(record: dict, canvas: int, block: int) -> bool:
    if record["maximum_length_termination"]:
        return True
    eos = record.get("eos_position")
    return eos is not None and eos >= canvas - (block - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path,
                        help="output roots (<root>/<benchmark>/<method>/records.jsonl) or method dirs")
    parser.add_argument("--to", type=int, required=True, help="the new canvas (continuation length)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    method_dirs: list[Path] = []
    for root in args.roots:
        if (root / "records.jsonl").is_file():
            method_dirs.append(root)
        else:
            method_dirs.extend(sorted(path.parent for path in root.glob("*/*/records.jsonl")))
    if not method_dirs:
        raise SystemExit("no records.jsonl under the given roots")

    total = 0
    for method_dir in method_dirs:
        metadata_path = method_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        canvas, block = int(metadata["continuation_length"]), metadata.get("block_length")
        if block is None:
            raise SystemExit(f"{method_dir}: no block_length in metadata")
        if canvas == args.to:
            print(f"{method_dir}: already at {canvas}")
            continue
        if canvas > args.to:
            raise SystemExit(f"{method_dir}: canvas {canvas} is longer than --to {args.to}")
        records = list(stream_jsonl(method_dir / "records.jsonl"))
        archive = method_dir / f"records.canvas{canvas}.jsonl"
        if archive.exists():
            raise SystemExit(f"{archive} exists; this directory was extended from {canvas} before")
        rerun = [r for r in records if needs_rerun(r, canvas, block)]
        keep = [r for r in records if not needs_rerun(r, canvas, block)]
        capped = sum(bool(r["maximum_length_termination"]) for r in rerun)
        total += len(rerun)
        print(f"{method_dir}: {canvas} -> {args.to}: rerun {len(rerun)}/{len(records)} "
              f"({capped} at the limit, {len(rerun) - capped} ended in the last partial block)"
              f"{' [dry run]' if args.dry_run else ''}")
        if args.dry_run:
            continue
        write_jsonl(archive, rerun)
        write_jsonl(method_dir / "records.jsonl", keep)
        metadata["continuation_length"] = args.to
        metadata["canvas_extended_from"] = [*metadata.get("canvas_extended_from", []), canvas]
        metadata["status"] = "running"
        write_json(metadata_path, metadata)
        for name in STALE_SCORING:
            (method_dir / name).unlink(missing_ok=True)
    print(f"{total} tasks to re-decode")


if __name__ == "__main__":
    main()
