#!/usr/bin/env python3
"""Check exp05 SDAR runs against exp04's records for the same tasks.

SDAR is block-causal, so early stop, the {<|im_end|>, <|endoftext|>} stop set
and a shorter canvas cannot change any token before the stop (DESIGN.md
"Baselines to re-run"). Every completion must therefore match exp04's text,
modulo the trailing newline the earlier stop drops. Three differences are
allowed and reported separately:

* canvas-truncated: exp04's answer ran past this run's canvas (MBPP+ 512 -> 256);
* im_end-cut: the answer stopped at a `<|im_end|>` that exp04 decoded past,
  because exp04 only stopped at `<|endoftext|>`. The new text is then a strict
  prefix of exp04's. At k >= 4 SDAR sometimes commits `<|im_end|>` a few tokens
  early and keeps writing; exp04 kept that tail and the new rule drops it;
* adaptive arms only: a borderline acceptance flipped by the vocabulary
  renormalization. These are listed by task id and need explaining.

Usage: compare_exp04.py <exp05 output root> <exp04 output root>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


IM_END = 151645


def load(path: Path) -> dict[str, dict]:
    return {row["example_id"]: row for row in map(json.loads, path.open())}


def main() -> int:
    new_root, old_root = Path(sys.argv[1]), Path(sys.argv[2])
    unexplained = 0
    for records in sorted(new_root.glob("*/*/records.jsonl")):
        bench, method = records.parts[-3], records.parts[-2]
        old_path = old_root / bench / method / "records.jsonl"
        if not old_path.exists():
            print(f"{bench:15s} {method:20s} no exp04 counterpart")
            continue
        canvas = json.loads((records.parent / "metadata.json").read_text())["continuation_length"]
        old = load(old_path)
        same, truncated, im_end_cut, differ = 0, [], [], []
        steps_new = steps_old = secs_new = secs_old = 0.0
        new = load(records)
        for example_id, row in new.items():
            reference = old[example_id]
            steps_new += row["denoising_steps"]; steps_old += reference["denoising_steps"]
            secs_new += row["wall_time_seconds"]; secs_old += reference["wall_time_seconds"]
            if row["raw_completion"].rstrip() == reference["raw_completion"].rstrip():
                same += 1
            elif reference["eos_position"] is None or reference["eos_position"] > canvas - 2:
                truncated.append(example_id)
            elif (
                row.get("stop_token_id") == IM_END
                and reference["raw_completion"].startswith(row["raw_completion"].rstrip())
            ):
                im_end_cut.append(example_id)
            else:
                differ.append(example_id)
        n = len(new)
        unexplained += len(differ)
        print(
            f"{bench:15s} {method:20s} n={n:3d} identical={same:3d} "
            f"canvas-truncated={len(truncated)} im_end-cut={len(im_end_cut)} "
            f"differ={len(differ)} {differ[:6]}  "
            f"steps {steps_old / n:6.1f} -> {steps_new / n:6.1f}  "
            f"s/task {secs_old / n:6.2f} -> {secs_new / n:6.2f}"
        )
    print(f"\nunexplained differences: {unexplained}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
