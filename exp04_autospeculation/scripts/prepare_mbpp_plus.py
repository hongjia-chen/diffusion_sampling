#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

from autospec.io import sha256_file, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=161803)
    parser.add_argument("--dev-size", type=int, default=20)
    args = parser.parse_args()
    from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash

    tasks = get_mbpp_plus()
    task_ids = sorted(tasks, key=lambda value: int(value.split("/")[-1]))
    if len(task_ids) != 378:
        raise RuntimeError(f"Expected 378 MBPP+ tasks, found {len(task_ids)}")
    rng = random.Random(args.seed)
    dev_ids = set(rng.sample(task_ids, args.dev_size))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, selected in (
        ("dev", [task_id for task_id in task_ids if task_id in dev_ids]),
        ("test", task_ids),
    ):
        rows = [
            {
                "benchmark": "mbpp_plus",
                "split": split,
                "example_id": task_id,
                "task_id": task_id,
                "entry_point": tasks[task_id]["entry_point"],
                "prompt": tasks[task_id]["prompt"].strip() + "\n",
            }
            for task_id in selected
        ]
        path = args.output_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        write_json(
            path.with_suffix(".metadata.json"),
            {"benchmark": "mbpp_plus", "split": split, "seed": args.seed,
             "evalplus_hash": get_mbpp_plus_hash(), "task_ids": selected,
             "count": len(rows), "sha256": sha256_file(path)},
        )


if __name__ == "__main__":
    main()
