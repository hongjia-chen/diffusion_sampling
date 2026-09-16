#!/usr/bin/env python3
"""Freeze MATH-500 (HuggingFaceH4/MATH-500), the OpenAI PRM800K test subset.

The prompt names \\boxed{} explicitly. That is not decoration: scoring reads
the last \\boxed{...} span, and a model that writes its answer as bare prose
scores zero no matter how correct it is. GSM8K can afford a format-free prompt
because flexible-extract takes the last number; MATH answers are expressions,
so there is no equivalent fallback.

`dev` is a seeded subset of `test`, not a disjoint split. It exists for smoke
tests, not for tuning.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from autospec.io import sha256_file, write_json, write_jsonl


PROMPT_TEMPLATE = (
    "Solve the following math problem. "
    "Put your final answer inside \\boxed{{}}.\n\n"
    "Problem: {problem}\n\n"
    "Solution:"
)


def make_record(row: dict, *, split: str, source_index: int) -> dict:
    answer = str(row["answer"]).strip()
    if not answer:
        raise ValueError(f"MATH-500 row {source_index} has an empty answer")
    return {
        "benchmark": "math_500",
        "split": split,
        "example_id": str(row.get("unique_id") or f"math500-{source_index}"),
        "source_index": source_index,
        "problem": row["problem"],
        "subject": row.get("subject"),
        "level": row.get("level"),
        "prompt": PROMPT_TEMPLATE.format(problem=row["problem"]),
        "target": answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--dev-size", type=int, default=20)
    args = parser.parse_args()
    from datasets import load_dataset

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if len(dataset) != 500:
        raise RuntimeError(f"Expected 500 MATH-500 problems, found {len(dataset)}")
    rows = [make_record(dataset[index], split="test", source_index=index)
            for index in range(len(dataset))]
    dev_indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.dev_size))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, selected in (
        ("dev", [dict(rows[index], split="dev") for index in dev_indices]),
        ("test", rows),
    ):
        path = args.output_dir / f"{split}.jsonl"
        write_jsonl(path, selected)
        write_json(
            path.with_suffix(".metadata.json"),
            {"benchmark": "math_500", "split": split, "seed": args.seed,
             "source_dataset": "HuggingFaceH4/MATH-500",
             "dev_is_subset_of_test": True,
             "dev_source_indices": dev_indices if split == "dev" else None,
             "count": len(selected), "sha256": sha256_file(path)},
        )


if __name__ == "__main__":
    main()
