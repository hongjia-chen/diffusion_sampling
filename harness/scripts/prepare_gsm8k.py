#!/usr/bin/env python3
"""GSM8K records in the harness format (ported from exp04's prepare_gsm8k.py).

    python scripts/prepare_gsm8k.py --output-dir /tmp/gsm8k --test-size 0
    # then: test.jsonl -> ../data/full/gsm8k.jsonl, test.metadata.json -> gsm8k.metadata.json

`--test-size 0` writes the whole 1,319-problem test split in source order:
the full set for paper numbers. The default 500 is how exp03's frozen GSM8K
sample was drawn (seed 12345), from which ../data/sample100/gsm8k.jsonl comes.
Records are identical either way, so every sampled task is exactly the full
set's record with the same ID.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sdar_harness.io import sha256_file, write_json, write_jsonl


def target_from_answer(answer: str) -> str:
    if "####" not in answer:
        raise ValueError("GSM8K answer is missing its canonical #### delimiter")
    return answer.rsplit("####", 1)[1].strip()


def make_record(row: dict, *, split: str, source_index: int) -> dict:
    # lm-eval's `gsm8k_cot` doc_to_text verbatim. The previous prompt named a
    # `#### <number>` delimiter, which the confidence-ordered proposer would
    # commit at canvas position 0, ending the response before any reasoning was
    # written. There is no format token to misplace here; the answer is read out
    # of the text by regex instead.
    return {
        "benchmark": "gsm8k",
        "split": split,
        "example_id": f"gsm8k-{split}-{source_index}",
        "source_index": source_index,
        "question": row["question"],
        "prompt": f"Q: {row['question']}\nA:",
        "target": target_from_answer(row["answer"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--dev-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=500,
                        help="0 = the whole test split, in source order")
    args = parser.parse_args()
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main")
    rng = random.Random(args.seed)
    train_indices = rng.sample(range(len(dataset["train"])), args.dev_size)
    test_indices = (
        list(range(len(dataset["test"]))) if args.test_size == 0
        else rng.sample(range(len(dataset["test"])), args.test_size)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, source_split, indices in (
        ("dev", "train", train_indices),
        ("test", "test", test_indices),
    ):
        rows = [
            make_record(dataset[source_split][index], split=split, source_index=index)
            for index in indices
        ]
        path = args.output_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        write_json(
            path.with_suffix(".metadata.json"),
            {"benchmark": "gsm8k", "split": split, "seed": args.seed,
             "source_split": source_split, "source_indices": indices,
             "count": len(rows), "sha256": sha256_file(path)},
        )


if __name__ == "__main__":
    main()

