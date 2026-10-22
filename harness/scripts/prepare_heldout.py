#!/usr/bin/env python3
"""Held-out prompts for calibration work (the η study, MI_decoding/ETA_STUDY.md).

    python scripts/prepare_heldout.py --output-dir ../data/heldout

Prompts drawn from splits that no evaluation set uses, formatted with SDAR's
OpenCompass templates (sdar_harness.opencompass.PROMPTS) so that a decoder sees
the same kind of prompt as at evaluation time:

    gsm8k_train    openai/gsm8k main train (7,473); seeded sample
    math_train     EleutherAI/hendrycks_math, all 7 subjects, train (7,500; MATH-500
                   is drawn from the test split); stratified by level in MATH-500's
                   proportions; MATH-500's template
    mbpp_heldout   google-research-datasets/mbpp "full" prompt+validation+train
                   (ids 1-10, 511-974; the test split is 11-510 and holds all of
                   sanitized MBPP's test), minus every MBPP+ task (EvalPlus draws on
                   all splits); seeded sample; sanitized MBPP's template

Each file is checked disjoint from data/opencompass/ (IDs and problem text) and,
for MBPP, from data/full/mbpp_plus.jsonl. Records keep the dataset order.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sdar_harness.io import sha256_file, stream_jsonl, write_json, write_jsonl
from sdar_harness.opencompass import PROMPTS, extract_boxed_answer, gsm8k_dataset_postprocess

MATH_SUBJECTS = (
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
)


def seeded_indices(population: int, size: int, seed: int) -> list[int]:
    """run_benchmark.select_sample's draw: sorted, so dataset order is kept."""
    if size > population:
        raise ValueError(f"asked for {size} of {population}")
    return sorted(random.Random(seed).sample(range(population), size))


def largest_remainder(weights: dict[int, int], total: int) -> dict[int, int]:
    scale = sum(weights.values())
    exact = {key: total * value / scale for key, value in weights.items()}
    quota = {key: int(value) for key, value in exact.items()}
    by_remainder = sorted(exact, key=lambda key: (exact[key] - quota[key], -key), reverse=True)
    for key in by_remainder[: total - sum(quota.values())]:
        quota[key] += 1
    return quota


def gsm8k_train(load, data_root: Path, size: int, seed: int):
    source = load("openai/gsm8k", "main", split="train")
    test_questions = {row["question"] for row in stream_jsonl(data_root / "opencompass/gsm8k.jsonl")}
    rows = []
    for index in seeded_indices(len(source), size, seed):
        row = source[index]
        if row["question"] in test_questions:
            raise RuntimeError(f"gsm8k train {index} repeats a test question")
        rows.append({
            "benchmark": "gsm8k_train", "protocol": "opencompass", "split": "train",
            "example_id": f"gsm8k-train-{index}", "source_index": index,
            "question": row["question"], "answer": row["answer"],
            "prompt": PROMPTS["gsm8k"].format(question=row["question"]),
            "target": gsm8k_dataset_postprocess(row["answer"]),
        })
    selection = f"seeded_indices({len(source)}, {size}, seed={seed})"
    return rows, "openai/gsm8k main train", selection


def math_train(load, data_root: Path, size: int, seed: int):
    eval_rows = list(stream_jsonl(data_root / "opencompass/math_500.jsonl"))
    eval_problems = {row["problem"] for row in eval_rows}
    pool: list[dict] = []
    for subject in MATH_SUBJECTS:
        for index, row in enumerate(load("EleutherAI/hendrycks_math", subject, split="train")):
            level = row["level"].removeprefix("Level ")
            if not level.isdigit():
                continue  # a handful are "Level ?"
            pool.append({**row, "subject": subject, "index": index, "level": int(level)})
    quota = largest_remainder(Counter(int(row["level"]) for row in eval_rows), size)
    rng = random.Random(seed)
    picked: list[dict] = []
    for level in sorted(quota):
        members = [row for row in pool if row["level"] == level]
        picked.extend(rng.sample(members, quota[level]))
    picked.sort(key=lambda row: (MATH_SUBJECTS.index(row["subject"]), row["index"]))
    rows = []
    for row in picked:
        if row["problem"] in eval_problems:
            raise RuntimeError(f"MATH train {row['subject']}/{row['index']} repeats a MATH-500 problem")
        rows.append({
            "benchmark": "math_train", "protocol": "opencompass", "split": "train",
            "example_id": f"train/{row['subject']}/{row['index']}",
            "problem": row["problem"], "solution": row["solution"],
            "subject": row["type"], "level": row["level"],
            "prompt": PROMPTS["math_500"].format(problem=row["problem"]),
            "target": extract_boxed_answer(row["solution"]),
        })
    selection = (f"stratified by level in MATH-500's proportions (quota {json.dumps(dict(sorted(quota.items())))}), "
                 f"random.Random({seed}).sample per level in level order 1-5, from {len(pool)} "
                 "train problems with a numeric level; subject-then-index order")
    return rows, "EleutherAI/hendrycks_math train, 7 subjects", selection


def mbpp_heldout(load, data_root: Path, size: int, seed: int):
    excluded = {int(row["task_id"]) for row in stream_jsonl(data_root / "opencompass/mbpp_sanitized.jsonl")}
    excluded |= {int(row["example_id"].split("/")[1]) for row in stream_jsonl(data_root / "full/mbpp_plus.jsonl")}
    pool = []
    for split in ("prompt", "validation", "train"):
        for row in load("google-research-datasets/mbpp", "full", split=split):
            if int(row["task_id"]) not in excluded:
                pool.append({**row, "split": split})
    pool.sort(key=lambda row: int(row["task_id"]))
    rows = []
    for index in seeded_indices(len(pool), size, seed):
        row = pool[index]
        tests = "\n".join(row["test_list"])
        rows.append({
            "benchmark": "mbpp_heldout", "protocol": "opencompass", "split": row["split"],
            "example_id": f"Mbpp/{row['task_id']}", "task_id": int(row["task_id"]),
            "text": row["text"], "test_list": list(row["test_list"]),
            "prompt": PROMPTS["mbpp_sanitized"].format(text=row["text"], test_list=tests),
            "target": tests,
        })
    selection = (f"seeded_indices({len(pool)}, {size}, seed={seed}) over full-MBPP "
                 "prompt+validation+train sorted by task_id, minus sanitized-test and MBPP+ ids")
    return rows, "google-research-datasets/mbpp full, non-test splits", selection


BUILDERS = {"gsm8k_train": gsm8k_train, "math_train": math_train, "mbpp_heldout": mbpp_heldout}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--only", nargs="+", choices=sorted(BUILDERS))
    args = parser.parse_args()
    from datasets import load_dataset

    eval_ids = {
        str(row["example_id"])
        for name in ("gsm8k", "math_500", "humaneval", "mbpp_sanitized")
        for row in stream_jsonl(args.data_root / f"opencompass/{name}.jsonl")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.only or BUILDERS:
        rows, source, selection = BUILDERS[name](load_dataset, args.data_root, args.size, args.seed)
        ids = [row["example_id"] for row in rows]
        if len(set(ids)) != len(ids) or set(ids) & eval_ids:
            raise RuntimeError(f"{name}: duplicate IDs or overlap with data/opencompass")
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        write_json(args.output_dir / f"{name}.metadata.json", {
            "benchmark": name, "protocol": "opencompass", "source": source,
            "selection": selection, "seed": args.seed, "count": len(rows),
            "example_ids": ids, "sha256": sha256_file(path),
        })
        print(f"{path}: {len(rows)} records")


if __name__ == "__main__":
    main()
