#!/usr/bin/env python3
"""The four benchmarks of SDAR's OpenCompass config, in the harness format.

    python scripts/prepare_opencompass.py --output-dir ../data/opencompass

Writes gsm8k.jsonl, math_500.jsonl, humaneval.jsonl and mbpp_sanitized.jsonl,
each with a `<name>.metadata.json`. Every record has `"protocol": "opencompass"`
and its `prompt` is the config's HUMAN turn (sdar_harness.opencompass.PROMPTS),
which the runner renders through the chat template, as BD3withChatTemplate does.

    gsm8k           openai/gsm8k main test (1,319). target = gsm8k_dataset_postprocess(answer)
    math_500        HuggingFaceH4/MATH-500 (500; OpenCompass's test_prm800k_500 problems).
                    target = extract_boxed_answer(solution), OpenCompass's gold; it equals
                    the dataset's `answer` field on all 500 (checked 2026-10-03)
    humaneval       openai/openai_humaneval (164), the original prompts and tests
                    (evalplus edits HumanEval/115's prompt)
    mbpp_sanitized  google-research-datasets/mbpp "sanitized" test (257, ids 11-479):
                    OpenCompass's SanitizedMBPPDataset rows [7:264]. target = the 3
                    asserts it is graded on

Example IDs match the legacy files (gsm8k-test-<i>, MATH unique ids, HumanEval/<n>,
Mbpp/<n>), so tasks pair across protocols.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sdar_harness.io import sha256_file, write_json, write_jsonl
from sdar_harness.opencompass import PROMPTS, extract_boxed_answer, gsm8k_dataset_postprocess


def gsm8k(load) -> tuple[list[dict], str]:
    rows = []
    for index, row in enumerate(load("openai/gsm8k", "main", split="test")):
        rows.append({
            "benchmark": "gsm8k", "protocol": "opencompass", "split": "test",
            "example_id": f"gsm8k-test-{index}", "source_index": index,
            "question": row["question"], "answer": row["answer"],
            "prompt": PROMPTS["gsm8k"].format(question=row["question"]),
            "target": gsm8k_dataset_postprocess(row["answer"]),
        })
    return rows, "openai/gsm8k main test; config gsm8k_0shot_v2_gen_17d799"


def math_500(load) -> tuple[list[dict], str]:
    rows = []
    for row in load("HuggingFaceH4/MATH-500", split="test"):
        rows.append({
            "benchmark": "math_500", "protocol": "opencompass",
            "example_id": row["unique_id"], "problem": row["problem"],
            "solution": row["solution"], "answer": row["answer"],
            "subject": row["subject"], "level": row["level"],
            "prompt": PROMPTS["math_500"].format(problem=row["problem"]),
            "target": extract_boxed_answer(row["solution"]),
        })
    return rows, "HuggingFaceH4/MATH-500 test; config math_prm800k_500_0shot_cot_gen_11c4b5"


def humaneval(load) -> tuple[list[dict], str]:
    rows = []
    for row in load("openai/openai_humaneval", split="test"):
        rows.append({
            "benchmark": "humaneval", "protocol": "opencompass",
            "example_id": row["task_id"], "entry_point": row["entry_point"],
            "problem_prompt": row["prompt"], "test": row["test"],
            "prompt": PROMPTS["humaneval"].format(prompt=row["prompt"]),
        })
    return rows, "openai/openai_humaneval test; config humaneval_openai_sample_evals_gen_dcae0e"


def mbpp_sanitized(load) -> tuple[list[dict], str]:
    rows = []
    for row in load("google-research-datasets/mbpp", "sanitized", split="test"):
        tests = "\n".join(row["test_list"])
        rows.append({
            "benchmark": "mbpp_sanitized", "protocol": "opencompass", "split": "test",
            "example_id": f"Mbpp/{row['task_id']}", "task_id": row["task_id"],
            "text": row["prompt"], "test_list": list(row["test_list"]),
            # OpenCompass does not prepend these to the program; kept for reference.
            "test_imports": list(row["test_imports"]),
            "prompt": PROMPTS["mbpp_sanitized"].format(text=row["prompt"], test_list=tests),
            "target": tests,
        })
    ids = [row["task_id"] for row in rows]
    if len(rows) != 257 or ids != sorted(ids):
        raise RuntimeError("expected the 257-task sanitized test split in task-id order")
    return rows, ("google-research-datasets/mbpp sanitized test = OpenCompass SanitizedMBPPDataset "
                  "rows [7:264]; config sanitized_mbpp_mdblock_0shot_nocot_gen_a2e416")


BUILDERS = {"gsm8k": gsm8k, "math_500": math_500, "humaneval": humaneval,
            "mbpp_sanitized": mbpp_sanitized}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only", nargs="+", choices=sorted(BUILDERS))
    args = parser.parse_args()
    from datasets import load_dataset

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.only or BUILDERS:
        rows, source = BUILDERS[name](load_dataset)
        if any(row.get("target") is None for row in rows if name != "humaneval"):
            raise RuntimeError(f"{name}: a record has no gold answer")
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        write_json(args.output_dir / f"{name}.metadata.json", {
            "benchmark": name, "protocol": "opencompass", "source": source,
            "count": len(rows), "example_ids": [row["example_id"] for row in rows],
            "sha256": sha256_file(path),
        })
        print(f"{path}: {len(rows)} records")


if __name__ == "__main__":
    main()
