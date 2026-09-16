#!/usr/bin/env python3
"""Score one generated code method with EvalPlus (MBPP+ or HumanEval+).

Generalized from the previous experiment's score_mbpp_plus.py, which hardcoded
the mbpp dataset and its loader.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from autospec.io import stream_jsonl, write_jsonl


PASS = "pass"

# benchmark name -> (evalplus dataset key, loader attribute, expected task count)
DATASETS = {
    "mbpp_plus": ("mbpp", "get_mbpp_plus", 378),
    "humaneval_plus": ("humaneval", "get_human_eval_plus", 164),
}


def load_problems(benchmark: str) -> tuple[str, dict]:
    import evalplus.data

    dataset, loader_name, expected = DATASETS[benchmark]
    problems = getattr(evalplus.data, loader_name)()
    if len(problems) != expected:
        raise RuntimeError(
            f"Expected {expected} {benchmark} problems, found {len(problems)}"
        )
    return dataset, problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method_dir", type=Path)
    parser.add_argument("--benchmark", choices=sorted(DATASETS), required=True)
    parser.add_argument("--parallel", type=int)
    parser.add_argument("--test-details", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    samples = args.method_dir / "evalplus_samples.jsonl"
    records = args.method_dir / "records.jsonl"
    output = args.method_dir / "evalplus_results.json"
    if not samples.is_file() or not records.is_file():
        parser.error("method_dir must contain records.jsonl and evalplus_samples.jsonl")
    if output.exists() and not args.force:
        parser.error(f"{output} exists; use --force to recompute")
    if output.exists():
        output.unlink()

    # EvalPlus 0.3.1 requires exactly one sample for every official problem,
    # even when an experiment intentionally evaluates a paired subset. Pad the
    # evaluator input with official canonical solutions, then discard those
    # padding rows when merging scores back into our records. This preserves
    # exact EvalPlus execution for the generated subset without pretending the
    # padding is part of the experiment metric.
    from evalplus.evaluate import evaluate

    dataset, problems = load_problems(args.benchmark)
    generated = list(stream_jsonl(samples))
    generated_by_id = {str(row["task_id"]): row for row in generated}
    if len(generated_by_id) != len(generated):
        raise ValueError("evalplus_samples.jsonl contains duplicate task IDs")
    unknown = sorted(set(generated_by_id) - set(problems))
    if unknown:
        raise ValueError(f"Unknown {args.benchmark} task IDs: {unknown}")

    complete_samples = args.method_dir / "evalplus_complete_samples.jsonl"
    native_output = args.method_dir / "evalplus_complete_samples_eval_results.json"
    write_jsonl(
        complete_samples,
        [
            generated_by_id.get(
                task_id,
                {"task_id": task_id, "solution": problem["canonical_solution"]},
            )
            for task_id, problem in problems.items()
        ],
    )
    if native_output.exists():
        native_output.unlink()
    evaluate(
        dataset=dataset,
        samples=str(complete_samples),
        parallel=args.parallel,
        test_details=args.test_details,
    )
    native_output.replace(output)
    complete_samples.unlink()
    result = json.loads(output.read_text(encoding="utf-8"))
    by_task = {task_id: values[0] for task_id, values in result["eval"].items()}
    scored = []
    for row in stream_jsonl(records):
        task_id = str(row["example_id"])
        detail = by_task[task_id]
        base_pass = detail["base_status"] == PASS
        plus_pass = base_pass and detail["plus_status"] == PASS
        if plus_pass:
            failure = None
            execution_category = None
        elif not bool(row.get("syntax_valid")):
            execution_category = "syntax_error"
            failure = "syntax_error"
        elif detail["base_status"] == "timeout" or detail["plus_status"] == "timeout":
            execution_category = "timeout"
            failure = "timeout"
        elif not base_pass:
            execution_category = "wrong_answer_or_runtime_error"
            failure = f"base_{detail['base_status']}"
        else:
            execution_category = "plus_wrong_answer_or_runtime_error"
            failure = f"plus_{detail['plus_status']}"
        row.update(
            {
                "score": plus_pass,
                "failure_type": failure,
                "base_test_pass": base_pass,
                "plus_test_pass": plus_pass,
                "evalplus_base_status": detail["base_status"],
                "evalplus_plus_status": detail["plus_status"],
                "execution_failure_category": execution_category,
            }
        )
        scored.append(row)
    write_jsonl(args.method_dir / "records.scored.jsonl", scored)
    print(
        f"{args.benchmark} subset pass@1 ({len(scored)} generated tasks): "
        f"{sum(row['score'] for row in scored) / len(scored):.4f}"
    )


if __name__ == "__main__":
    main()
