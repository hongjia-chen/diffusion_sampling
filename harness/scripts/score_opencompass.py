#!/usr/bin/env python3
"""Grade code the way SDAR's OpenCompass config does: base tests, its extraction.

    python score_opencompass.py --benchmark mbpp_sanitized <method_dir>
    python score_opencompass.py --benchmark humaneval <method_dir>
    python score_opencompass.py --benchmark humaneval_plus <method_dir>   # after score_evalplus.py

mbpp_sanitized  `MBPPEvaluator`: the extracted answer (opencompass.mbpp_answer,
                already in `parsed_output`) + "\\n" + the task's 3 asserts
                (record["source"]["test_list"]), 9 s limit.
humaneval       `HumanEvalEvaluator` (human-eval): problem prompt + the first fenced
                block (already in `parsed_output`) + test + check(), 3 s limit.
humaneval_plus  (legacy evalplus records) the same grading: problem prompt + the first fenced
                block of the completion + the original HumanEval test + check(),
                3 s limit. Adds `opencompass_pass` to records.scored.jsonl and makes
                it `score`; evalplus's `base_test_pass` / `plus_test_pass` stay.

Both evaluators run `exec(program, {})` with I/O swallowed under a time limit,
and so does each subprocess here: a `__main__` guard, input() or exit() fail
exactly as they do there. Writes records.scored.jsonl.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Runnable from any directory: the package lives next to scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sdar_harness.io import stream_jsonl, write_jsonl
from sdar_harness.opencompass import humaneval_answer

LIMITS = {"mbpp_sanitized": 9.0, "humaneval": 3.0, "humaneval_plus": 3.0}

DRIVER = r"""
import contextlib, io, os, signal, sys
source = open(sys.argv[1]).read()
class Expired(Exception):
    pass
def expire(*_):
    raise Expired()
signal.signal(signal.SIGALRM, expire)
signal.setitimer(signal.ITIMER_REAL, float(sys.argv[2]))
status = 0
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(source, {})
except Expired:
    status = 4
except AssertionError:
    status = 2
except BaseException:
    status = 3
signal.setitimer(signal.ITIMER_REAL, 0)
os._exit(status)
"""
STATUS = {0: "pass", 2: "wrong_answer", 3: "failed", 4: "timeout"}


def execute(program: str, limit: float) -> str:
    with tempfile.TemporaryDirectory() as workdir:
        path = Path(workdir) / "program.py"
        path.write_text(program, encoding="utf-8")
        try:
            done = subprocess.run(
                [sys.executable, "-c", DRIVER, str(path), str(limit)],
                cwd=workdir, stdin=subprocess.DEVNULL, capture_output=True,
                timeout=limit + 10,
            )
        except subprocess.TimeoutExpired:
            return "timeout"
    return STATUS.get(done.returncode, "failed")


def programs(benchmark: str, rows: list[dict]) -> list[str]:
    if benchmark == "mbpp_sanitized":
        # OpenCompass's test_list_2: the three asserts joined by newlines. The
        # runner keeps them in record["source"] (it drops the row's `target`).
        return [row["parsed_output"] + "\n" + "\n".join(row["source"]["test_list"]) for row in rows]
    if benchmark == "humaneval":
        # OpenAI HumanEval rows carry the problem with them (record["source"]).
        return [
            row["source"]["problem_prompt"] + row["parsed_output"] + "\n"
            + row["source"]["test"] + "\n" + f"check({row['source']['entry_point']})"
            for row in rows
        ]
    from evalplus.data import get_human_eval_plus

    problems = get_human_eval_plus()
    out = []
    for row in rows:
        problem = problems[str(row["example_id"])]
        completion = humaneval_answer(row["raw_completion"])
        out.append(
            problem["prompt"] + completion + "\n" + problem["test"] + "\n"
            + f"check({problem['entry_point']})"
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method_dir", type=Path)
    parser.add_argument("--benchmark", choices=sorted(LIMITS), required=True)
    parser.add_argument("--parallel", type=int, default=8)
    args = parser.parse_args()
    scored_path = args.method_dir / "records.scored.jsonl"
    if args.benchmark == "humaneval_plus":
        if not scored_path.is_file():
            parser.error("run score_evalplus.py first: humaneval_plus keeps its base/plus fields")
        source = scored_path
    else:
        source = args.method_dir / "records.jsonl"
    rows = list(stream_jsonl(source))
    if not rows or any(row["benchmark"] != args.benchmark for row in rows):
        parser.error(f"{source} does not hold {args.benchmark} records")
    limit = LIMITS[args.benchmark]
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        statuses = list(pool.map(lambda program: execute(program, limit), programs(args.benchmark, rows)))
    for row, status in zip(rows, statuses):
        passed = status == "pass"
        row.update({
            "score": passed,
            "failure_type": None if passed else f"opencompass_{status}",
            "opencompass_pass": passed,
            "opencompass_status": status,
        })
    write_jsonl(scored_path, rows)
    print(f"{args.benchmark} OpenCompass pass@1 ({len(rows)} tasks): "
          f"{sum(s == 'pass' for s in statuses) / len(rows):.4f}")


if __name__ == "__main__":
    main()
