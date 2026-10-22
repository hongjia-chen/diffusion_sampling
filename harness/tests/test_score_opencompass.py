"""SDAR's (OpenCompass's) code grading: extraction and execution semantics."""
from __future__ import annotations

import sys
from pathlib import Path

from sdar_harness.opencompass import humaneval_answer, mbpp_answer
from sdar_harness.scoring import score_record

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from score_opencompass import execute  # noqa: E402


def test_mbpp_answer_takes_the_fenced_code() -> None:
    text = "Here you go:\n```python\ndef f(x):\n    return x + 1\n```\nDone."
    assert mbpp_answer(text) == "def f(x):\n    return x + 1"


def test_humaneval_answer_is_the_first_block_or_the_text() -> None:
    text = "```python\ndef f():\n    return 1\n```\n```python\nprint(f())\n```"
    assert humaneval_answer(text) == "def f():\n    return 1\n"
    assert humaneval_answer("   return 1\n") == "return 1\n"


def test_mbpp_sanitized_defers_its_score_to_execution() -> None:
    record = {"benchmark": "mbpp_sanitized", "target": "assert f(1) == 2"}
    result = score_record(record, "```python\ndef f(x):\n    return x + 1\n```")
    assert result["score"] is None
    assert result["parsed_output"] == "def f(x):\n    return x + 1"
    assert result["syntax_valid"] is True


def test_execution_statuses_match_opencompass() -> None:
    assert execute("def f(x):\n    return x + 1\nassert f(1) == 2", 9.0) == "pass"
    assert execute("def f(x):\n    return x\nassert f(1) == 2", 9.0) == "wrong_answer"
    assert execute("raise ValueError('boom')", 9.0) == "failed"
    assert execute("while True:\n    pass", 1.0) == "timeout"
    # Under exec(program, {}), __name__ resolves to builtins' "builtins", so a
    # __main__ block is skipped; stdin is unreadable there, closed here.
    assert execute("if __name__ == '__main__':\n    raise SystemExit(1)", 9.0) == "pass"
    assert execute("input()", 9.0) == "failed"
    assert execute("exit()", 9.0) == "failed"
    assert execute("print('noise')\nassert True", 9.0) == "pass"


# ------------------------------------------------- GSM8K (MATHEvaluator v2)

def test_opencompass_gsm8k_grades_the_last_boxed_answer() -> None:
    from sdar_harness.opencompass import math_equiv_v2
    record = {"benchmark": "gsm8k", "protocol": "opencompass", "target": "18"}
    good = score_record(record, "16 - 3 - 4 = 9 eggs, 9 * 2 = 18.\nThe answer is $\\boxed{18}$.")
    assert good["score"] is True and good["parsed_output"] == "18"
    assert score_record(record, "\\boxed{18.00}")["score"] is True
    assert score_record(record, "\\boxed{\\$18}")["score"] is True
    wrong = score_record(record, "so it is \\boxed{9}")
    assert wrong["score"] is False and wrong["failure_type"] == "wrong_answer"
    unboxed = score_record(record, "16 - 7 = 9. She makes 9 * 2 dollars")
    assert unboxed["score"] is False and unboxed["failure_type"] == "no_boxed_answer"
    assert math_equiv_v2("\\frac12", "\\frac{1}{2}") and math_equiv_v2("x=5", "5")


def test_the_protocol_field_picks_the_grader() -> None:
    completion = "Each costs 2, so \\boxed{18}. Check: 9 + 9 = 18, not 7"
    legacy = score_record({"benchmark": "gsm8k", "target": "18"}, completion)
    opencompass = score_record({"benchmark": "gsm8k", "protocol": "opencompass",
                                "target": "18"}, completion)
    assert legacy["parsed_output"] == "7" and legacy["score"] is False   # last number
    assert opencompass["parsed_output"] == "18" and opencompass["score"] is True


def test_humaneval_programs_come_from_the_record() -> None:
    from score_opencompass import programs
    row = {
        "parsed_output": "def add(a, b):\n    return a + b\n",
        "source": {"problem_prompt": "def add(a, b):\n    \"\"\"Add.\"\"\"\n",
                   "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
                   "entry_point": "add"},
    }
    (program,) = programs("humaneval", [row])
    assert execute(program, 3.0) == "pass"
    row["parsed_output"] = "def add(a, b):\n    return a - b\n"
    assert execute(programs("humaneval", [row])[0], 3.0) == "wrong_answer"


def test_mbpp_programs_come_from_the_record_source() -> None:
    # A real record: the runner drops the dataset row's `target` and keeps the
    # asserts under `source` (this shape broke the first smoke run).
    from score_opencompass import programs
    row = {"parsed_output": "def f(x):\n    return x + 1",
           "source": {"test_list": ["assert f(1) == 2", "assert f(2) == 3", "assert f(0) == 1"]}}
    (program,) = programs("mbpp_sanitized", [row])
    assert program == "def f(x):\n    return x + 1\nassert f(1) == 2\nassert f(2) == 3\nassert f(0) == 1"
    assert execute(program, 9.0) == "pass"
