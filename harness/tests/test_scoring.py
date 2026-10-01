from __future__ import annotations

from sdar_harness.scoring import (
    extract_boxed,
    math_answers_agree,
    normalize_math,
    normalize_number,
    score_gsm8k,
    score_math_500,
    score_record,
)


# ---------------------------------------------------------------- GSM8K

def test_gsm8k_flexible_extract_takes_the_last_number() -> None:
    result = score_gsm8k("work 12\nrevision\nThe answer is 1,200.00", "1200")
    assert result["score"] is True
    assert result["parsed_output"] == "1200"
    assert result["strict_match"] is True


def test_gsm8k_scores_a_bare_answer_without_the_strict_phrasing() -> None:
    # At num_fewshot=0 the model has never seen "The answer is N" demonstrated,
    # so flexible-extract is the metric that carries the result.
    result = score_gsm8k("193", "193")
    assert result["score"] is True
    assert result["strict_match"] is False


def test_gsm8k_stop_strings_cut_run_on_into_a_fabricated_next_turn() -> None:
    completion = "So 72 clips.\nThe answer is 72.\nQ: Weng earns $12 an hour\nA: 0.2 per minute"
    result = score_gsm8k(completion, "72")
    # Without the `until` truncation flexible-extract would return 0.2, taken
    # from the hallucinated follow-up question.
    assert result["score"] is True
    assert result["parsed_output"] == "72"
    assert result["truncated_by_stop_string"] is True


def test_gsm8k_reports_a_completion_with_no_number() -> None:
    result = score_gsm8k("I am not sure how to solve this.", "4")
    assert result["score"] is False
    assert result["failure_type"] == "no_number_in_completion"


def test_number_normalization_rejects_non_numbers() -> None:
    assert normalize_number("$1,200.0") == "1200"
    assert normalize_number("twelve") is None


# ------------------------------------------------------------- MATH-500

def test_boxed_extraction_handles_nested_braces() -> None:
    assert extract_boxed(r"so \boxed{\frac{1}{2}}.") == r"\frac{1}{2}"


def test_boxed_extraction_takes_the_last_box() -> None:
    # A model that revises its answer means the final one.
    assert extract_boxed(r"first \boxed{3}, actually \boxed{4}") == "4"


def test_boxed_extraction_handles_a_braceless_box() -> None:
    assert extract_boxed(r"\boxed 5") == "5"


def test_boxed_extraction_returns_none_when_absent() -> None:
    assert extract_boxed("the answer is seven") is None


def test_unbalanced_box_returns_its_content_rather_than_nothing() -> None:
    # The canvas ran out mid-answer. Surfacing the partial text makes this a
    # visible wrong answer instead of an indistinguishable extraction failure.
    assert extract_boxed(r"\boxed{\frac{1}{2") == r"\frac{1}{2"


def test_math_normalization_strips_latex_noise() -> None:
    assert normalize_math(r"\left( \dfrac{1}{2} \right)") == r"(\frac{1}{2})"
    assert normalize_math(r"\text{12 dollars}") == "12"
    assert normalize_math(r"45^\circ") == "45"


def test_math_numeric_equivalence_bridges_fraction_and_decimal() -> None:
    assert math_answers_agree(r"\frac{1}{2}", "0.5")
    assert math_answers_agree("2", "2.00")


def test_math_comparison_is_conservative_not_symbolic() -> None:
    # A normalizer, not a CAS: this under-credits rather than over-credits.
    assert not math_answers_agree("(x+1)^2", "x^2+2x+1")


def test_math_500_scoring_end_to_end() -> None:
    record = {"benchmark": "math_500", "target": r"\frac{1}{2}"}
    good = score_record(record, r"Halving gives \boxed{\dfrac{1}{2}}.")
    assert good["score"] is True
    assert good["failure_type"] is None
    bad = score_record(record, "The answer is one half.")
    assert bad["score"] is False
    assert bad["failure_type"] == "no_boxed_answer"
    assert bad["answer_extraction_success"] is False


# ------------------------------------------------------- code benchmarks

def test_code_benchmarks_defer_their_score_to_evalplus() -> None:
    for benchmark in ("mbpp_plus", "humaneval_plus"):
        record = {"benchmark": benchmark, "entry_point": "f"}
        result = score_record(record, "```python\ndef f():\n    return 1\n```")
        # `score` stays None until evalplus executes the tests; the run never
        # claims a pass rate it has not measured.
        assert result["score"] is None
        assert result["syntax_valid"] is True
        assert "def f()" in result["parsed_output"]


def test_code_syntax_failure_is_recorded_not_raised() -> None:
    record = {"benchmark": "humaneval_plus", "entry_point": "f"}
    result = score_record(record, "def f(:\n  pass")
    assert result["syntax_valid"] is False
    assert result["syntax_error"]


def test_humaneval_completion_style_is_repaired_and_recorded() -> None:
    """The failure that would have zeroed an arm silently.

    HumanEval's prompt IS the signature, so a completion-style model emits a
    bare body with no `def` for the sanitizer to find. Measured on the real
    benchmark: 0 of 164 canonical solutions are usable without this fallback,
    164 of 164 with it.
    """
    prompt = "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n"
    body_only = "    return a + b\n"
    record = {"benchmark": "humaneval_plus", "entry_point": "add", "prompt": prompt}

    repaired = score_record(record, body_only)
    assert repaired["syntax_valid"] is True
    assert "def add" in repaired["parsed_output"]
    assert repaired["code_repair"] == "prompt_prefix"

    whole = f"```python\n{prompt}{body_only}```"
    direct = score_record(record, whole)
    assert "def add" in direct["parsed_output"]
    # The least aggressive reading worked, so nothing was repaired.
    assert direct["code_repair"] is None


def test_repair_is_not_applied_when_the_prompt_is_absent() -> None:
    from sdar_harness.scoring import sanitize_code

    solution, repair = sanitize_code("    return 1\n", "f", None)
    assert repair is None


def test_gsm8k_reads_through_a_trailing_latex_delimiter() -> None:
    """`The answer is $\\boxed{15}$.` must score, not fail to parse.

    SDAR-1.7B-Chat answers GSM8K in this LaTeX form on 25 of 30 sampled tasks.
    lm-eval's flexible regex matches the trailing `$.` as its LAST match, which
    normalizes to nothing -- so before this fix the answer sitting right there
    was discarded and the task was scored `unparseable_number`.
    """
    result = score_gsm8k("Bill was 45 - 30 = 15 years old.\nThe answer is $\\boxed{15}$.\n", "15")
    assert result["parsed_output"] == "15"
    assert result["score"] is True
    assert result["answer_extraction_success"] is True


def test_gsm8k_reads_through_a_trailing_double_period() -> None:
    """The same failure with `..`, which Dream produces on run-on completions."""
    result = score_gsm8k("Robert went down the slide 4 * 21 = 84 times..", "84")
    assert result["parsed_output"] == "84"
    assert result["score"] is True


def test_gsm8k_still_reports_a_completion_that_is_only_punctuation() -> None:
    """Walking back must not invent an answer where there is no number."""
    result = score_gsm8k("The answer is $.", "15")
    assert result["parsed_output"] is None
    assert result["score"] is False
    assert result["failure_type"] == "unparseable_number"


def test_sanitize_carries_the_prompts_typing_import_into_the_solution() -> None:
    """A re-emitted `def` that drops `from typing import List` must still run.

    The sanitized solution is executed on its own, so an annotation naming a
    type the prompt imported raises NameError -- correct code scored as a
    failure. `_usable` is a substring check and cannot see it.
    """
    from sdar_harness.scoring import sanitize_code

    prompt = "from typing import List\n\n\ndef factorize(n: int) -> List[int]:\n"
    completion = "def factorize(n: int) -> List[int]:\n    return []\n"
    solution, repair = sanitize_code(completion, "factorize", prompt)

    assert repair == "prompt_imports"
    assert "from typing import List" in solution
    exec(compile(solution, "<solution>", "exec"), {})  # would NameError before


def test_sanitize_does_not_report_a_repair_when_the_import_is_already_there() -> None:
    from sdar_harness.scoring import sanitize_code

    prompt = "from typing import List\n\n\ndef f(n: int) -> List[int]:\n"
    completion = "from typing import List\ndef f(n: int) -> List[int]:\n    return []\n"
    solution, repair = sanitize_code(completion, "f", prompt)

    assert repair is None
    assert solution.count("from typing import List") == 1


def test_sanitize_does_not_hoist_an_import_from_inside_a_function_body() -> None:
    """Only column-0 imports are carried; an indented one stays in its scope."""
    from sdar_harness.scoring import sanitize_code

    prompt = "def f():\n    import math\n"
    completion = "def f():\n    return 1\n"
    solution, repair = sanitize_code(completion, "f", prompt)

    assert repair is None
    assert "import math" not in solution
