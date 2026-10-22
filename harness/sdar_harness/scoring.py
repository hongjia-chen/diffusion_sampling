"""Answer extraction and grading.

Records with `"protocol": "opencompass"` (../data/opencompass/, the default
since 2026-10-03) are graded exactly as SDAR's OpenCompass config grades them
(opencompass.py):

    gsm8k           math_postprocess_v2 (last \\boxed), MATHEvaluator v2 equivalence
    math_500        math_verify (MATHVerifyEvaluator)
    humaneval       first fenced block; base tests via score_opencompass.py
    mbpp_sanitized  MBPPEvaluator extraction; its 3 asserts via score_opencompass.py

Legacy records (exp04-06 prompts, no protocol field):

    gsm8k           lm-eval `until` truncation, then flexible-extract
    math_500        "Problem:" truncation, then math_verify
    mbpp_plus       evalplus `sanitize`, then evalplus execution (base headline)
    humaneval_plus  evalplus as above; headline is OpenCompass's base-test grading

Schema JSON, the fourth benchmark of the previous experiment, is deliberately
absent. See ../README.md.
"""

from __future__ import annotations

import ast
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from sdar_harness.opencompass import (
    extract_boxed_answer,
    humaneval_answer,
    math_equiv_v2,
    math_postprocess_v2,
    math_verify_agree,
    mbpp_answer,
)

OPENCOMPASS = "opencompass"


# ----------------------------------------------------------------------
# GSM8K
# ----------------------------------------------------------------------

# lm-eval `gsm8k_cot` extraction. `strict-match` looks for the demonstrated
# "The answer is N" phrasing; `flexible-extract` takes the last number in the
# completion (group_select: -1). At num_fewshot=0 the model has never seen the
# strict phrasing modelled, so flexible-extract is the meaningful metric and is
# what `score` reports.
GSM8K_UNTIL = ("Q:", "</s>", "<|im_end|>")
STRICT_ANSWER = re.compile(r"The answer is (\-?[0-9\.\,]+)")
FLEXIBLE_ANSWER = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def truncate_at_stop_strings(text: str, stops: Iterable[str]) -> str:
    """Apply lm-eval's `until` list as post-hoc truncation.

    A diffusion model cannot halt mid-generation, so Dream and ADJUST both cut
    the decoded text at these markers afterwards instead. Without it, run-on
    into a fabricated next "Q:" turn would hand flexible-extract a number from
    the hallucinated follow-up, since it takes the last match.
    """
    for stop in stops:
        text = text.split(stop)[0]
    return text


def normalize_number(value: str) -> str | None:
    cleaned = value.strip().replace(",", "").replace("$", "")
    cleaned = cleaned.rstrip(". ")
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"-0", ""} else normalized


def score_gsm8k(completion: str, target: str) -> dict[str, Any]:
    text = truncate_at_stop_strings(completion, GSM8K_UNTIL)
    expected = normalize_number(target)

    # lm-eval takes the LAST match (group_select: -1). Its regex can match text
    # that is not a number at all: the alternation `[$0-9.,]{2,}` happily eats
    # the trailing `$.` of a LaTeX answer like `The answer is $\boxed{15}$.`,
    # which then fails to parse and discards an answer that was right there.
    # "The last number" can only mean the last match that IS a number, so scan
    # backwards to the first one that parses. On a completion whose final match
    # already parses -- every Dream completion in the ported frontier -- this is
    # exactly the old behaviour.
    flexible_matches = FLEXIBLE_ANSWER.findall(text)
    flexible = None
    for candidate in reversed(flexible_matches):
        flexible = normalize_number(candidate[0] or candidate[1])
        if flexible is not None:
            break
    strict_match = STRICT_ANSWER.search(text)
    strict = normalize_number(strict_match.group(1)) if strict_match else None

    correct = flexible is not None and expected is not None and flexible == expected
    if not flexible_matches:
        failure = "no_number_in_completion"
    elif flexible is None:
        failure = "unparseable_number"
    elif not correct:
        failure = "wrong_answer"
    else:
        failure = None
    return {
        "parsed_output": flexible,
        "score": bool(correct),
        "failure_type": failure,
        "answer_extraction_success": flexible is not None,
        "strict_match_output": strict,
        "strict_match": bool(
            strict is not None and expected is not None and strict == expected
        ),
        "truncated_by_stop_string": text != completion,
    }


# ----------------------------------------------------------------------
# MATH-500
# ----------------------------------------------------------------------

MATH_UNTIL = ("Problem:", "</s>", "<|im_end|>")

_BOXED = "\\boxed"


def extract_boxed(text: str) -> str | None:
    """Content of the LAST ``\\boxed{...}``, matched by brace depth.

    A regex cannot do this: MATH answers routinely nest braces
    (``\\boxed{\\frac{1}{2}}``), and a non-greedy regex stops at the first
    ``}`` while a greedy one runs to the end of the completion. Taking the last
    box rather than the first follows the convention that a model which revises
    its answer means the final one.
    """
    start = text.rfind(_BOXED)
    if start == -1:
        return None
    cursor = start + len(_BOXED)
    # `\boxed 5` (no braces) is legal LaTeX for a single token.
    while cursor < len(text) and text[cursor] == " ":
        cursor += 1
    if cursor >= len(text):
        return None
    if text[cursor] != "{":
        return text[cursor]
    depth = 0
    for index in range(cursor, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[cursor + 1 : index]
    # Unbalanced: the canvas ran out mid-answer. Return what there is rather
    # than dropping it, so the failure shows up as a wrong answer with visible
    # content instead of an indistinguishable extraction failure.
    return text[cursor + 1 :]


_TEXT_WRAPPERS = re.compile(r"\\(?:text|mbox|textbf|mathrm|mathbf)\s*\{([^{}]*)\}")
_UNIT_SUFFIX = re.compile(
    r"\\?(?:\\,|\\!|\\ )*\s*(?:dollars?|cents?|degrees?|units?|cm|mm|km|kg|ft|in)\s*$",
    re.IGNORECASE,
)


def normalize_math(answer: str) -> str:
    """Canonicalize a LaTeX answer enough to compare two of them as strings.

    This is a normalizer, not a computer algebra system: it will call
    ``\\frac{1}{2}`` and ``0.5`` different unless both reduce to a number, and
    it will not expand ``(x+1)^2``. It under-credits rather than over-credits.
    Since 2026-10-03 it only feeds the `string_match` field; `score` comes from
    `opencompass.math_verify_agree`.
    """
    text = answer.strip()
    for _ in range(3):  # unwrap nesting like \text{\mathrm{x}}
        replaced = _TEXT_WRAPPERS.sub(r"\1", text)
        if replaced == text:
            break
        text = replaced
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("dfrac", "frac").replace("tfrac", "frac")
    text = text.replace("^{\\circ}", "").replace("^\\circ", "")
    text = text.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    text = _UNIT_SUFFIX.sub("", text)
    text = text.replace(" ", "")
    # "0.50" == ".5" == "1/2" only via the numeric path below; here just drop
    # a trailing period and a redundant leading zero-width.
    text = text.rstrip(".")
    if text.startswith("{") and text.endswith("}") and text.count("{") == 1:
        text = text[1:-1]
    return text


def _as_number(text: str) -> Decimal | None:
    """Numeric value of an answer, if it has one. Handles ``\\frac{a}{b}``."""
    fraction = re.fullmatch(r"\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", text)
    if fraction:
        try:
            numerator = Decimal(fraction.group(1))
            denominator = Decimal(fraction.group(2))
            return numerator / denominator if denominator != 0 else None
        except InvalidOperation:
            return None
    plain = re.fullmatch(r"(-?[\d,]*\.?\d+)", text)
    if not plain:
        return None
    try:
        value = Decimal(plain.group(1).replace(",", ""))
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def math_answers_agree(predicted: str, target: str) -> bool:
    left, right = normalize_math(predicted), normalize_math(target)
    if left == right:
        return True
    left_number, right_number = _as_number(left), _as_number(right)
    if left_number is not None and right_number is not None:
        return left_number == right_number
    return False


def score_gsm8k_opencompass(completion: str, target: str) -> dict[str, Any]:
    """gsm8k_0shot_v2_gen_17d799: math_postprocess_v2, MATHEvaluator(version='v2').

    `target` is already `gsm8k_dataset_postprocess(answer)` (the number after
    ####, commas removed). No stop-string truncation: OpenCompass cuts only at
    the model's stop tokens, which the decoder already did.
    """
    boxed = extract_boxed_answer(completion, strip_double_curly_brace=True)
    predicted = math_postprocess_v2(completion)
    correct = math_equiv_v2(predicted, target)
    if correct:
        failure = None
    elif not boxed:
        failure = "no_boxed_answer"
    else:
        failure = "wrong_answer"
    return {
        "parsed_output": predicted,
        "score": bool(correct),
        "failure_type": failure,
        "answer_extraction_success": bool(boxed),
    }


def score_math_500(completion: str, target: str, *, truncate: bool = True) -> dict[str, Any]:
    # OpenCompass does not cut at "Problem:"; the legacy prompt, which ends
    # in "Solution:", needs it against a hallucinated next problem.
    text = truncate_at_stop_strings(completion, MATH_UNTIL) if truncate else completion
    extracted = extract_boxed(text)
    correct, verified_answer = math_verify_agree(text, target)
    if correct:
        failure = None
    elif extracted is None:
        failure = "no_boxed_answer"
    else:
        failure = "wrong_answer"
    return {
        "parsed_output": extracted,
        "score": correct,
        "failure_type": failure,
        "answer_extraction_success": extracted is not None,
        "math_verify_answer": verified_answer,
        # The pre-2026-10-03 string scorer, kept so exp05/06 numbers stay
        # traceable. Never the headline.
        "string_match": extracted is not None and math_answers_agree(extracted, target),
        "normalized_output": None if extracted is None else normalize_math(extracted),
        "normalized_target": normalize_math(target),
        "truncated_by_stop_string": text != completion,
    }


# ----------------------------------------------------------------------
# Code benchmarks (MBPP+, HumanEval+)
# ----------------------------------------------------------------------

CODE_BENCHMARKS = {"mbpp_plus": "mbpp", "humaneval_plus": "humaneval"}


def score_code_syntax(completion: str) -> dict[str, Any]:
    try:
        ast.parse(completion)
        return {"syntax_valid": True, "syntax_error": None}
    except SyntaxError as error:
        return {
            "syntax_valid": False,
            "syntax_error": f"{error.msg} (line {error.lineno})",
        }
    except (MemoryError, RecursionError) as error:
        # Python 3.10's parser raises MemoryError, not SyntaxError, on very deep
        # nesting such as a long run of unary operators ("----...1"), which a
        # runaway completion can be (8B-b16 fixed:8, HumanEval, 2026-10-16). It
        # cannot compile either, so the OpenCompass grade fails it too.
        return {
            "syntax_valid": False,
            "syntax_error": f"too deeply nested to parse ({type(error).__name__})",
        }


def _sanitize_once(text: str, entry_point: str) -> str:
    try:
        from evalplus.sanitize import sanitize

        return sanitize(text, entrypoint=entry_point)
    except ImportError:
        # Lightweight fallback for environments where only unit tests are run.
        fenced = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
        return fenced.group(1).strip() if fenced else text.strip()


def _usable(solution: str, entry_point: str) -> bool:
    return bool(solution.strip()) and f"def {entry_point}" in solution


# Top-level imports only: column 0, so a line inside the function body is not
# hoisted out of its scope.
_TOP_LEVEL_IMPORT = re.compile(
    r"^(?:import[ \t]+\S.*|from[ \t]+\S+[ \t]+import[ \t]+\S.*)$", re.M
)


def _carry_imports(solution: str, prompt: str) -> tuple[str, bool]:
    """Prepend the prompt's imports that the solution does not already have.

    HumanEval's prompt carries the signature's imports (`from typing import
    List`), and the sanitized solution is executed on its own -- so a model
    that re-emits the whole `def` without the import produces code that PARSES
    but raises `NameError` on the annotation. `_usable` is a substring check,
    so it cannot see this: the task is scored as a failure while the code is
    correct. Measured on the Dream Phase 1 run: 3-6 such tasks per arm.

    Imports are idempotent, so a carried-over line that turns out to be unused
    is harmless; only an actual addition is reported, so the diagnostic stays
    meaningful.
    """
    missing = [
        line
        for line in _TOP_LEVEL_IMPORT.findall(prompt)
        if line not in solution
    ]
    if not missing:
        return solution, False
    return "\n".join(missing) + "\n" + solution, True


def sanitize_code(
    completion: str, entry_point: str, prompt: str | None = None
) -> tuple[str, str | None]:
    """Extract a runnable solution. Returns `(solution, repair)`.

    HumanEval's prompt *is* the function signature and docstring, so a model
    that continues it in completion style emits only an indented body -- with
    no `def` for the sanitizer to find. Measured: all 164 HumanEval+ canonical
    solutions survive sanitization when presented as a whole fenced function,
    and **0 of 164** survive when presented as a bare body. An arm that decoded
    in completion style would score 0.00 on content that is perfectly correct.

    That is the Schema JSON failure of the previous experiment exactly: a
    scoring artifact indistinguishable from a generation failure, discovered
    only after the GPU hours were spent. So try the least aggressive reading
    first, fall back to prepending the prompt, and record which one worked --
    a run whose scores lean on `prompt_prefix` is a run decoding in completion
    style, and that should be visible rather than silent.

    MBPP+ is unaffected (its prompt asks for a whole function) but goes through
    the same path so the diagnostic exists for both.
    """
    direct = _sanitize_once(completion, entry_point)
    if prompt is None:
        return direct, None
    if _usable(direct, entry_point):
        # Usable, but possibly missing the imports its own signature needs.
        repaired, added = _carry_imports(direct, prompt)
        return repaired, ("prompt_imports" if added else None)
    joined = _sanitize_once(prompt + completion, entry_point)
    if _usable(joined, entry_point):
        return joined, "prompt_prefix"
    return direct, None


# ----------------------------------------------------------------------

def score_record(record: dict[str, Any], completion: str) -> dict[str, Any]:
    benchmark = record["benchmark"]
    opencompass = record.get("protocol") == OPENCOMPASS
    if benchmark == "gsm8k":
        if opencompass:
            return score_gsm8k_opencompass(completion, str(record["target"]))
        return score_gsm8k(completion, str(record["target"]))
    if benchmark == "math_500":
        return score_math_500(completion, str(record["target"]), truncate=not opencompass)
    if benchmark in ("humaneval", "mbpp_sanitized"):
        # SDAR's code benchmarks. Executed afterwards by score_opencompass.py.
        solution = humaneval_answer(completion) if benchmark == "humaneval" else mbpp_answer(completion)
        return {
            "parsed_output": solution,
            "score": None,
            "failure_type": None,
            **score_code_syntax(solution),
        }
    if benchmark in CODE_BENCHMARKS:
        # `score` stays None until evalplus executes the tests; the run itself
        # never claims a pass rate it has not measured.
        solution, repair = sanitize_code(
            completion, str(record["entry_point"]), record.get("prompt")
        )
        return {
            "parsed_output": solution,
            "score": None,
            "failure_type": None,
            "code_repair": repair,
            **score_code_syntax(solution),
        }
    raise ValueError(f"Unknown benchmark: {benchmark}")
