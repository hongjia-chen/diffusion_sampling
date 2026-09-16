from __future__ import annotations

import ast
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from jsonschema import Draft202012Validator


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

    flexible_matches = FLEXIBLE_ANSWER.findall(text)
    flexible = (
        normalize_number(flexible_matches[-1][0] or flexible_matches[-1][1])
        if flexible_matches
        else None
    )
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


def _field_accuracy(expected: Any, actual: Any) -> tuple[int, int]:
    if isinstance(expected, dict):
        correct = 0
        total = 0
        for key, value in expected.items():
            child_correct, child_total = _field_accuracy(
                value, actual.get(key) if isinstance(actual, dict) else object()
            )
            correct += child_correct
            total += child_total
        return correct, total
    return (int(expected == actual), 1)


# Markdown fences around JSON. The instruct checkpoint emits them at some
# sampling settings and not others -- the fixed arm of outputs/v2 fenced 100% of
# its Schema JSON completions while the adaptive arm fenced 7-50%, rising with
# delta. Parsing the raw string therefore did not measure JSON quality, it
# measured fence rate, and it scored the entire fixed arm 0.00 on content that
# was byte-identical to adaptive completions scoring 1.00.
#
# Stripping them is consistent with how the other two benchmarks already work:
# MBPP+ goes through evalplus's `sanitize`, which unwraps ```python fences, and
# GSM8K applies lm-eval's `until` truncation. Schema JSON was the odd one out.
_FENCE_OPEN = re.compile(r"\A\s*```[^\S\n]*[A-Za-z0-9_+-]*[^\S\n]*\n?")
_FENCE_CLOSE = re.compile(r"```\s*\Z")


def _strip_fence(text: str) -> str:
    without_open = _FENCE_OPEN.sub("", text)
    if without_open == text:
        return text
    return _FENCE_CLOSE.sub("", without_open).strip()


def _brace_span(text: str) -> str:
    """The outermost `{`...`}` span, ignoring prose on either side."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def extract_json_object(completion: str) -> tuple[Any, bool, str | None]:
    """Parse a JSON object out of a completion.

    Returns `(parsed, parse_valid, repair)`. `repair` names the least
    aggressive transform that made the text parse, and is `None` when the raw
    completion already parsed. Recording it keeps the repair auditable rather
    than silent: a run whose scores depend heavily on `brace_span` is a run
    whose model is not actually emitting clean JSON.
    """
    if not isinstance(completion, str):
        return None, False, None
    text = completion.strip()
    candidates: list[tuple[str, str | None]] = [(text, None)]
    unfenced = _strip_fence(text)
    if unfenced != text:
        candidates.append((unfenced, "fence"))
    for source, label in list(candidates):
        spanned = _brace_span(source)
        if spanned != source:
            candidates.append((spanned, "brace_span" if label is None else f"{label}+brace_span"))
    for candidate, repair in candidates:
        try:
            return json.loads(candidate), True, repair
        except (json.JSONDecodeError, TypeError):
            continue
    return None, False, None


def score_schema_json(
    completion: str,
    target: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    parsed, parse_valid, repair = extract_json_object(completion)
    schema_errors = (
        list(Draft202012Validator(schema).iter_errors(parsed)) if parse_valid else []
    )
    schema_valid = parse_valid and not schema_errors
    exact = schema_valid and parsed == target
    correct_fields, total_fields = _field_accuracy(target, parsed)
    missing = (
        sorted(set(schema.get("required", [])) - set(parsed))
        if isinstance(parsed, dict)
        else list(schema.get("required", []))
    )
    extra = (
        sorted(set(parsed) - set(schema.get("properties", {})))
        if isinstance(parsed, dict)
        else []
    )
    if not parse_valid:
        failure = "json_parse_error"
    elif not schema_valid:
        failure = "schema_violation"
    elif not exact:
        failure = "semantic_mismatch"
    else:
        failure = None
    return {
        "parsed_output": parsed,
        "score": bool(exact),
        "failure_type": failure,
        "json_parse_valid": parse_valid,
        "json_repair": repair,
        "schema_valid": schema_valid,
        "field_accuracy": correct_fields / total_fields if total_fields else 1.0,
        "missing_required_fields": missing,
        "extra_fields": extra,
        "constraint_violations": [error.message for error in schema_errors],
    }


def score_mbpp_syntax(completion: str) -> dict[str, Any]:
    try:
        ast.parse(completion)
        return {"syntax_valid": True, "syntax_error": None}
    except SyntaxError as error:
        return {
            "syntax_valid": False,
            "syntax_error": f"{error.msg} (line {error.lineno})",
        }


def sanitize_mbpp(completion: str, entry_point: str) -> str:
    try:
        from evalplus.sanitize import sanitize

        return sanitize(completion, entrypoint=entry_point)
    except ImportError:
        # Lightweight fallback for environments where only unit tests are run.
        fenced = re.search(r"```(?:python)?\s*(.*?)```", completion, re.DOTALL)
        return fenced.group(1).strip() if fenced else completion.strip()


def score_record(record: dict[str, Any], completion: str) -> dict[str, Any]:
    benchmark = record["benchmark"]
    if benchmark == "gsm8k":
        return score_gsm8k(completion, str(record["target"]))
    if benchmark == "schema_json":
        return score_schema_json(completion, record["target"], record["schema"])
    if benchmark == "mbpp_plus":
        solution = sanitize_mbpp(completion, str(record["entry_point"]))
        return {
            "parsed_output": solution,
            "score": None,
            "failure_type": None,
            **score_mbpp_syntax(solution),
        }
    raise ValueError(f"Unknown benchmark: {benchmark}")
