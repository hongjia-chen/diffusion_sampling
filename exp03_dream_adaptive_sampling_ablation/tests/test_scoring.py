from __future__ import annotations

from dream_ablation.scoring import (
    extract_json_object,
    normalize_number,
    score_gsm8k,
    score_schema_json,
)


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


def test_schema_json_semantic_exact_match() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["x"],
        "properties": {"x": {"type": "integer"}},
    }
    assert score_schema_json('{"x": 2}', {"x": 2}, schema)["score"] is True
    wrong = score_schema_json('{"x": 3}', {"x": 2}, schema)
    assert wrong["score"] is False
    assert wrong["field_accuracy"] == 0.0


def test_number_normalization_rejects_non_numbers() -> None:
    assert normalize_number("$1,200.0") == "1200"
    assert normalize_number("twelve") is None


SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "integer"}},
    "required": ["x"],
    "additionalProperties": False,
}


def test_unfenced_json_needs_no_repair() -> None:
    result = score_schema_json('{"x": 2}', {"x": 2}, SCHEMA)
    assert result["score"] is True
    assert result["json_repair"] is None


def test_fenced_json_scores_the_same_as_unfenced() -> None:
    """The outputs/v2 bug: the fixed arm fenced every completion and scored 0.00.

    A fence is a formatting choice made by the sampler, not a property of the
    JSON, so the two must score identically or the metric measures fence rate.
    """
    for fenced in (
        '```json\n{"x": 2}\n```',
        '```\n{"x": 2}\n```',
        '```  {"x": 2}\n```',
        '```JSON\n  {"x": 2}  \n```',
    ):
        result = score_schema_json(fenced, {"x": 2}, SCHEMA)
        assert result["score"] is True, fenced
        assert result["json_parse_valid"] is True
        assert result["json_repair"] == "fence"


def test_prose_around_json_is_recovered_and_labelled() -> None:
    result = score_schema_json('Here is the object:\n{"x": 2}\nHope that helps.',
                               {"x": 2}, SCHEMA)
    assert result["score"] is True
    assert result["json_repair"] == "brace_span"


def test_repair_does_not_rescue_a_wrong_answer() -> None:
    """Extraction must not change the verdict, only reach the content."""
    result = score_schema_json('```json\n{"x": 3}\n```', {"x": 2}, SCHEMA)
    assert result["json_parse_valid"] is True
    assert result["score"] is False
    assert result["failure_type"] == "semantic_mismatch"


def test_genuinely_broken_json_still_fails() -> None:
    """A dropped opening brace is a generation failure, not a formatting one.

    fixed_k_4 in outputs/v2 produced exactly this, and it must keep scoring 0.
    """
    result = score_schema_json('```\n  "x": 2\n}\n```', {"x": 2}, SCHEMA)
    assert result["json_parse_valid"] is False
    assert result["score"] is False
    assert result["failure_type"] == "json_parse_error"
    assert result["json_repair"] is None


def test_extractor_prefers_the_least_aggressive_repair() -> None:
    assert extract_json_object('{"x": 1}') == ({"x": 1}, True, None)
    assert extract_json_object('```json\n{"x": 1}\n```')[2] == "fence"
    assert extract_json_object("not json at all") == (None, False, None)


def test_schema_violation_is_distinct_from_a_parse_error() -> None:
    result = score_schema_json('```json\n{"x": "two"}\n```', {"x": 2}, SCHEMA)
    assert result["json_parse_valid"] is True
    assert result["failure_type"] == "schema_violation"
