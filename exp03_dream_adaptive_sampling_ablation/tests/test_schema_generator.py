from __future__ import annotations

import random

from jsonschema import Draft202012Validator

from scripts.prepare_schema_json import make_record


def test_generated_targets_validate_and_are_deterministic() -> None:
    left = make_record(random.Random(7), "test", 0)
    right = make_record(random.Random(7), "test", 0)
    assert left == right
    Draft202012Validator(left["schema"]).validate(left["target"])

