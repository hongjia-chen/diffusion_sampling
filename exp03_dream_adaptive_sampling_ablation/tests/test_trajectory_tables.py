from __future__ import annotations

import importlib.util
from pathlib import Path


PATH = Path(__file__).resolve().parent.parent / "report" / "make_trajectory_tables.py"
SPEC = importlib.util.spec_from_file_location("make_trajectory_tables", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_aggregate_uses_conditional_mean_and_reached_count() -> None:
    rows = MODULE.aggregate(
        {
            "a": [
                {"commitments_by_step": [4, 2, 1]},
                {"commitments_by_step": [2, 2]},
            ],
            "b": [
                {"commitments_by_step": [3]},
                {"commitments_by_step": [1, 1]},
            ],
        }
    )
    assert rows[0]["a_mean_tokens"] == 3
    assert rows[0]["a_reached"] == 2
    assert rows[1]["b_mean_tokens"] == 1
    assert rows[1]["b_reached"] == 1
    assert rows[2]["a_mean_tokens"] == 1
    assert rows[2]["a_reached"] == 1
    assert rows[2]["b_reached"] == 0
