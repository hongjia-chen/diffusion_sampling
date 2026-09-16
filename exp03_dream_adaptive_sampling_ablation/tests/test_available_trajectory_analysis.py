from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path


PATH = Path(__file__).resolve().parent.parent / "report" / "make_available_trajectory_analysis.py"
SPEC = importlib.util.spec_from_file_location("make_available_trajectory_analysis", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_completed_record_diagnostics_are_complete_and_consistent() -> None:
    survival, widths = MODULE.derive()
    survival_groups = defaultdict(list)
    for row in survival:
        survival_groups[(row["benchmark"], row["method"])].append(row)
    assert len(survival_groups) == 24
    for rows in survival_groups.values():
        counts = [row["tasks_reached"] for row in rows]
        assert counts[0] == 100
        assert all(left >= right for left, right in zip(counts, counts[1:]))
        assert counts[-1] >= 1

    width_groups = defaultdict(list)
    for row in widths:
        width_groups[(row["benchmark"], row["method"])].append(row)
    assert len(width_groups) == 24
    for rows in width_groups.values():
        assert len(rows) == 6
        assert abs(sum(row["step_share"] for row in rows) - 1.0) < 1e-12


def test_selected_survival_counts_match_completed_records() -> None:
    survival, _ = MODULE.derive()
    lookup = {
        (row["benchmark"], row["method"], row["step"]): row["tasks_reached"]
        for row in survival
    }
    assert lookup[("gsm8k", "adaptive_delta_0p10", 100)] == 43
    assert lookup[("mbpp_plus", "adaptive_delta_0p10", 100)] == 81
    assert lookup[("mbpp_plus", "adaptive_delta_1p50", 50)] == 4
