"""The summary's stage breakdown: shares, calls and the profiling slowdown."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_results import stage_breakdown  # noqa: E402


def row(wall, profiled, seconds, counts=None, steps=10):
    out = {"wall_time_seconds": wall, "profiled_wall_seconds": profiled,
           "stage_seconds": seconds, "denoising_steps": steps}
    if counts is not None:
        out["stage_counts"] = counts
    return out


def test_an_unprofiled_method_has_no_breakdown() -> None:
    assert stage_breakdown([{"wall_time_seconds": 1.0, "stage_seconds": {}}]) is None


def test_calls_and_milliseconds_per_call_come_from_the_counts() -> None:
    rows = [
        row(1.0, 1.2, {"propose_forward": 0.6, "verify_forward": 0.4},
            {"propose_forward": 10, "verify_forward": 20}),
        row(1.0, 1.2, {"propose_forward": 0.6, "verify_forward": 0.4},
            {"propose_forward": 10, "verify_forward": 20}),
    ]
    out = stage_breakdown(rows)
    assert math.isclose(out["seconds"]["residual"], 0.2)
    assert math.isclose(out["profiling_slowdown"], 1.2)
    assert out["calls_per_task"] == {"propose_forward": 10, "verify_forward": 20}
    assert out["calls_per_step"] == {"propose_forward": 1.0, "verify_forward": 2.0}
    assert math.isclose(out["ms_per_call"]["propose_forward"], 60.0)
    assert math.isclose(out["ms_per_call"]["verify_forward"], 20.0)


def test_records_from_before_the_counts_still_summarize() -> None:
    out = stage_breakdown([row(1.0, 1.0, {"proposer": 0.5, "commit": 0.1})])
    assert "ms_per_call" not in out
    assert math.isclose(out["share"]["proposer"], 0.5)
