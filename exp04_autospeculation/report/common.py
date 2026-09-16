"""Shared loading for the report builders.

Everything is read through outputs/final/, so the report can only ever describe
the four deliverables. Nothing here reaches into outputs/baseline or
outputs/unbounded directly -- if a symlink in final/ is repointed, the report
follows it.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
FINAL = PROJECT / "outputs" / "final"
REPORT = PROJECT / "report"

BENCHMARKS = ["gsm8k", "math_500", "humaneval_plus", "mbpp_plus"]
BENCH_LABEL = {
    "gsm8k": "GSM8K",
    "math_500": "MATH-500",
    "humaneval_plus": "HumanEval+",
    "mbpp_plus": "MBPP+",
}
METRIC_LABEL = {
    "gsm8k": "exact match",
    "math_500": "accuracy",
    "humaneval_plus": "pass@1",
    "mbpp_plus": "pass@1",
}

GROUPS = {
    "1_dream7b_baseline": "Dream-7B fixed:k",
    "2_sdar_b64_baseline": "SDAR-1.7B-b64 fixed:k",
    "3_adaptive_delta": "SDAR adaptive delta",
    "4_autospec": "SDAR autospec (Alg. 3)",
}

# Validated all-pairs in light mode (dataviz validator): worst CVD dE 9.2,
# worst normal-vision dE 16.3. Aqua warns on contrast vs surface, so every
# series is also direct-labelled and every value appears in a table.
COLOR = {
    "1_dream7b_baseline": "#2a78d6",
    "2_sdar_b64_baseline": "#eb6834",
    "3_adaptive_delta": "#1baf7a",
    "4_autospec": "#4a3aa7",
}
MARKER = {
    "1_dream7b_baseline": "o",
    "2_sdar_b64_baseline": "s",
    "3_adaptive_delta": "^",
    "4_autospec": "D",
}
LINESTYLE = {
    "1_dream7b_baseline": "-",
    "2_sdar_b64_baseline": "--",
    "3_adaptive_delta": "-.",
    "4_autospec": "none",
}

FIXED = ["fixed_k_1", "fixed_k_2", "fixed_k_4", "fixed_k_8"]
DELTAS = [
    "adaptive_delta_0p10",
    "adaptive_delta_0p20",
    "adaptive_delta_0p50",
    "adaptive_delta_1p00",
    "adaptive_delta_1p50",
]
DELTA_VALUE = {m: float(m.split("_")[-1].replace("p", ".")) for m in DELTAS}
AUTOSPEC = "autospec_rho_auto"

METHODS = {
    "1_dream7b_baseline": FIXED,
    "2_sdar_b64_baseline": FIXED,
    "3_adaptive_delta": DELTAS,
    "4_autospec": [AUTOSPEC],
}


def run_dir(group: str, benchmark: str, method: str) -> Path:
    return FINAL / group / benchmark / method


def records(group: str, benchmark: str, method: str) -> list[dict]:
    """Per-task records, preferring the rescored file where one exists."""
    d = run_dir(group, benchmark, method)
    path = d / "records.scored.jsonl"
    if not path.exists():
        path = d / "records.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def metadata(group: str, benchmark: str, method: str) -> dict:
    return json.loads((run_dir(group, benchmark, method) / "metadata.json").read_text())


def summary_rows() -> list[dict]:
    """The combined table built during the outputs cleanup."""
    return json.loads((FINAL / "summary.json").read_text())


def mean(rows: list[dict], key: str) -> float:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return st.fmean(values) if values else float("nan")


def fmt(value: float, places: int = 2) -> str:
    if value != value:  # NaN
        return "--"
    return f"{value:,.{places}f}"
