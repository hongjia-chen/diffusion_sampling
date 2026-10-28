"""The summary's loop-free columns: per cell, paired within a root, and x fixed:1."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import summarize_results  # noqa: E402
from summarize_results import (  # noqa: E402
    LOOP_FREE_FIELDS, loop_free_summary, paired_loop_free, summarize_file,
)


def record(example_id, tokens, wall, *, capped=False, prefill=0.5, score=False,
           benchmark="gsm8k", method="fixed_k_1", **extra):
    """The fields summarize_file reads, nothing else."""
    return {
        "benchmark": benchmark, "method": method, "example_id": example_id,
        "score": score, "output_token_count": tokens, "wall_time_seconds": wall,
        "prefill_seconds": prefill, "maximum_length_termination": capped,
        "commitment_histogram": [3, 1], "decoded_positions": tokens,
        "model_forward_passes": 4, "early_stopped": not capped, "denoising_steps": 4,
        "dream_proposal_passes": 4, "verifier_rows": 0, "verifier_forward_passes": 0,
        "peak_gpu_memory_bytes": 1, **extra,
    }


def write_cell(root: Path, rows: list[dict]) -> Path:
    path = root / rows[0]["benchmark"] / rows[0]["method"] / "records.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_a_cell_drops_only_its_own_capped_tasks() -> None:
    rows = [record("a", 100, 2.0), record("b", 300, 4.0), record("c", 4096, 20.0, capped=True)]
    out = loop_free_summary(rows)
    assert out["n_capped"] == 1 and out["loop_free_n"] == 2
    assert math.isclose(out["loop_free_tokens_per_second"], 400 / 6.0)
    assert math.isclose(out["loop_free_seconds_per_task"], 3.0)
    assert math.isclose(out["loop_free_decode_tokens_per_second"], 400 / (6.0 - 1.0))


def test_without_capped_tasks_the_columns_equal_the_all_task_ones(tmp_path: Path) -> None:
    # Exactly equal, not close: the same formula over the same rows.
    rows = [record("a", 123, 1.7, prefill=0.11), record("b", 457, 3.3, prefill=0.13),
            record("c", 89, 0.9, prefill=0.07)]
    summary = summarize_file(write_cell(tmp_path, rows))
    out = loop_free_summary(rows)
    assert out["n_capped"] == 0 and out["loop_free_n"] == 3
    assert out["loop_free_tokens_per_second"] == summary["tokens_per_second"]
    assert out["loop_free_seconds_per_task"] == summary["seconds_per_task"]
    assert out["loop_free_decode_tokens_per_second"] == summary["decode_tokens_per_second"]


def test_no_decode_rate_when_a_task_lacks_its_prefill() -> None:
    # Even when the task without a prefill is the capped one: the all-task
    # decode column does not exist for this cell, so neither does this one.
    rows = [record("a", 100, 2.0), record("b", 4096, 20.0, capped=True, prefill=None)]
    out = loop_free_summary(rows)
    assert out["loop_free_decode_tokens_per_second"] is None
    assert math.isclose(out["loop_free_tokens_per_second"], 50.0)


def test_an_all_capped_cell_has_no_rate() -> None:
    out = loop_free_summary([record("a", 4096, 20.0, capped=True)])
    assert out["loop_free_n"] == 0
    assert math.isnan(out["loop_free_tokens_per_second"])
    assert math.isnan(out["loop_free_seconds_per_task"])


def test_paired_pairs_each_method_with_fixed1_only() -> None:
    fixed1 = [record("a", 100, 10.0), record("b", 100, 10.0, capped=True),
              record("c", 200, 10.0), record("d", 100, 5.0)]
    dynamic = [record("a", 100, 2.0, method="dynamic_theta_0p95"),
               record("b", 100, 2.0, method="dynamic_theta_0p95"),
               record("c", 4096, 1.0, capped=True, method="dynamic_theta_0p95"),
               record("d", 100, 1.0, method="dynamic_theta_0p95")]
    fixed8 = [record("a", 4096, 1.0, capped=True, method="fixed_k_8"),
              record("b", 100, 1.0, method="fixed_k_8"),
              record("c", 200, 1.0, method="fixed_k_8"), record("d", 100, 0.5, method="fixed_k_8")]
    out = paired_loop_free([("gsm8k", "fixed_k_1", fixed1), ("gsm8k", "dynamic_theta_0p95", dynamic),
                            ("gsm8k", "fixed_k_8", fixed8)])
    # fixed:1 keeps its own uncapped tasks (a, c, d) and is its own reference.
    assert out[0]["paired_loop_free_n"] == 3
    assert math.isclose(out[0]["paired_loop_free_tokens_per_second"], 400 / 25.0)
    assert out[0]["paired_loop_free_speedup_vs_fixed1"] == 1.0
    # dynamic: b capped in fixed:1, c in dynamic. fixed:8's runaway on a does
    # not reach it, as it would if the set were the union over every method.
    assert out[1]["paired_loop_free_n"] == 2
    assert math.isclose(out[1]["paired_loop_free_tokens_per_second"], 200 / 3.0)
    assert math.isclose(out[1]["paired_loop_free_speedup_vs_fixed1"], (200 / 3.0) / (200 / 15.0))
    # fixed:8: b capped in fixed:1, a in fixed:8, so c and d, fixed:1 on them too.
    assert out[2]["paired_loop_free_n"] == 2
    assert math.isclose(out[2]["paired_loop_free_tokens_per_second"], 300 / 1.5)
    assert math.isclose(out[2]["paired_loop_free_speedup_vs_fixed1"], (300 / 1.5) / (300 / 15.0))


def test_paired_keeps_only_the_tasks_both_ran() -> None:
    full = [record(i, 100, 1.0) for i in "abcd"]
    subset = [record(i, 100, 0.5, method="dynamic_theta_0p90") for i in "ab"]
    out = paired_loop_free([("gsm8k", "fixed_k_1", full), ("gsm8k", "dynamic_theta_0p90", subset)])
    assert [cell["paired_loop_free_n"] for cell in out] == [4, 2]
    assert math.isclose(out[1]["paired_loop_free_speedup_vs_fixed1"], 2.0)


def test_paired_is_per_benchmark_and_blank_without_a_fixed1() -> None:
    cells = [
        ("gsm8k", "fixed_k_1", [record("a", 100, 1.0), record("b", 100, 1.0)]),
        ("gsm8k", "dynamic_theta_0p95", [record("a", 100, 0.5, method="dynamic_theta_0p95"),
                                         record("b", 100, 0.5, method="dynamic_theta_0p95")]),
        ("math_500", "xadaptive_noveto_delta_0p10",
         [record("a", 100, 1.0, benchmark="math_500"),
          record("b", 4096, 1.0, capped=True, benchmark="math_500")]),
    ]
    out = paired_loop_free(cells)
    # gsm8k is untouched by math_500's runaway on task b.
    assert out[1]["paired_loop_free_n"] == 2
    assert math.isclose(out[1]["paired_loop_free_speedup_vs_fixed1"], 2.0)
    # math_500 has no fixed:1: all three columns blank.
    assert out[2] == {"paired_loop_free_n": None, "paired_loop_free_tokens_per_second": None,
                      "paired_loop_free_speedup_vs_fixed1": None}


def run_main(monkeypatch, root: Path, out: Path, *extra: str) -> tuple[list[str], list[dict]]:
    monkeypatch.setattr(sys, "argv", ["summarize_results.py", "--output-root", str(root),
                                      "--output-dir", str(out), *extra])
    summarize_results.main()
    with (out / "summary.csv").open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    with (out / "summary.csv").open(newline="", encoding="utf-8") as handle:
        return header, list(csv.DictReader(handle))


def test_main_appends_the_columns_after_every_older_one(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "root"
    write_cell(root, [record("g1", 100, 2.0), record("g2", 4096, 40.0, capped=True)])
    write_cell(root, [record("g1", 100, 1.0, method="dynamic_theta_0p95"),
                      record("g2", 100, 1.0, method="dynamic_theta_0p95")])
    # A code benchmark brings older columns the gsm8k rows lack (mean_syntax_valid);
    # they must still come before the loop-free ones.
    write_cell(root, [record("h1", 50, 1.0, benchmark="humaneval", method="dynamic_theta_0p95",
                             syntax_valid=True)])
    header, rows = run_main(monkeypatch, root, tmp_path / "out")
    assert tuple(header[-len(LOOP_FREE_FIELDS):]) == LOOP_FREE_FIELDS
    assert "mean_syntax_valid" in header[:-len(LOOP_FREE_FIELDS)]
    by_cell = {(row["benchmark"], row["method"]): row for row in rows}
    fixed1 = by_cell[("gsm8k", "fixed_k_1")]
    assert fixed1["n_capped"] == "1" and fixed1["loop_free_n"] == "1"
    assert float(fixed1["loop_free_tokens_per_second"]) == 50.0
    assert fixed1["paired_loop_free_n"] == "1"
    assert float(fixed1["paired_loop_free_speedup_vs_fixed1"]) == 1.0
    dyn = by_cell[("gsm8k", "dynamic_theta_0p95")]
    assert dyn["paired_loop_free_n"] == "1"
    assert float(dyn["paired_loop_free_speedup_vs_fixed1"]) == pytest.approx(2.0)
    # No fixed:1 for humaneval in this root: all three paired columns blank,
    # not zero or nan, and null in the JSON.
    humaneval = by_cell[("humaneval", "dynamic_theta_0p95")]
    assert [humaneval[f] for f in LOOP_FREE_FIELDS[-3:]] == ["", "", ""]
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert all(set(LOOP_FREE_FIELDS) <= set(row) for row in summary)
    assert all(row[f] is None for row in summary if row["benchmark"] == "humaneval"
               for f in LOOP_FREE_FIELDS[-3:])


def test_recorded_output_root_rewrites_only_the_records_path(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "root"
    write_cell(root, [record("g1", 100, 2.0)])
    _, plain = run_main(monkeypatch, root, tmp_path / "plain")
    _, moved = run_main(monkeypatch, root, tmp_path / "moved",
                        "--recorded-output-root", "/home/ec2-user/outputs/run")
    assert moved[0]["records_path"] == "/home/ec2-user/outputs/run/gsm8k/fixed_k_1/records.jsonl"
    assert {k: v for k, v in moved[0].items() if k != "records_path"} == \
        {k: v for k, v in plain[0].items() if k != "records_path"}
