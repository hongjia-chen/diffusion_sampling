"""Run-metadata file names stay under the file system's 255-byte limit."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_benchmark import run_metadata_tag  # noqa: E402


def test_short_method_sets_keep_their_joined_names() -> None:
    labels = ["fixed_k_1", "dynamic_theta_0p95", "dynamic_theta_0p90"]
    assert run_metadata_tag(labels) == "fixed_k_1+dynamic_theta_0p95+dynamic_theta_0p90"


def test_a_wide_delta_grid_gets_a_short_distinct_name() -> None:
    grid = ["0p00", "0p001", "0p01", "0p03", "0p10", "0p30", "1p00", "3p00", "10p00"]
    labels = [f"xadaptive_noveto_delta_{d}" for d in grid]
    tag = run_metadata_tag(labels)
    name = f".run_metadata_worker_0+1__{tag}.json.abcdefgh.tmp"
    assert len(name.encode()) < 255
    assert tag.startswith("xadaptive_noveto_delta_0p00+8more_")
    assert tag != run_metadata_tag(labels[:-1] + ["xadaptive_noveto_delta_30p00"])
