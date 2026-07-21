#!/bin/bash
# =============================================================================
# Exhaustive-subset Dream factorization pipeline smoke test
# =============================================================================
# Run this script from anywhere after placing it inside the repository's
# factorization/ directory. It validates the edited Python APIs, runs local
# tests, performs a tiny real-GPU rollout, audits every JSON record, and then
# exercises the plotting pipeline.
#
# Default integration shape:
#   K=3, subset sizes=(1,2,3), two rollout steps
#   2^3 - 1 = 7 nonempty subsets per step
#   2 * 7 = 14 expected estimates.jsonl rows
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Repository location and user-overridable smoke settings
# -----------------------------------------------------------------------------

# The script is intended to live at factorization/<this-file>.sh, so its parent
# directory is the repository root regardless of the caller's working folder.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

GPU_ID="${GPU_ID:-0}"
NICE_LEVEL="${NICE_LEVEL:-10}"
EXPECTED_CONDA_ENV="${EXPECTED_CONDA_ENV:-}"

SMOKE_CANVAS_LENGTH="${SMOKE_CANVAS_LENGTH:-12}"
SMOKE_NUM_STEPS="${SMOKE_NUM_STEPS:-2}"
SMOKE_MAX_K="${SMOKE_MAX_K:-3}"
SMOKE_SUBSET_SIZES="${SMOKE_SUBSET_SIZES:-1 2 3}"
SMOKE_RUN_ID="${SMOKE_RUN_ID:-smoke-exhaustive-k3-run5}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-outputs/factorization}"
SMOKE_DPI="${SMOKE_DPI:-100}"

RUN_DIR="$SMOKE_OUTPUT_DIR/$SMOKE_RUN_ID"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/${SMOKE_RUN_ID}.log"

# TEST_FILE may name one specific Python file, for example:
#   TEST_FILE=factorization/test_probe.py
# TEST_MODE may be "pytest", "python", or "auto". Without TEST_FILE, every
# factorization/test_*.py file is run through pytest when present.
TEST_FILE="${TEST_FILE:-}"
TEST_MODE="${TEST_MODE:-auto}"
TEST_ARGS="${TEST_ARGS:-}"

mkdir -p "$LOG_DIR" "$SMOKE_OUTPUT_DIR"

# -----------------------------------------------------------------------------
# Cluster/runtime safeguards
# -----------------------------------------------------------------------------

# LOAD-BEARING on this cluster: capped virtual memory can cause framework or
# compiler thread creation to fail even when physical memory is available.
ulimit -v unlimited

# Keep tokenizer behavior predictable, force immediate log flushing, and avoid
# network fallbacks because the real rollout uses --local-files-only.
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# -----------------------------------------------------------------------------
# Fail-early environment and file checks
# -----------------------------------------------------------------------------

if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "ERROR: no conda environment is active."
    echo "Activate the environment used by the Dream factorization project."
    exit 1
fi

if [ -n "$EXPECTED_CONDA_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "$EXPECTED_CONDA_ENV" ]; then
    echo "ERROR: expected conda env '$EXPECTED_CONDA_ENV', but '$CONDA_DEFAULT_ENV' is active."
    exit 1
fi

echo "[OK] conda environment: $CONDA_DEFAULT_ENV"
echo "[OK] repository root:  $PROJECT_DIR"

REQUIRED_FILES=(
    "factorization/dream_adapter.py"
    "factorization/probe.py"
    "factorization/rollout.py"
    "scripts/run_factorization_rollout.py"
    "scripts/plot_factorization_results.py"
)

for path in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$path" ]; then
        echo "ERROR: required file not found: $path"
        exit 1
    fi
done
printf '%s\n' "[OK] all edited pipeline files are present."

# -----------------------------------------------------------------------------
# Static syntax, CLI, and stale-interface checks
# -----------------------------------------------------------------------------

python -m py_compile "${REQUIRED_FILES[@]}"
python scripts/run_factorization_rollout.py --help >/dev/null
python scripts/plot_factorization_results.py --help >/dev/null
printf '%s\n' "[OK] Python syntax and both command-line interfaces load."

# Reject the obsolete prefix-only k_values interface without false-matching
# ordinary names such as chunk_values.
python - <<'PY'
from pathlib import Path
import re

paths = [
    Path("factorization/probe.py"),
    Path("factorization/rollout.py"),
    Path("scripts/run_factorization_rollout.py"),
]
pattern = re.compile(r"(?<![A-Za-z0-9_])k_values(?![A-Za-z0-9_])|--k-values")
violations: list[str] = []
for path in paths:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if pattern.search(line):
            violations.append(f"{path}:{line_number}: {line.strip()}")

if violations:
    raise SystemExit(
        "Obsolete k_values interface remains:\n" + "\n".join(violations)
    )
print("[OK] no obsolete k_values/--k-values interface remains.")
PY

# -----------------------------------------------------------------------------
# Python API contract and exhaustive-combination checks
# -----------------------------------------------------------------------------

python - <<'PY'
from dataclasses import MISSING, fields
from inspect import Parameter, signature

from factorization.probe import _enumerate_subsets, probe_state
from factorization.rollout import RolloutConfig

# probe_state must require both experiment-defining keyword arguments.
sig = signature(probe_state)
for name in ("max_k", "subset_sizes"):
    parameter = sig.parameters[name]
    assert parameter.kind is Parameter.KEYWORD_ONLY, (name, parameter.kind)
    assert parameter.default is Parameter.empty, f"{name} still has a default"

# RolloutConfig must also require both values rather than silently inventing a
# scientific configuration.
field_map = {field.name: field for field in fields(RolloutConfig)}
for name in ("max_k", "subset_sizes"):
    field = field_map[name]
    assert field.default is MISSING and field.default_factory is MISSING, name

config = RolloutConfig(
    max_k=3,
    subset_sizes=(1, 2, 3),
    num_steps=2,
    conditional_microbatch_size=1,
    rollout_id="api-smoke",
)
assert config.estimates_per_step == 7

expected = (
    (0,), (1,), (2,),
    (0, 1), (0, 2), (1, 2),
    (0, 1, 2),
)
actual = _enumerate_subsets(3, (1, 2, 3))
assert actual == expected, (actual, expected)
print("[OK] required arguments and exhaustive K=3 subset enumeration are correct.")
PY

# -----------------------------------------------------------------------------
# Run the user's factorization-folder test file(s), when present
# -----------------------------------------------------------------------------

run_selected_test_file() {
    local test_path="$1"
    local mode="$2"
    local -a extra_args=()

    if [ ! -f "$test_path" ]; then
        echo "ERROR: TEST_FILE does not exist: $test_path"
        exit 1
    fi

    if [ -n "$TEST_ARGS" ]; then
        # TEST_ARGS is intentionally a simple whitespace-separated convenience
        # override for local smoke usage.
        read -r -a extra_args <<< "$TEST_ARGS"
    fi

    if [ "$mode" = "auto" ]; then
        case "$(basename "$test_path")" in
            test_*.py) mode="pytest" ;;
            *)         mode="python" ;;
        esac
    fi

    case "$mode" in
        pytest)
            echo "Running pytest file: $test_path"
            python -m pytest -q "$test_path" "${extra_args[@]}"
            ;;
        python)
            echo "Running Python smoke file: $test_path"
            python "$test_path" "${extra_args[@]}"
            ;;
        *)
            echo "ERROR: TEST_MODE must be auto, pytest, or python; received '$mode'."
            exit 1
            ;;
    esac
}

if [ -n "$TEST_FILE" ]; then
    run_selected_test_file "$TEST_FILE" "$TEST_MODE"
else
    mapfile -t AUTO_TEST_FILES < <(
        find factorization -maxdepth 1 -type f -name 'test_*.py' -print | sort
    )
    if [ "${#AUTO_TEST_FILES[@]}" -gt 0 ]; then
        echo "Running factorization pytest files:"
        printf '  %s\n' "${AUTO_TEST_FILES[@]}"
        python -m pytest -q "${AUTO_TEST_FILES[@]}"
    else
        echo "[NOTE] no factorization/test_*.py files were auto-detected."
        echo "       Set TEST_FILE=factorization/<your_test>.py to run a differently named file."
    fi
fi

# -----------------------------------------------------------------------------
# Selected physical GPU check
# -----------------------------------------------------------------------------

CUDA_VISIBLE_DEVICES="$GPU_ID" python - <<'PY'
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    sys.exit("ERROR: PyTorch cannot see the requested CUDA GPU.")

properties = torch.cuda.get_device_properties(0)
print(
    f"[OK] CUDA logical device 0: {properties.name} "
    f"({properties.total_memory / 1024**3:.1f} GiB)"
)
if not torch.cuda.is_bf16_supported():
    sys.exit("ERROR: selected GPU/PyTorch build does not report BF16 support.")
print("[OK] bfloat16 is supported.")
PY

# -----------------------------------------------------------------------------
# Tiny foreground real-model rollout
# -----------------------------------------------------------------------------

read -r -a SUBSET_SIZE_ARGS <<< "$SMOKE_SUBSET_SIZES"
if [ "${#SUBSET_SIZE_ARGS[@]}" -eq 0 ]; then
    echo "ERROR: SMOKE_SUBSET_SIZES must contain at least one integer."
    exit 1
fi
EXPECTED_SMOKE_SUBSETS="$(python - "$SMOKE_MAX_K" "${SUBSET_SIZE_ARGS[@]}" <<'PY'
import math
import sys

max_k = int(sys.argv[1])
sizes = [int(value) for value in sys.argv[2:]]
print(sum(math.comb(max_k, size) for size in sizes))
PY
)"
EXPECTED_SMOKE_ROWS=$((SMOKE_NUM_STEPS * EXPECTED_SMOKE_SUBSETS))

echo ""
echo "Running real Dream exhaustive-subset smoke rollout..."
echo "  physical GPU : $GPU_ID"
echo "  logical device: cuda:0"
echo "  canvas/steps : $SMOKE_CANVAS_LENGTH/$SMOKE_NUM_STEPS"
echo "  max K        : $SMOKE_MAX_K"
echo "  subset sizes : ${SUBSET_SIZE_ARGS[*]}"
echo "  run directory: $RUN_DIR"
echo "  log          : $LOG_FILE"
echo ""

# Keep this run in the foreground: the shell exit code must reflect the Python
# result, and the JSON/plot audits should start only after completion.
CUDA_VISIBLE_DEVICES="$GPU_ID" \
nice -n "$NICE_LEVEL" \
python -u scripts/run_factorization_rollout.py \
    --mode unconditional \
    --device cuda:0 \
    --dtype bfloat16 \
    --canvas-length "$SMOKE_CANVAS_LENGTH" \
    --num-steps "$SMOKE_NUM_STEPS" \
    --max-k "$SMOKE_MAX_K" \
    --subset-sizes "${SUBSET_SIZE_ARGS[@]}" \
    --conditional-microbatch-size 1 \
    --run-id "$SMOKE_RUN_ID" \
    --output-dir "$SMOKE_OUTPUT_DIR" \
    --local-files-only \
    --overwrite \
    2>&1 | tee "$LOG_FILE"

# -----------------------------------------------------------------------------
# Machine-readable output audit
# -----------------------------------------------------------------------------

export RUN_DIR SMOKE_MAX_K SMOKE_NUM_STEPS SMOKE_SUBSET_SIZES
python - <<'PY'
from __future__ import annotations

import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Any

run_dir = Path(os.environ["RUN_DIR"])
max_k = int(os.environ["SMOKE_MAX_K"])
num_steps = int(os.environ["SMOKE_NUM_STEPS"])
subset_sizes = tuple(int(value) for value in os.environ["SMOKE_SUBSET_SIZES"].split())
expected_subsets = {
    subset
    for size in subset_sizes
    for subset in combinations(range(1, max_k + 1), size)
}
expected_per_step = len(expected_subsets)
expected_rows = num_steps * expected_per_step


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict), path
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert all(isinstance(row, dict) for row in rows), path
    return rows

metadata = read_json(run_dir / "metadata.json")
status = read_json(run_dir / "status.json")
summary = read_json(run_dir / "summary.json")
steps = read_jsonl(run_dir / "steps.jsonl")
estimates = read_jsonl(run_dir / "estimates.jsonl")

assert metadata["schema_version"] == 2
assert status["status"] == "completed", status
experiment = metadata["experiment"]
assert int(experiment["max_k"]) == max_k
assert tuple(experiment["subset_sizes"]) == subset_sizes
assert int(experiment["estimates_per_step"]) == expected_per_step
assert int(experiment["expected_total_estimate_rows"]) == expected_rows

assert len(steps) == num_steps, len(steps)
assert len(estimates) == expected_rows, len(estimates)
assert len({(int(row["step_index"]), int(row["subset_mask"])) for row in estimates}) == expected_rows

for step_index, step in enumerate(steps):
    assert int(step["step_index"]) == step_index
    assert int(step["remaining_mask_count_before"]) - int(step["remaining_mask_count_after"]) == 1
    probe = step["probe"]
    assert int(probe["max_k"]) == max_k
    assert tuple(probe["subset_sizes"]) == subset_sizes
    assert len(probe["selected"]) == max_k
    assert int(probe["estimate_count"]) == expected_per_step

    step_rows = [row for row in estimates if int(row["step_index"]) == step_index]
    observed_subsets = {tuple(int(rank) for rank in row["subset_ranks"]) for row in step_rows}
    assert observed_subsets == expected_subsets, (step_index, observed_subsets)

    for row in step_rows:
        ranks = tuple(int(rank) for rank in row["subset_ranks"])
        subset_size = len(ranks)
        expected_mask = sum(1 << (rank - 1) for rank in ranks)

        assert int(row["schema_version"]) == 2
        assert int(row["k"]) == subset_size
        assert int(row["subset_size"]) == subset_size
        assert int(row["subset_mask"]) == expected_mask
        assert row["subset_mask_binary"] == format(expected_mask, f"0{max_k}b")
        assert row["subset_label"] == "{" + ",".join(map(str, ranks)) + "}"
        assert len(row["selected_sequence_positions"]) == max_k
        assert len(row["selected_token_ids"]) == max_k

        for field in (
            "subset_sequence_positions",
            "subset_canvas_positions",
            "subset_token_ids",
            "subset_token_text_repr",
            "subset_original_probabilities",
            "subset_original_log_probabilities",
            "marginal_log_factors",
            "ordered_joint_log_factors",
        ):
            assert len(row[field]) == subset_size, (field, ranks)

        finite_fields = (
            "ordered_joint_probability",
            "marginal_product_probability",
            "difference_joint_minus_product",
            "absolute_difference",
            "ordered_joint_log_probability",
            "marginal_product_log_probability",
            "log_gap_joint_minus_product",
            "absolute_log_gap",
        )
        for field in finite_fields:
            assert math.isfinite(float(row[field])), (field, ranks)

        log_p = math.fsum(float(value) for value in row["ordered_joint_log_factors"])
        log_q = math.fsum(float(value) for value in row["marginal_log_factors"])
        assert math.isclose(log_p, float(row["ordered_joint_log_probability"]), rel_tol=1e-9, abs_tol=1e-8)
        assert math.isclose(log_q, float(row["marginal_product_log_probability"]), rel_tol=1e-9, abs_tol=1e-8)
        assert math.isclose(
            log_p - log_q,
            float(row["log_gap_joint_minus_product"]),
            rel_tol=1e-9,
            abs_tol=1e-8,
        )

        if subset_size == 1:
            assert abs(float(row["log_gap_joint_minus_product"])) <= 1e-8
            assert abs(float(row["difference_joint_minus_product"])) <= 1e-10

assert int(summary["completed_steps"]) == num_steps
assert int(summary["max_k"]) == max_k
assert tuple(summary["subset_sizes"]) == subset_sizes
assert int(summary["estimates_per_step"]) == expected_per_step
assert int(summary["total_estimate_rows"]) == expected_rows
assert set(summary["aggregate_by_subset_size"]) == {str(size) for size in subset_sizes}
assert set(summary["aggregate_by_subset"]) == {",".join(map(str, subset)) for subset in expected_subsets}

print(
    "[OK] JSON audit passed: "
    f"{num_steps} steps, {expected_per_step} subsets/step, {expected_rows} rows."
)
PY

# -----------------------------------------------------------------------------
# Plotting integration and expected-artifact audit
# -----------------------------------------------------------------------------

python scripts/plot_factorization_results.py "$RUN_DIR" --dpi "$SMOKE_DPI"

EXPECTED_PLOTS=(
    "summary_statistics.csv"
    "summary_statistics_by_subset_size.csv"
    "summary_statistics_by_exact_subset.csv"
    "denoising_trajectory.csv"
    "denoising_trajectory.txt"
    "report.tex"
    "factorization_gap_histograms.png"
    "factorization_gap_boxplots.png"
    "absolute_log_gap_trajectory.png"
    "absolute_log_gap_ecdf.png"
    "top_exact_subsets_by_median_absolute_log_gap.png"
)

if [[ " ${SUBSET_SIZE_ARGS[*]} " == *" 2 "* ]]; then
    EXPECTED_PLOTS+=("pairwise_summary_statistics.csv")
fi

for filename in "${EXPECTED_PLOTS[@]}"; do
    path="$RUN_DIR/plots/$filename"
    if [ ! -s "$path" ]; then
        echo "ERROR: expected nonempty plot artifact was not created: $path"
        exit 1
    fi
done

RETIRED_ARTIFACTS=(
    "report.md"
    "absolute_log_gap_histograms.png"
    "absolute_log_gap_boxplot.png"
    "absolute_raw_gap_histograms.png"
    "absolute_raw_gap_boxplot.png"
    "pairwise_absolute_log_gap_boxplot.png"
    "pairwise_median_absolute_log_gap_heatmap.png"
    "pairwise_median_signed_log_gap_heatmap.png"
)

for filename in "${RETIRED_ARTIFACTS[@]}"; do
    path="$RUN_DIR/plots/$filename"
    if [ -e "$path" ]; then
        echo "ERROR: retired plot artifact still exists: $path"
        exit 1
    fi
done

if compgen -G "$RUN_DIR/plots/exact_subset_absolute_log_gap_boxplot_k*.png" >/dev/null; then
    echo "ERROR: separate exact-subset boxplots remain after consolidation."
    exit 1
fi

RUN_DIR="$RUN_DIR" python - <<'PY'
import os
from pathlib import Path

import pandas as pd

plots_dir = Path(os.environ["RUN_DIR"]) / "plots"
for filename in (
    "summary_statistics.csv",
    "summary_statistics_by_subset_size.csv",
    "summary_statistics_by_exact_subset.csv",
):
    frame = pd.read_csv(plots_dir / filename)
    assert set(frame["subset_size"]) == {
        int(value) for value in os.environ["SMOKE_SUBSET_SIZES"].split()
    }, filename

report = (plots_dir / "report.tex").read_text(encoding="utf-8")
assert "\\begin{document}" in report
assert "\\end{document}" in report
assert "Visual key" in report
assert "singletons" in report.lower()
if 2 in {int(value) for value in os.environ["SMOKE_SUBSET_SIZES"].split()}:
    pairwise = pd.read_csv(plots_dir / "pairwise_summary_statistics.csv")
    assert (pairwise["subset_size"] == 2).all()
    assert "Pairwise confidence-rank summary" in report
print("[OK] plotting outputs include singletons, consolidate boxplots, and exclude heatmaps.")
PY

# -----------------------------------------------------------------------------
# Final result
# -----------------------------------------------------------------------------

echo ""
echo "Smoke test passed"
echo "-----------------"
echo "Run directory: $RUN_DIR"
echo "Log:           $LOG_FILE"
echo "Plots/report:  $RUN_DIR/plots"
echo "Expected rows: $EXPECTED_SMOKE_ROWS ($EXPECTED_SMOKE_SUBSETS exact subsets per step)"
echo ""
echo "The production run can now use: --max-k 6 --subset-sizes 2 3 4 5 6"
