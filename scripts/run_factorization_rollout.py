#!/usr/bin/env python3
"""Run and persist one exhaustive-subset Dream factorization rollout.

The scientific model behavior remains in ``factorization.dream_adapter``,
``factorization.probe``, and ``factorization.rollout``. This executable owns
command-line configuration, reproducibility metadata, progress reporting,
crash-tolerant JSONL output, decoded canvas snapshots, and summary statistics.

For every canonical denoising state, the probe freezes the top ``max_k``
confidence-ranked events and emits every subset requested by
``subset_sizes``. The rollout then permanently commits only ``x_1``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

# Make repository-local imports work even when this file is launched directly
# as ``python scripts/run_factorization_rollout.py``.
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from factorization.dream_adapter import (  # noqa: E402
    DreamAdapter,
    DreamState,
    ModelLoadOptions,
    TokenPrediction,
)
from factorization.probe import FactorizationEstimate, ProbeResult  # noqa: E402
from factorization.rollout import (  # noqa: E402
    RolloutConfig,
    RolloutResult,
    RolloutStep,
    canvas_token_ids,
    decode_canvas,
    run_rollout,
)

# Schema 2 replaces one-prefix-per-k records with subset-identified records.
SCHEMA_VERSION = 2


# -----------------------------------------------------------------------------
# Small environment and argument-parsing helpers
# -----------------------------------------------------------------------------

# Return an explicit UTC timestamp for metadata, status, and failure records.
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Read an installed package version without failing when the package is absent.
def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


# Convert user-friendly dtype spellings into the torch dtype used by the model.
def parse_dtype(value: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[value.lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(mapping))
        raise argparse.ArgumentTypeError(
            f"Unsupported dtype {value!r}; choose one of: {choices}."
        ) from exc


# Turn a torch dtype back into a compact name for JSON metadata and log output.
def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


# Accept either a positive integer or ``all`` for conditional-state batching.
def parse_microbatch_size(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"all", "none"}:
        return None
    try:
        result = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "conditional microbatch size must be a positive integer or 'all'."
        ) from exc
    if result < 1:
        raise argparse.ArgumentTypeError(
            "conditional microbatch size must be at least 1."
        )
    return result


# Define the executable interface and reject inconsistent experiments before
# creating an output directory or loading a seven-billion-parameter model.
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a greedy Dream trajectory and record exhaustive confidence-"
            "ordered subset joint-versus-product measurements."
        )
    )

    # Conditioning mode and optional visible input text.
    parser.add_argument(
        "--mode",
        choices=("unconditional", "prefix", "chat"),
        default="unconditional",
    )
    text_group = parser.add_mutually_exclusive_group()
    text_group.add_argument(
        "--text",
        help="Visible prose prefix or chat prompt, depending on --mode.",
    )
    text_group.add_argument(
        "--text-file",
        type=Path,
        help="UTF-8 file containing the visible prefix or chat prompt.",
    )

    # Checkpoint, device, and numerical precision.
    parser.add_argument(
        "--model",
        help=(
            "Hugging Face checkpoint. Defaults to Dream-Base for "
            "unconditional/prefix and Dream-Instruct for chat."
        ),
    )
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", type=parse_dtype, default=torch.bfloat16)

    # Canonical trajectory length and exhaustive-subset experiment definition.
    # Both max_k and subset_sizes are intentionally required: neither the
    # runner nor probe should silently invent the scientific configuration.
    parser.add_argument("--canvas-length", type=int, default=72)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument(
        "--max-k",
        type=int,
        required=True,
        help="Size K of the frozen confidence-ranked candidate pool.",
    )
    parser.add_argument(
        "--subset-sizes",
        type=int,
        nargs="+",
        required=True,
        metavar="S",
        help=(
            "Subset cardinalities to enumerate from the top-K pool. "
            "For all informative K=6 subsets, pass 2 3 4 5 6."
        ),
    )
    parser.add_argument(
        "--conditional-microbatch-size",
        type=parse_microbatch_size,
        default=1,
        metavar="N|all",
        help=(
            "Temporary conditional states per model call. Use 1 for the "
            "lowest peak memory or 'all' to batch every conditional state."
        ),
    )
    parser.add_argument(
        "--allow-special-tokens",
        action="store_true",
        help="Do not remove BOS/EOS/PAD/MASK from the scored vocabulary.",
    )

    # Reproducibility, output ownership, and terminal progress behavior.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-id", default="greedy-000")
    parser.add_argument(
        "--run-id",
        help="Output subdirectory name. A unique name is generated by default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/factorization"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print one compact progress summary every N steps; use 0 for quiet.",
    )
    parser.add_argument(
        "--print-all-subsets",
        action="store_true",
        help="Also print one detailed line for every measured subset.",
    )
    parser.add_argument(
        "--fsync-every-step",
        action="store_true",
        help="Force JSONL buffers to disk after every step (safer but slower).",
    )

    args = parser.parse_args(argv)

    # Basic scalar validation.
    if args.canvas_length < 1:
        parser.error("--canvas-length must be at least 1.")
    if args.num_steps < 1:
        parser.error("--num-steps must be at least 1.")
    if args.max_k < 1:
        parser.error("--max-k must be at least 1.")
    if args.max_k > args.canvas_length:
        parser.error("--max-k cannot exceed --canvas-length.")
    if args.print_every < 0:
        parser.error("--print-every must be non-negative.")

    # Preserve a single, predictable ordering in metadata and output rows.
    if not args.subset_sizes or any(size < 1 for size in args.subset_sizes):
        parser.error("--subset-sizes must contain positive integers.")
    if len(set(args.subset_sizes)) != len(args.subset_sizes):
        parser.error("--subset-sizes must not contain duplicates.")
    if tuple(args.subset_sizes) != tuple(sorted(args.subset_sizes)):
        parser.error("--subset-sizes must be supplied in increasing order.")
    if args.subset_sizes[-1] > args.max_k:
        parser.error("The largest --subset-sizes value cannot exceed --max-k.")

    # A probe runs before each commitment, so the final measured state must
    # still contain max_k masks.
    maximum_steps = args.canvas_length - args.max_k + 1
    if args.num_steps > maximum_steps:
        parser.error(
            f"--num-steps={args.num_steps} is too large for "
            f"--canvas-length={args.canvas_length} and --max-k={args.max_k}; "
            f"the maximum is {maximum_steps}."
        )

    # Conditioning text is forbidden for unconditional runs and required for
    # both visible-prefix modes.
    if args.mode == "unconditional":
        if args.text is not None or args.text_file is not None:
            parser.error("--text/--text-file are not used in unconditional mode.")
    elif args.text is None and args.text_file is None:
        parser.error(f"--mode {args.mode} requires --text or --text-file.")

    # Prevent a run ID from escaping the configured output root.
    if args.run_id is not None:
        if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
            parser.error("--run-id must be one safe path component.")

    return args


# Read text from either the direct CLI argument or the selected UTF-8 file.
def resolve_text(args: argparse.Namespace) -> str | None:
    if args.text_file is not None:
        text = args.text_file.read_text(encoding="utf-8")
    else:
        text = args.text
    if args.mode in {"prefix", "chat"} and not text:
        raise ValueError(f"Conditioning text for mode={args.mode!r} is empty.")
    return text


# Use the supplied run ID or create one that records the main experiment shape.
def make_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return args.run_id
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return (
        f"{args.mode}-greedy-N{args.canvas_length}-T{args.num_steps}-"
        f"K{args.max_k}-{timestamp}-{suffix}"
    )


# Create a fresh run directory, with an explicit opt-in before deleting an old
# run that has the same ID.
def prepare_run_directory(
    output_root: Path,
    run_id: str,
    *,
    overwrite: bool,
) -> Path:
    output_root = output_root.expanduser().resolve()
    run_dir = output_root / run_id
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {run_dir}. "
                "Choose another --run-id or pass --overwrite."
            )
        if run_dir == output_root:
            raise RuntimeError("Refusing to remove the output root itself.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


# Write small JSON files through a temporary file and atomic rename so a crash
# cannot leave metadata, status, or summary half-written.
def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


# Capture the repository commit and whether tracked/untracked files are dirty.
def git_metadata(repo_root: Path) -> dict[str, Any]:
    def run_git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    commit = run_git("rev-parse", "HEAD")
    status = run_git("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
    }


# -----------------------------------------------------------------------------
# Conversion from in-memory dataclasses to stable JSON records
# -----------------------------------------------------------------------------

# Serialize one frozen position/token proposal, including the raw-vocabulary
# winner used to audit special-token filtering.
def prediction_to_dict(prediction: TokenPrediction) -> dict[str, Any]:
    return {
        "sequence_position": prediction.sequence_position,
        "canvas_position": prediction.canvas_position,
        "token_id": prediction.token_id,
        "token_text_repr": prediction.token_text,
        "probability": prediction.probability,
        "log_probability": prediction.log_probability,
        "raw_token_id": prediction.raw_token_id,
        "raw_token_text_repr": prediction.raw_token_text,
        "raw_probability": prediction.raw_probability,
        "excluded_probability_mass": prediction.excluded_probability_mass,
        "changed_by_filter": prediction.changed_by_filter,
    }


# Exponentiate a log ratio only when the result fits in an ordinary Python
# float; keep overflow explicit as null rather than writing invalid JSON.
def safe_probability_ratio(log_gap: float) -> float | None:
    if log_gap > math.log(sys.float_info.max):
        return None
    if log_gap < math.log(sys.float_info.min):
        return 0.0
    return math.exp(log_gap)


# Serialize one exact subset measurement. Subset identity is explicit, and the
# individual marginal/conditional factors are retained for scientific audits.
def estimate_to_dict(estimate: FactorizationEstimate) -> dict[str, Any]:
    signed_difference = estimate.difference_joint_minus_product
    signed_log_gap = estimate.log_gap_joint_minus_product
    subset_label = "{" + ",".join(str(rank) for rank in estimate.subset_ranks) + "}"
    return {
        # ``k`` is retained as a compatibility alias for analysis code that
        # previously grouped prefix measurements by k. New code should prefer
        # the unambiguous ``subset_size`` field.
        "k": estimate.subset_size,
        "subset_label": subset_label,
        "subset_ranks": list(estimate.subset_ranks),
        "subset_mask": estimate.subset_mask,
        "subset_size": estimate.subset_size,
        "marginal_log_factors": list(estimate.marginal_log_factors),
        "ordered_joint_log_factors": list(
            estimate.ordered_joint_log_factors
        ),
        "ordered_joint_probability": estimate.ordered_joint_probability,
        "marginal_product_probability": estimate.marginal_product_probability,
        "difference_joint_minus_product": signed_difference,
        "absolute_difference": getattr(
            estimate,
            "absolute_difference",
            abs(signed_difference),
        ),
        "ordered_joint_log_probability": (
            estimate.ordered_joint_log_probability
        ),
        "marginal_product_log_probability": (
            estimate.marginal_product_log_probability
        ),
        "log_gap_joint_minus_product": signed_log_gap,
        "absolute_log_gap": getattr(
            estimate,
            "absolute_log_gap",
            abs(signed_log_gap),
        ),
        "probability_ratio_joint_over_product": safe_probability_ratio(
            signed_log_gap
        ),
    }


# Serialize the complete probe at one canonical state: the frozen top-K pool,
# original marginal factors, and every exact subset estimate.
def probe_to_dict(probe: ProbeResult) -> dict[str, Any]:
    return {
        "remaining_mask_count": probe.remaining_mask_count,
        "max_k": probe.max_k,
        "subset_sizes": list(probe.subset_sizes),
        "estimate_count": len(probe.estimates),
        "selected": [prediction_to_dict(item) for item in probe.selected],
        "marginal_log_factors": list(probe.marginal_log_factors),
        "estimates": [estimate_to_dict(item) for item in probe.estimates],
    }


# Serialize the canonical c_t -> c_{t+1} transition and nest its full probe.
def step_to_dict(
    *,
    run_id: str,
    rollout_id: str,
    seed: int,
    step: RolloutStep,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "rollout_id": rollout_id,
        "seed": seed,
        "step_index": step.step_index,
        "committed_count_before": step.committed_count_before,
        "committed_count_after": step.committed_count_after,
        "remaining_mask_count_before": step.remaining_mask_count_before,
        "remaining_mask_count_after": step.remaining_mask_count_after,
        "progress_before": step.progress_before,
        "progress_after": step.progress_after,
        "probe_elapsed_seconds": step.probe_elapsed_seconds,
        "commit_elapsed_seconds": step.commit_elapsed_seconds,
        "total_step_elapsed_seconds": step.total_elapsed_seconds,
        "committed": prediction_to_dict(step.committed),
        "probe": probe_to_dict(step.probe),
    }


# Resolve one estimate's one-based confidence ranks back to the frozen proposal
# objects so the tidy row records the subset's actual positions and token IDs.
def subset_predictions(
    probe: ProbeResult,
    estimate: FactorizationEstimate,
) -> tuple[TokenPrediction, ...]:
    predictions: list[TokenPrediction] = []
    for rank in estimate.subset_ranks:
        if rank < 1 or rank > len(probe.selected):
            raise RuntimeError(
                f"Subset rank {rank} is outside the selected top-K pool."
            )
        predictions.append(probe.selected[rank - 1])
    return tuple(predictions)


# Build one flat, analysis-friendly estimates.jsonl row. The pair
# (step_index, subset_mask) uniquely identifies a measurement within a run.
def estimate_row(
    *,
    run_id: str,
    rollout_id: str,
    seed: int,
    step: RolloutStep,
    estimate: FactorizationEstimate,
) -> dict[str, Any]:
    subset = subset_predictions(step.probe, estimate)
    row = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "rollout_id": rollout_id,
        "seed": seed,
        "step_index": step.step_index,
        "committed_count_before": step.committed_count_before,
        "remaining_mask_count_before": step.remaining_mask_count_before,
        "progress_before": step.progress_before,
        "committed_sequence_position": step.committed.sequence_position,
        "committed_canvas_position": step.committed.canvas_position,
        "committed_token_id": step.committed.token_id,
        "committed_token_text_repr": step.committed.token_text,
        "committed_confidence": step.committed.probability,
        "probe_elapsed_seconds": step.probe_elapsed_seconds,
        "max_k": step.probe.max_k,
        "requested_subset_sizes": list(step.probe.subset_sizes),
        "selected_sequence_positions": [
            item.sequence_position for item in step.probe.selected
        ],
        "selected_canvas_positions": [
            item.canvas_position for item in step.probe.selected
        ],
        "selected_token_ids": [
            item.token_id for item in step.probe.selected
        ],
        "selected_token_text_repr": [
            item.token_text for item in step.probe.selected
        ],
        # Binary masks are zero-padded to K bits, so K=6 subset {1,3}
        # appears as ``000101``. Ranks remain the human-readable identity.
        "subset_mask_binary": format(
            estimate.subset_mask,
            f"0{step.probe.max_k}b",
        ),
        "subset_sequence_positions": [
            item.sequence_position for item in subset
        ],
        "subset_canvas_positions": [item.canvas_position for item in subset],
        "subset_token_ids": [item.token_id for item in subset],
        "subset_token_text_repr": [item.token_text for item in subset],
        "subset_original_probabilities": [item.probability for item in subset],
        "subset_original_log_probabilities": [
            item.log_probability for item in subset
        ],
    }
    row.update(estimate_to_dict(estimate))
    return row


# -----------------------------------------------------------------------------
# Streaming JSONL writer and terminal progress reporting
# -----------------------------------------------------------------------------

# Summarize all subsets of each requested size without printing 57 individual
# measurements on every K=6 step.
def format_subset_size_progress(probe: ProbeResult) -> str:
    chunks: list[str] = []
    for subset_size in probe.subset_sizes:
        estimates = probe.estimates_for_size(subset_size)
        absolute_log_gaps = [item.absolute_log_gap for item in estimates]
        chunks.append(
            f"s={subset_size} n={len(estimates)} "
            f"med|log|={statistics.median(absolute_log_gaps):.4f} "
            f"max|log|={max(absolute_log_gaps):.4f}"
        )
    return "  ".join(chunks)


# Format one exact subset for the optional verbose progress mode.
def format_subset_detail(estimate: FactorizationEstimate) -> str:
    return (
        f"S={estimate.subset_ranks} mask={estimate.subset_mask} "
        f"|log(P/Q)|={estimate.absolute_log_gap:.6f} "
        f"log(P/Q)={estimate.log_gap_joint_minus_product:+.6f} "
        f"|P-Q|={estimate.absolute_difference:.6e}"
    )


# Stream one validated step plus all of its flat subset rows immediately, so a
# long run retains useful data even if a later step or the cluster job fails.
class StepJsonlWriter:
    def __init__(
        self,
        *,
        run_dir: Path,
        run_id: str,
        rollout_id: str,
        seed: int,
        print_every: int,
        print_all_subsets: bool,
        fsync_every_step: bool,
        adapter: DreamAdapter,
        initial_state: DreamState,
    ) -> None:
        self.run_id = run_id
        self.rollout_id = rollout_id
        self.seed = seed
        self.print_every = print_every
        self.print_all_subsets = print_all_subsets
        self.fsync_every_step = fsync_every_step
        self.adapter = adapter

        # The callback receives a RolloutStep rather than c_{t+1}; reconstruct
        # the exact canvas incrementally from the known initial canvas.
        self.canvas_token_ids = list(canvas_token_ids(initial_state))

        self.steps_handle = (run_dir / "steps.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        self.estimates_handle = (run_dir / "estimates.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )

    # Close both streams; the context manager calls this on success or failure.
    def close(self) -> None:
        self.steps_handle.close()
        self.estimates_handle.close()

    # Return the writer itself when entering the ``with`` block.
    def __enter__(self) -> "StepJsonlWriter":
        return self

    # Always close open JSONL files when control leaves the ``with`` block.
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # Persist one fully validated rollout step and all subset measurements.
    def __call__(self, step: RolloutStep) -> None:
        # The callback runs after x_1 has been permanently committed. Update
        # the exact canvas snapshot and decode the partially denoised sequence.
        self.canvas_token_ids[step.committed.canvas_position] = (
            step.committed.token_id
        )
        canvas_after_with_masks = self.adapter.tokenizer.decode(
            self.canvas_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        canvas_after_without_specials = self.adapter.tokenizer.decode(
            self.canvas_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(canvas_after_with_masks, str):
            raise TypeError("Tokenizer decode did not return a string.")
        if not isinstance(canvas_after_without_specials, str):
            raise TypeError("Tokenizer decode did not return a string.")

        # Write the nested per-step audit record.
        step_record = step_to_dict(
            run_id=self.run_id,
            rollout_id=self.rollout_id,
            seed=self.seed,
            step=step,
        )
        step_record["canvas_token_ids_after"] = list(self.canvas_token_ids)
        step_record["canvas_after_with_masks"] = canvas_after_with_masks
        step_record["canvas_after_without_specials"] = (
            canvas_after_without_specials
        )
        self.steps_handle.write(
            json.dumps(step_record, ensure_ascii=False, allow_nan=False) + "\n"
        )

        # Write one flat row for every exact subset at this state.
        for estimate in step.probe.estimates:
            row = estimate_row(
                run_id=self.run_id,
                rollout_id=self.rollout_id,
                seed=self.seed,
                step=step,
                estimate=estimate,
            )
            self.estimates_handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )

        # Flush Python buffers every step. Optional fsync also asks the OS to
        # push data to storage before the next expensive model computation.
        self.steps_handle.flush()
        self.estimates_handle.flush()
        if self.fsync_every_step:
            os.fsync(self.steps_handle.fileno())
            os.fsync(self.estimates_handle.fileno())

        # Print a compact by-size summary, with exact subset lines only when the
        # user explicitly requests the much noisier verbose mode.
        if self.print_every and (step.step_index + 1) % self.print_every == 0:
            metric_text = format_subset_size_progress(step.probe)
            print(
                f"step={step.step_index:02d} "
                f"masks={step.remaining_mask_count_before:02d}"
                f"->{step.remaining_mask_count_after:02d} "
                f"canvas_pos={step.committed.canvas_position:02d} "
                f"token={step.committed.token_text} "
                f"confidence={step.committed.probability:.6f} "
                f"seconds={step.total_elapsed_seconds:.3f}  "
                f"{metric_text}",
                flush=True,
            )
            if self.print_all_subsets:
                for estimate in step.probe.estimates:
                    print(f"  {format_subset_detail(estimate)}", flush=True)


# -----------------------------------------------------------------------------
# Summary-statistic helpers
# -----------------------------------------------------------------------------

# Compute a deterministic linearly interpolated percentile without depending on
# pandas/numpy in the experiment runner.
def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1].")

    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]

    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


# Return the same core descriptive statistics for each scalar metric.
def distribution_summary(values: Sequence[float]) -> dict[str, Any]:
    normalized = [float(value) for value in values]
    if not normalized:
        raise ValueError("Cannot summarize an empty sequence.")
    return {
        "count": len(normalized),
        "mean": statistics.fmean(normalized),
        "median": statistics.median(normalized),
        "sample_standard_deviation": (
            statistics.stdev(normalized) if len(normalized) > 1 else None
        ),
        "minimum": min(normalized),
        "percentile_25": percentile(normalized, 0.25),
        "percentile_75": percentile(normalized, 0.75),
        "maximum": max(normalized),
    }


# Summarize raw and log-space gaps for any collection of subset estimates.
def summarize_estimates(
    estimates: Sequence[FactorizationEstimate],
) -> dict[str, Any]:
    if not estimates:
        raise ValueError("Cannot summarize an empty estimate collection.")

    signed_difference = [
        item.difference_joint_minus_product for item in estimates
    ]
    absolute_difference = [abs(value) for value in signed_difference]
    signed_log_gap = [item.log_gap_joint_minus_product for item in estimates]
    absolute_log_gap = [abs(value) for value in signed_log_gap]

    return {
        "count": len(estimates),
        "signed_difference": distribution_summary(signed_difference),
        "absolute_difference": distribution_summary(absolute_difference),
        "signed_log_gap": distribution_summary(signed_log_gap),
        "absolute_log_gap": distribution_summary(absolute_log_gap),
        "fraction_joint_greater_than_product": (
            sum(value > 0 for value in signed_log_gap) / len(signed_log_gap)
        ),
        "fraction_joint_equal_to_product": (
            sum(value == 0 for value in signed_log_gap) / len(signed_log_gap)
        ),
    }


# Pool all exact subsets with the same cardinality across all rollout states.
def aggregate_by_subset_size(result: RolloutResult) -> dict[str, Any]:
    grouped: dict[int, list[FactorizationEstimate]] = {}
    for step in result.steps:
        for estimate in step.probe.estimates:
            grouped.setdefault(estimate.subset_size, []).append(estimate)

    aggregates: dict[str, Any] = {}
    for subset_size, estimates in sorted(grouped.items()):
        summary = summarize_estimates(estimates)
        summary.update(
            {
                "subset_size": subset_size,
                "unique_subsets": len(
                    {estimate.subset_mask for estimate in estimates}
                ),
                "expected_subsets_per_step": math.comb(
                    result.config.max_k,
                    subset_size,
                ),
            }
        )
        aggregates[str(subset_size)] = summary
    return aggregates


# Aggregate each exact confidence-rank subset separately across denoising steps.
def aggregate_by_subset(result: RolloutResult) -> dict[str, Any]:
    grouped: dict[tuple[int, ...], list[FactorizationEstimate]] = {}
    for step in result.steps:
        for estimate in step.probe.estimates:
            grouped.setdefault(estimate.subset_ranks, []).append(estimate)

    aggregates: dict[str, Any] = {}
    for ranks, estimates in sorted(
        grouped.items(),
        key=lambda item: (len(item[0]), item[0]),
    ):
        first = estimates[0]
        key = ",".join(str(rank) for rank in ranks)
        summary = summarize_estimates(estimates)
        summary.update(
            {
                "subset_ranks": list(ranks),
                "subset_mask": first.subset_mask,
                "subset_size": first.subset_size,
            }
        )
        aggregates[key] = summary
    return aggregates


# -----------------------------------------------------------------------------
# Metadata, final summary, and executable entry point
# -----------------------------------------------------------------------------

# Record enough configuration, software, model, git, and GPU detail to explain
# exactly what produced the JSONL files.
def build_metadata(
    *,
    args: argparse.Namespace,
    config: RolloutConfig,
    run_id: str,
    text: str | None,
    model_id: str,
    adapter: DreamAdapter,
    model_load_seconds: float,
) -> dict[str, Any]:
    device = torch.device(args.device)
    gpu: dict[str, Any] | None = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "command": sys.argv,
        "repository_root": str(REPO_ROOT),
        "git": git_metadata(REPO_ROOT),
        "experiment": {
            "mode": args.mode,
            "text": text,
            "canvas_length": args.canvas_length,
            "num_steps": config.num_steps,
            "max_k": config.max_k,
            "subset_sizes": list(config.subset_sizes),
            "estimates_per_step": config.estimates_per_step,
            "expected_total_estimate_rows": (
                config.num_steps * config.estimates_per_step
            ),
            "subset_enumeration": (
                "all_combinations_of_frozen_top_k_for_requested_sizes"
            ),
            "factorization_order": "original_confidence_descending",
            "position_confidence": "top1_probability",
            "rollout_policy": "greedy_argmax_one_token_per_step",
            "conditional_microbatch_size": (
                config.conditional_microbatch_size
            ),
            "filter_special_tokens": not args.allow_special_tokens,
            "rollout_id": config.rollout_id,
            "seed": config.seed,
        },
        "model": {
            "requested_id": model_id,
            "requested_revision": args.revision,
            "resolved_commit_hash": getattr(adapter.config, "_commit_hash", None),
            "tokenizer_name_or_path": getattr(
                adapter.tokenizer,
                "name_or_path",
                None,
            ),
            "dtype": dtype_name(args.dtype),
            "device": str(device),
            "load_seconds": model_load_seconds,
            "vocab_size": int(getattr(adapter.config, "vocab_size")),
            "bos_token_id": adapter.bos_token_id,
            "mask_token_id": adapter.mask_token_id,
            "eos_token_id": getattr(adapter.tokenizer, "eos_token_id", None),
            "pad_token_id": getattr(adapter.tokenizer, "pad_token_id", None),
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
        },
        "hardware": {"gpu": gpu},
    }


# Build the final human- and machine-readable rollup after every step succeeds.
def build_summary(
    *,
    run_id: str,
    adapter: DreamAdapter,
    result: RolloutResult,
    peak_allocated_bytes: int | None,
    peak_reserved_bytes: int | None,
) -> dict[str, Any]:
    total_estimate_rows = sum(
        len(step.probe.estimates) for step in result.steps
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "run_id": run_id,
        "completed_steps": result.completed_steps,
        "max_k": result.config.max_k,
        "subset_sizes": list(result.config.subset_sizes),
        "estimates_per_step": result.config.estimates_per_step,
        "total_estimate_rows": total_estimate_rows,
        "initial_mask_count": result.initial_mask_count,
        "final_mask_count": result.final_mask_count,
        "total_rollout_seconds": result.total_elapsed_seconds,
        "mean_step_seconds": statistics.fmean(
            step.total_elapsed_seconds for step in result.steps
        ),
        "mean_probe_seconds": statistics.fmean(
            step.probe_elapsed_seconds for step in result.steps
        ),
        "peak_cuda_allocated_bytes": peak_allocated_bytes,
        "peak_cuda_reserved_bytes": peak_reserved_bytes,
        "committed_sequence_positions": list(
            result.committed_sequence_positions
        ),
        "committed_canvas_positions": [
            step.committed.canvas_position for step in result.steps
        ],
        "committed_token_ids": list(result.committed_token_ids),
        "committed_token_text_repr": [
            step.committed.token_text for step in result.steps
        ],
        "final_input_ids": (
            result.final_state.input_ids[0].detach().cpu().tolist()
        ),
        "final_canvas_token_ids": list(canvas_token_ids(result.final_state)),
        "final_canvas_with_masks": decode_canvas(
            adapter,
            result.final_state,
            skip_special_tokens=False,
        ),
        "final_canvas_without_specials": decode_canvas(
            adapter,
            result.final_state,
            skip_special_tokens=True,
        ),
        "aggregate_by_subset_size": aggregate_by_subset_size(result),
        "aggregate_by_subset": aggregate_by_subset(result),
    }


# Seed Python and torch so tie-breaking and any future stochastic extensions can
# be reproduced from metadata.
def set_reproducibility_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Orchestrate validation, model loading, state construction, streaming rollout,
# final summary creation, and failure-status recording.
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # Construct the required rollout configuration before allocating output or
    # GPU resources. This is the final normalization boundary for max_k and
    # subset_sizes.
    config = RolloutConfig(
        max_k=args.max_k,
        subset_sizes=tuple(args.subset_sizes),
        num_steps=args.num_steps,
        conditional_microbatch_size=args.conditional_microbatch_size,
        rollout_id=args.rollout_id,
        seed=args.seed,
    )

    text = resolve_text(args)
    model_id = args.model or DreamAdapter.default_model_id(args.mode)
    run_id = make_run_id(args)
    run_dir = prepare_run_directory(
        args.output_dir,
        run_id,
        overwrite=args.overwrite,
    )

    # A small status file lets cluster monitoring distinguish startup, running,
    # completion, and Python failures without parsing the full log.
    status_path = run_dir / "status.json"
    write_json_atomic(
        status_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "starting",
            "run_id": run_id,
            "started_at_utc": utc_now(),
        },
    )

    set_reproducibility_seed(config.seed)

    try:
        # Load the frozen Dream checkpoint and tokenizer using the adapter's
        # single, shared model-loading path.
        load_start = time.perf_counter()
        adapter = DreamAdapter.from_pretrained(
            ModelLoadOptions(
                model_id=model_id,
                revision=args.revision,
                cache_dir=args.cache_dir,
                local_files_only=args.local_files_only,
                dtype=args.dtype,
                device=args.device,
            ),
            filter_special_tokens=not args.allow_special_tokens,
        )
        model_load_seconds = time.perf_counter() - load_start

        # Build the visible prefix plus fixed fully masked generation canvas.
        initial_state = adapter.prepare_initial_state(
            mode=args.mode,
            canvas_length=args.canvas_length,
            text=text,
        )

        # Persist immutable run context before beginning expensive probes.
        metadata = build_metadata(
            args=args,
            config=config,
            run_id=run_id,
            text=text,
            model_id=model_id,
            adapter=adapter,
            model_load_seconds=model_load_seconds,
        )
        write_json_atomic(run_dir / "metadata.json", metadata)
        write_json_atomic(
            status_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "running",
                "run_id": run_id,
                "started_at_utc": metadata["created_at_utc"],
            },
        )

        print(f"Run directory:  {run_dir}")
        print(f"Checkpoint:     {model_id}")
        print(f"Mode:           {args.mode}")
        print(f"Canvas/steps:   {args.canvas_length}/{config.num_steps}")
        print(f"Candidate pool: K={config.max_k}")
        print(f"Subset sizes:   {config.subset_sizes}")
        print(f"Subsets/step:   {config.estimates_per_step}")
        print(
            f"Expected rows:  "
            f"{config.num_steps * config.estimates_per_step}"
        )
        print(
            "Microbatch:     "
            + (
                "all conditional states"
                if config.conditional_microbatch_size is None
                else str(config.conditional_microbatch_size)
            )
        )
        print()

        # Reset peak counters immediately before the measured rollout so model
        # loading does not contaminate the reported rollout peak.
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # The writer callback persists each validated step as soon as the
        # rollout completes it.
        with StepJsonlWriter(
            run_dir=run_dir,
            run_id=run_id,
            rollout_id=config.rollout_id,
            seed=config.seed,
            print_every=args.print_every,
            print_all_subsets=args.print_all_subsets,
            fsync_every_step=args.fsync_every_step,
            adapter=adapter,
            initial_state=initial_state,
        ) as writer:
            result = run_rollout(
                adapter,
                initial_state,
                config=config,
                on_step=writer,
            )

        # Capture CUDA peaks after the rollout and before process teardown.
        peak_allocated: int | None = None
        peak_reserved: int | None = None
        if device.type == "cuda":
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))

        # Write the final summary and atomically mark the run completed.
        summary = build_summary(
            run_id=run_id,
            adapter=adapter,
            result=result,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
        )
        write_json_atomic(run_dir / "summary.json", summary)
        write_json_atomic(
            status_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "completed",
                "run_id": run_id,
                "completed_at_utc": summary["completed_at_utc"],
                "completed_steps": result.completed_steps,
                "total_estimate_rows": summary["total_estimate_rows"],
                "summary_file": "summary.json",
            },
        )

        print()
        print("Completed rollout")
        print("-----------------")
        print(f"Steps:          {result.completed_steps}")
        print(
            f"Masks:          "
            f"{result.initial_mask_count}->{result.final_mask_count}"
        )
        print(f"Subsets/step:   {config.estimates_per_step}")
        print(f"Estimate rows:  {summary['total_estimate_rows']}")
        print(f"Seconds:        {result.total_elapsed_seconds:.3f}")
        print(f"Steps JSONL:    {run_dir / 'steps.jsonl'}")
        print(f"Tidy JSONL:     {run_dir / 'estimates.jsonl'}")
        print(f"Summary:        {run_dir / 'summary.json'}")
        return 0

    except Exception as exc:
        # Preserve the Python exception and traceback in status.json while also
        # re-raising it so the shell receives a nonzero exit code.
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "run_id": run_id,
            "failed_at_utc": utc_now(),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json_atomic(status_path, failure)
        raise


# Convert main's integer return code into the process exit status.
if __name__ == "__main__":
    raise SystemExit(main())
