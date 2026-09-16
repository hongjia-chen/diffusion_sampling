#!/usr/bin/env python3
"""Audit a completed Dream adaptive-scheduler run before scoring it."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from audit_dream_topk import (
    atomic_write_json,
    load_json,
    read_jsonl,
    resolve_path,
    validate_dataset,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--dataset-metadata", type=Path)
    parser.add_argument("--prefix-length", type=int)
    parser.add_argument("--continuation-length", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.prefix_length is not None and args.prefix_length < 1:
        parser.error("--prefix-length must be positive.")
    if args.continuation_length is not None and args.continuation_length < 1:
        parser.error("--continuation-length must be positive.")
    return args


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-10)


def validate_thresholds(
    values: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(values, list) or not values:
        raise ValueError("Experiment metadata has no thresholds.")
    thresholds: list[dict[str, Any]] = []
    by_label: dict[str, dict[str, Any]] = {}
    for raw in values:
        label = str(raw["label"])
        multiplier = float(raw["multiplier"])
        tau = float(raw["tau"])
        if label in by_label:
            raise ValueError(f"Duplicate threshold label: {label}")
        if (
            not math.isfinite(multiplier)
            or multiplier <= 1.0
            or not math.isfinite(tau)
            or tau < 0.0
            or not close(tau, math.log(multiplier))
        ):
            raise ValueError(f"Invalid threshold definition: {label}")
        value = {"label": label, "multiplier": multiplier, "tau": tau}
        thresholds.append(value)
        by_label[label] = value
    return thresholds, by_label


def validate_timings(
    path: Path,
    *,
    thresholds: Sequence[dict[str, Any]],
    timing_repeats: int,
    expected_examples: int,
    continuation_length: int,
    max_candidates: int,
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    labels = [value["label"] for value in thresholds]
    expected = {
        (label, repeat)
        for label in labels
        for repeat in range(timing_repeats)
    }
    actual = {
        (str(row["threshold_label"]), int(row["timing_repeat"]))
        for row in rows
    }
    if actual != expected or len(rows) != len(expected):
        raise ValueError("Timing rows do not form the threshold/repeat grid.")

    deterministic_stats: dict[str, tuple[Any, ...]] = {}
    for row in rows:
        label = str(row["threshold_label"])
        threshold = next(value for value in thresholds if value["label"] == label)
        if row.get("schedule_type") != "adaptive":
            raise ValueError("A timing row has the wrong schedule type.")
        if not close(float(row["tau"]), threshold["tau"]):
            raise ValueError("A timing row has the wrong tau.")
        if not close(
            float(row["tau_multiplier"]),
            threshold["multiplier"],
        ):
            raise ValueError("A timing row has the wrong multiplier.")
        if int(row["completed_samples"]) != expected_examples:
            raise ValueError("A timing row has the wrong sample count.")
        if int(row["batch_size"]) != 1:
            raise ValueError("Adaptive timing requires batch_size=1.")
        if int(row["conditional_microbatch_size"]) != 1:
            raise ValueError("Adaptive timing requires conditional microbatch 1.")
        if int(row["max_candidates"]) != max_candidates:
            raise ValueError("A timing row has the wrong candidate cap.")
        elapsed = float(row["elapsed_seconds"])
        amortized = float(row["amortized_seconds_per_sample"])
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("A timing elapsed value is invalid.")
        if not close(amortized, elapsed / expected_examples):
            raise ValueError("A timing amortized value is inconsistent.")

        steps = int(row["total_scheduler_steps"])
        probes = int(row["total_conditional_probes"])
        histogram = [
            int(value) for value in row["accepted_prefix_histogram"]
        ]
        if len(histogram) != max_candidates or any(x < 0 for x in histogram):
            raise ValueError("An accepted-prefix histogram is invalid.")
        if steps <= 0 or probes < 0 or sum(histogram) != steps:
            raise ValueError("A timing scheduler count is inconsistent.")
        if sum(
            (index + 1) * count for index, count in enumerate(histogram)
        ) != expected_examples * continuation_length:
            raise ValueError("A timing histogram does not fill every canvas.")
        if probes > steps * (max_candidates - 1):
            raise ValueError("A timing conditional-probe count is impossible.")
        expected_mean = expected_examples * continuation_length / steps
        if not close(float(row["mean_accepted_prefix_length"]), expected_mean):
            raise ValueError("A timing mean accepted prefix is inconsistent.")
        stats = (steps, probes, tuple(histogram))
        prior = deterministic_stats.setdefault(label, stats)
        if prior != stats:
            raise ValueError(
                "Deterministic scheduler statistics differ across repeats."
            )

    if len({str(row["hardware"]) for row in rows}) != 1:
        raise ValueError("Hardware differs across timing rows.")
    if len({str(row["precision"]) for row in rows}) != 1:
        raise ValueError("Precision differs across timing rows.")
    return rows


def validate_generations(
    run_dir: Path,
    artifacts: dict[str, Any],
    *,
    thresholds: Sequence[dict[str, Any]],
    expected_examples: int,
    continuation_length: int,
    mask_token_id: int,
) -> None:
    expected_ids = list(range(expected_examples))
    for threshold in thresholds:
        label = threshold["label"]
        relative = artifacts.get(label)
        if not relative:
            raise ValueError(f"No generation artifact for {label}.")
        rows = read_jsonl(resolve_path(relative, base=run_dir))
        if len(rows) != expected_examples:
            raise ValueError(f"Generation count is wrong for {label}.")
        if [int(row["example_id"]) for row in rows] != expected_ids:
            raise ValueError(f"Generation IDs are wrong for {label}.")
        for row in rows:
            if (
                row.get("schedule_type") != "adaptive"
                or str(row["threshold_label"]) != label
                or not close(float(row["tau"]), threshold["tau"])
            ):
                raise ValueError(f"Generation policy fields are wrong for {label}.")
            token_ids = [int(value) for value in row["generated_token_ids"]]
            if len(token_ids) != continuation_length:
                raise ValueError(f"Wrong continuation length for {label}.")
            if mask_token_id in token_ids:
                raise ValueError(f"MASK remains in a generation for {label}.")
            if not isinstance(row.get("generated_text"), str):
                raise ValueError(f"Generated text is missing for {label}.")


def validate_numeric_trace(
    row: dict[str, Any],
    *,
    tau: float,
    candidate_count: int,
) -> int:
    marginal = [float(value) for value in row["marginal_log_factors"]]
    conditional = [
        float(value) for value in row["conditional_log_factors"]
    ]
    cumulative_marginal = [
        float(value) for value in row["cumulative_log_marginals"]
    ]
    cumulative_joint = [
        float(value) for value in row["cumulative_log_joints"]
    ]
    errors = [float(value) for value in row["cumulative_errors"]]
    probes = int(row["conditional_probes"])

    if len(marginal) != candidate_count:
        raise ValueError("A trajectory marginal-factor list has wrong length.")
    if probes != len(conditional) or probes > candidate_count - 1:
        raise ValueError("A trajectory conditional-probe count is wrong.")
    evaluated = probes + 1
    if not (
        len(cumulative_marginal)
        == len(cumulative_joint)
        == len(errors)
        == evaluated
    ):
        raise ValueError("A trajectory cumulative trace has wrong lengths.")
    all_values = (
        marginal
        + conditional
        + cumulative_marginal
        + cumulative_joint
        + errors
    )
    if any(not math.isfinite(value) for value in all_values):
        raise ValueError("A trajectory numeric trace is non-finite.")

    expected_marginal: list[float] = []
    running = 0.0
    for value in marginal[:evaluated]:
        running = math.fsum((running, value))
        expected_marginal.append(running)
    expected_joint = [marginal[0]]
    for value in conditional:
        expected_joint.append(math.fsum((expected_joint[-1], value)))
    expected_errors = [
        abs(joint - independent)
        for joint, independent in zip(expected_joint, expected_marginal)
    ]
    if any(
        not close(found, expected)
        for found, expected in zip(cumulative_marginal, expected_marginal)
    ):
        raise ValueError("Cumulative marginal log probabilities are wrong.")
    if any(
        not close(found, expected)
        for found, expected in zip(cumulative_joint, expected_joint)
    ):
        raise ValueError("Cumulative joint log probabilities are wrong.")
    if any(
        not close(found, expected)
        for found, expected in zip(errors, expected_errors)
    ):
        raise ValueError("Cumulative dependency errors are wrong.")
    if not close(errors[0], 0.0):
        raise ValueError("Singleton dependency error must be zero.")

    accepted = 1
    failed = False
    for error in errors[1:]:
        if error <= tau:
            if failed:
                raise ValueError("A trace continued after a failed prefix.")
            accepted += 1
        else:
            failed = True
    if failed:
        if probes != candidate_count - 1 and errors[-1] <= tau:
            raise ValueError("A short-circuit trace did not end on failure.")
        if errors[-1] <= tau:
            raise ValueError("A failed trace has no terminal violation.")
    elif probes != candidate_count - 1:
        raise ValueError("A passing trace stopped before testing all candidates.")
    return accepted


def validate_trajectories(
    path: Path,
    *,
    thresholds: Sequence[dict[str, Any]],
    trajectory_ids: Sequence[int],
    prefix_length: int,
    continuation_length: int,
    max_candidates: int,
    mask_token_id: int,
) -> None:
    rows = read_jsonl(path)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    allowed_labels = {value["label"] for value in thresholds}
    for row in rows:
        key = (str(row["threshold_label"]), int(row["example_id"]))
        if key[0] not in allowed_labels or key[1] not in trajectory_ids:
            raise ValueError("A trajectory has an unknown policy or example.")
        grouped.setdefault(key, []).append(row)
    expected_groups = {
        (value["label"], example_id)
        for value in thresholds
        for example_id in trajectory_ids
    }
    if set(grouped) != expected_groups:
        raise ValueError("Trajectory policy/example coverage is incomplete.")

    threshold_by_label = {
        value["label"]: value for value in thresholds
    }
    for (label, _), group in grouped.items():
        group.sort(key=lambda row: int(row["step_index"]))
        if [int(row["step_index"]) for row in group] != list(range(len(group))):
            raise ValueError("Trajectory step indices are not contiguous.")
        tau = float(threshold_by_label[label]["tau"])
        remaining = continuation_length
        committed: set[int] = set()
        for row in group:
            before = int(row["remaining_masks_before"])
            after = int(row["remaining_masks_after"])
            candidate_count = min(max_candidates, before)
            canvas = [int(value) for value in row["candidate_canvas_positions"]]
            sequence = [
                int(value) for value in row["candidate_sequence_positions"]
            ]
            token_ids = [int(value) for value in row["candidate_token_ids"]]
            token_text = [str(value) for value in row["candidate_token_text"]]
            confidence = [
                float(value) for value in row["candidate_confidences"]
            ]
            if before != remaining:
                raise ValueError("Trajectory remaining-mask progression is wrong.")
            if not (
                len(canvas)
                == len(sequence)
                == len(token_ids)
                == len(token_text)
                == len(confidence)
                == candidate_count
            ):
                raise ValueError("Trajectory candidate fields have wrong lengths.")
            if len(set(canvas)) != candidate_count:
                raise ValueError("A trajectory repeats a candidate position.")
            if committed.intersection(canvas):
                raise ValueError("A trajectory proposes an already-filled position.")
            if any(value < 0 or value >= continuation_length for value in canvas):
                raise ValueError("A candidate canvas position is out of range.")
            if sequence != [
                1 + prefix_length + value for value in canvas
            ]:
                raise ValueError("Canvas and sequence positions disagree.")
            if mask_token_id in token_ids:
                raise ValueError("A trajectory proposed the MASK token.")
            if any(
                not math.isfinite(value) or value < 0.0 or value > 1.0
                for value in confidence
            ):
                raise ValueError("A trajectory confidence is invalid.")
            if any(
                confidence[index] < confidence[index + 1]
                for index in range(len(confidence) - 1)
            ):
                raise ValueError("Frozen candidates are not confidence-ranked.")

            accepted = validate_numeric_trace(
                row,
                tau=tau,
                candidate_count=candidate_count,
            )
            if int(row["accepted_prefix_length"]) != accepted:
                raise ValueError("Recorded accepted prefix violates the policy.")
            committed_canvas = [
                int(value) for value in row["committed_canvas_positions"]
            ]
            committed_sequence = [
                int(value) for value in row["committed_sequence_positions"]
            ]
            committed_tokens = [
                int(value) for value in row["committed_token_ids"]
            ]
            if (
                committed_canvas != canvas[:accepted]
                or committed_sequence != sequence[:accepted]
                or committed_tokens != token_ids[:accepted]
            ):
                raise ValueError("Committed values are not the frozen prefix.")
            if after != before - accepted:
                raise ValueError("A trajectory commitment count is wrong.")
            committed.update(committed_canvas)
            remaining = after
        if remaining != 0 or len(committed) != continuation_length:
            raise ValueError("A trajectory did not fill the canvas exactly once.")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    metadata = load_json(run_dir / "metadata.json")
    if metadata.get("status") != "complete":
        raise ValueError("Run metadata status is not complete.")
    experiment = metadata["experiment"]
    if (
        experiment.get("schedule_type") != "adaptive"
        or experiment.get("policy") != "short_circuit_frozen_proposals"
    ):
        raise ValueError("Run metadata does not describe the accepted policy.")
    prefix_length = int(experiment["prefix_length"])
    continuation_length = int(experiment["continuation_length"])
    if args.prefix_length is not None and args.prefix_length != prefix_length:
        raise ValueError("--prefix-length disagrees with run metadata.")
    if (
        args.continuation_length is not None
        and args.continuation_length != continuation_length
    ):
        raise ValueError("--continuation-length disagrees with run metadata.")
    if int(experiment["batch_size"]) != 1:
        raise ValueError("Adaptive outer batch size must be one.")
    if int(experiment["conditional_microbatch_size"]) != 1:
        raise ValueError("Adaptive conditional microbatch size must be one.")
    max_candidates = int(experiment["max_candidates"])
    thresholds, _ = validate_thresholds(experiment["thresholds"])

    dataset_info = metadata["dataset"]
    dataset_path = (
        args.dataset.expanduser().resolve()
        if args.dataset
        else Path(dataset_info["path"]).expanduser().resolve()
    )
    dataset_metadata_path = (
        args.dataset_metadata.expanduser().resolve()
        if args.dataset_metadata
        else Path(dataset_info["metadata_path"]).expanduser().resolve()
    )
    expected_examples = int(dataset_info["num_examples"])
    _, dataset_metadata = validate_dataset(
        dataset_path,
        dataset_metadata_path,
        expected_digest=str(dataset_info["sha256"]),
        prefix_length=prefix_length,
        continuation_length=continuation_length,
        expected_examples=expected_examples,
    )
    mask_token_id = int(dataset_metadata["tokenizer"]["mask_token_id"])
    artifacts = metadata["artifacts"]
    timings = validate_timings(
        resolve_path(artifacts["timings"], base=run_dir),
        thresholds=thresholds,
        timing_repeats=int(experiment["timing_repeats"]),
        expected_examples=expected_examples,
        continuation_length=continuation_length,
        max_candidates=max_candidates,
    )
    validate_generations(
        run_dir,
        artifacts["generations"],
        thresholds=thresholds,
        expected_examples=expected_examples,
        continuation_length=continuation_length,
        mask_token_id=mask_token_id,
    )
    trajectory_ids = [
        int(value) for value in experiment["trajectory_example_ids"]
    ]
    validate_trajectories(
        resolve_path(artifacts["trajectories"], base=run_dir),
        thresholds=thresholds,
        trajectory_ids=trajectory_ids,
        prefix_length=prefix_length,
        continuation_length=continuation_length,
        max_candidates=max_candidates,
        mask_token_id=mask_token_id,
    )
    checks = metadata.get("determinism_checks", {})
    if any(checks.get(value["label"]) is not True for value in thresholds):
        raise ValueError("One or more determinism checks did not pass.")
    return {
        "schema_version": 1,
        "status": "verified",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "dataset_sha256": dataset_info["sha256"],
        "num_examples": expected_examples,
        "prefix_length": prefix_length,
        "continuation_length": continuation_length,
        "threshold_labels": [value["label"] for value in thresholds],
        "timing_records": len(timings),
        "trajectory_example_ids": trajectory_ids,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit(args)
    output = (
        args.output.expanduser().resolve()
        if args.output
        else args.run_dir.expanduser().resolve() / "audit.json"
    )
    atomic_write_json(output, result)
    print(f"[verified] adaptive audit: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
