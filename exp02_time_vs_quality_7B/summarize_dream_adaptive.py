#!/usr/bin/env python3
"""Combine adaptive Dream timing statistics and MAUVE into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Sequence

from summarize_dream_topk import load_json, read_jsonl, sample_std


FIELDNAMES = (
    "schedule_type",
    "threshold_label",
    "tau_multiplier",
    "tau",
    "N",
    "P",
    "L",
    "batch_size",
    "conditional_microbatch_size",
    "max_candidates",
    "hardware",
    "precision",
    "timing_repeats",
    "mean_elapsed_seconds",
    "std_elapsed_seconds",
    "mean_amortized_seconds_per_sample",
    "std_amortized_seconds_per_sample",
    "mean_scheduler_steps_per_sample",
    "mean_conditional_probes_per_sample",
    "mean_model_forwards_per_sample",
    "mean_accepted_prefix_length",
    "accepted_prefix_histogram",
    "MAUVE",
    "frontier_integral",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--timings", type=Path)
    parser.add_argument("--mauve", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def atomic_write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def summarize(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path]:
    run_dir = args.run_dir.expanduser().resolve()
    metadata = load_json(run_dir / "metadata.json")
    audit = load_json(run_dir / "audit.json")
    if metadata.get("status") != "complete":
        raise ValueError("Adaptive run is not complete.")
    if audit.get("status") != "verified":
        raise ValueError("Adaptive audit has not passed.")
    experiment = metadata["experiment"]
    if experiment.get("schedule_type") != "adaptive":
        raise ValueError("Run is not an adaptive-scheduler experiment.")

    timings_path = (
        args.timings.expanduser().resolve()
        if args.timings
        else run_dir / "timings.jsonl"
    )
    mauve_path = (
        args.mauve.expanduser().resolve()
        if args.mauve
        else run_dir / "adaptive_mauve.jsonl"
    )
    timings = read_jsonl(timings_path)
    scores = read_jsonl(mauve_path)
    thresholds = experiment["thresholds"]
    labels = [str(value["label"]) for value in thresholds]
    threshold_by_label = {
        str(value["label"]): value for value in thresholds
    }
    num_examples = int(metadata["dataset"]["num_examples"])
    prefix_length = int(experiment["prefix_length"])
    continuation_length = int(experiment["continuation_length"])
    repeats = int(experiment["timing_repeats"])
    batch_size = int(experiment["batch_size"])
    microbatch = int(experiment["conditional_microbatch_size"])
    max_candidates = int(experiment["max_candidates"])
    dataset_digest = str(metadata["dataset"]["sha256"])

    scores_by_label: dict[str, dict[str, Any]] = {}
    score_settings: dict[str, Any] | None = None
    for row in scores:
        label = str(row["threshold_label"])
        if label in scores_by_label or label not in threshold_by_label:
            raise ValueError("MAUVE has a duplicate or unknown threshold.")
        if int(row["num_examples"]) != num_examples:
            raise ValueError("MAUVE sample count is wrong.")
        if int(row["prefix_length"]) != prefix_length:
            raise ValueError("MAUVE prefix length is wrong.")
        if int(row["continuation_length"]) != continuation_length:
            raise ValueError("MAUVE continuation length is wrong.")
        if str(row["dataset_sha256"]) != dataset_digest:
            raise ValueError("MAUVE dataset checksum is wrong.")
        if score_settings is None:
            score_settings = row["settings"]
        elif score_settings != row["settings"]:
            raise ValueError("MAUVE settings differ across thresholds.")
        score = float(row["mauve"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("A MAUVE score is invalid.")
        scores_by_label[label] = row
    if set(scores_by_label) != set(labels):
        raise ValueError("MAUVE does not cover every adaptive threshold.")

    timings_by_label: dict[str, list[dict[str, Any]]] = {
        label: [] for label in labels
    }
    for row in timings:
        label = str(row["threshold_label"])
        if label not in timings_by_label:
            raise ValueError("Timing data has an unknown threshold.")
        timings_by_label[label].append(row)

    output_rows: list[dict[str, Any]] = []
    for label in labels:
        rows = sorted(
            timings_by_label[label],
            key=lambda row: int(row["timing_repeat"]),
        )
        if len(rows) != repeats or [
            int(row["timing_repeat"]) for row in rows
        ] != list(range(repeats)):
            raise ValueError(f"Timing repetitions are incomplete for {label}.")
        if any(int(row["completed_samples"]) != num_examples for row in rows):
            raise ValueError("A timing sample count is wrong.")
        hardware = {str(row["hardware"]) for row in rows}
        precision = {str(row["precision"]) for row in rows}
        if len(hardware) != 1 or len(precision) != 1:
            raise ValueError("Hardware or precision differs within a threshold.")
        if any(int(row["batch_size"]) != batch_size for row in rows):
            raise ValueError("Outer batch size differs in timing data.")
        if any(
            int(row["conditional_microbatch_size"]) != microbatch
            for row in rows
        ):
            raise ValueError("Conditional microbatch differs in timing data.")

        elapsed = [float(row["elapsed_seconds"]) for row in rows]
        amortized = [
            float(row["amortized_seconds_per_sample"]) for row in rows
        ]
        scheduler_steps = [
            int(row["total_scheduler_steps"]) for row in rows
        ]
        probes = [int(row["total_conditional_probes"]) for row in rows]
        histograms = [
            tuple(int(value) for value in row["accepted_prefix_histogram"])
            for row in rows
        ]
        if len(set(scheduler_steps)) != 1:
            raise ValueError("Scheduler steps differ across deterministic repeats.")
        if len(set(probes)) != 1 or len(set(histograms)) != 1:
            raise ValueError("Adaptive decisions differ across repeats.")
        steps = scheduler_steps[0]
        conditional = probes[0]
        threshold = threshold_by_label[label]
        output_rows.append(
            {
                "schedule_type": "adaptive",
                "threshold_label": label,
                "tau_multiplier": float(threshold["multiplier"]),
                "tau": float(threshold["tau"]),
                "N": num_examples,
                "P": prefix_length,
                "L": continuation_length,
                "batch_size": batch_size,
                "conditional_microbatch_size": microbatch,
                "max_candidates": max_candidates,
                "hardware": next(iter(hardware)),
                "precision": next(iter(precision)),
                "timing_repeats": repeats,
                "mean_elapsed_seconds": statistics.fmean(elapsed),
                "std_elapsed_seconds": sample_std(elapsed),
                "mean_amortized_seconds_per_sample": statistics.fmean(
                    amortized
                ),
                "std_amortized_seconds_per_sample": sample_std(amortized),
                "mean_scheduler_steps_per_sample": steps / num_examples,
                "mean_conditional_probes_per_sample": (
                    conditional / num_examples
                ),
                "mean_model_forwards_per_sample": (
                    (steps + conditional) / num_examples
                ),
                "mean_accepted_prefix_length": (
                    num_examples * continuation_length / steps
                ),
                "accepted_prefix_histogram": json.dumps(
                    list(histograms[0]),
                    separators=(",", ":"),
                ),
                "MAUVE": float(scores_by_label[label]["mauve"]),
                "frontier_integral": float(
                    scores_by_label[label]["frontier_integral"]
                ),
            }
        )

    if len({str(row["hardware"]) for row in timings}) != 1:
        raise ValueError("Hardware differs across thresholds.")
    if len({str(row["precision"]) for row in timings}) != 1:
        raise ValueError("Precision differs across thresholds.")
    output = (
        args.output.expanduser().resolve()
        if args.output
        else run_dir / "adaptive_summary.csv"
    )
    return output_rows, output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows, output = summarize(args)
    atomic_write_csv(output, rows)
    print(f"[done] adaptive summary: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
