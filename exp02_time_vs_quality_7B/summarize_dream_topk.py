#!/usr/bin/env python3
"""Combine verified Dream timing repetitions and MAUVE scores into a CSV."""

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


FIELDNAMES = (
    "k",
    "N",
    "P",
    "L",
    "batch_size",
    "hardware",
    "precision",
    "timing_repeats",
    "mean_elapsed_seconds",
    "std_elapsed_seconds",
    "mean_amortized_seconds_per_sample",
    "std_amortized_seconds_per_sample",
    "MAUVE",
    "frontier_integral",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--prefix-length", type=int)
    parser.add_argument("--continuation-length", type=int)
    parser.add_argument("--k-values", type=int, nargs="+")
    parser.add_argument("--timings", type=Path)
    parser.add_argument("--mauve", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.prefix_length is not None and args.prefix_length < 1:
        parser.error("--prefix-length must be at least 1.")
    if args.continuation_length is not None and args.continuation_length < 1:
        parser.error("--continuation-length must be at least 1.")
    if args.k_values is not None:
        if any(k < 1 for k in args.k_values):
            parser.error("--k-values must contain positive integers.")
        if len(set(args.k_values)) != len(args.k_values):
            parser.error("--k-values must not contain duplicates.")
    return args


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {path}.")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}.")
            rows.append(value)
    if not rows:
        raise ValueError(f"No records found in {path}.")
    return rows


def sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def atomic_write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
) -> None:
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


def validate_requested_configuration(
    args: argparse.Namespace,
    experiment: dict[str, Any],
) -> tuple[int, int, list[int]]:
    prefix_length = int(experiment["prefix_length"])
    continuation_length = int(experiment["continuation_length"])
    k_values = [int(value) for value in experiment["k_values"]]
    if args.prefix_length is not None and args.prefix_length != prefix_length:
        raise ValueError("--prefix-length disagrees with run metadata.")
    if (
        args.continuation_length is not None
        and args.continuation_length != continuation_length
    ):
        raise ValueError("--continuation-length disagrees with run metadata.")
    if args.k_values is not None and args.k_values != k_values:
        raise ValueError("--k-values disagrees with run metadata.")
    return prefix_length, continuation_length, k_values


def summarize(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path]:
    run_dir = args.run_dir.expanduser().resolve()
    run_metadata = load_json(run_dir / "metadata.json")
    audit = load_json(run_dir / "audit.json")
    if run_metadata.get("status") != "complete":
        raise ValueError("Run is not complete.")
    if audit.get("status") != "verified":
        raise ValueError("Run audit has not passed.")

    prefix_length, continuation_length, k_values = (
        validate_requested_configuration(args, run_metadata["experiment"])
    )
    timings_path = (
        args.timings.expanduser().resolve()
        if args.timings
        else run_dir / "timings.jsonl"
    )
    mauve_path = (
        args.mauve.expanduser().resolve()
        if args.mauve
        else run_dir / "mauve.jsonl"
    )
    timing_rows = read_jsonl(timings_path)
    mauve_rows = read_jsonl(mauve_path)
    expected_examples = int(run_metadata["dataset"]["num_examples"])
    dataset_sha256 = str(run_metadata["dataset"]["sha256"])

    mauve_by_k: dict[int, dict[str, Any]] = {}
    settings: dict[str, Any] | None = None
    for row in mauve_rows:
        k = int(row["k"])
        if k in mauve_by_k:
            raise ValueError(f"Duplicate MAUVE record for k={k}.")
        if int(row["num_examples"]) != expected_examples:
            raise ValueError("MAUVE sample count mismatch.")
        if int(row["prefix_length"]) != prefix_length:
            raise ValueError("MAUVE prefix length mismatch.")
        if int(row["continuation_length"]) != continuation_length:
            raise ValueError("MAUVE continuation length mismatch.")
        if str(row["dataset_sha256"]) != dataset_sha256:
            raise ValueError("MAUVE dataset checksum mismatch.")
        if settings is None:
            settings = row["settings"]
        elif row["settings"] != settings:
            raise ValueError("MAUVE settings differ across k values.")
        score = float(row["mauve"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Invalid MAUVE score.")
        mauve_by_k[k] = row
    if set(mauve_by_k) != set(k_values):
        raise ValueError("MAUVE results do not cover every k value.")

    timing_by_k: dict[int, list[dict[str, Any]]] = {
        k: [] for k in k_values
    }
    for row in timing_rows:
        k = int(row["k"])
        if k not in timing_by_k:
            raise ValueError(f"Unknown timing k={k}.")
        if int(row["completed_samples"]) != expected_examples:
            raise ValueError("Timing sample count mismatch.")
        timing_by_k[k].append(row)

    output_rows: list[dict[str, Any]] = []
    expected_repeats = int(run_metadata["experiment"]["timing_repeats"])
    expected_batch_size = int(run_metadata["experiment"]["batch_size"])
    for k in k_values:
        rows = sorted(
            timing_by_k[k],
            key=lambda row: int(row["timing_repeat"]),
        )
        if len(rows) != expected_repeats:
            raise ValueError(f"Wrong timing repeat count for k={k}.")
        if [int(row["timing_repeat"]) for row in rows] != list(
            range(expected_repeats)
        ):
            raise ValueError(f"Timing repeat indices are wrong for k={k}.")
        batch_sizes = {int(row["batch_size"]) for row in rows}
        hardware_values = {str(row["hardware"]) for row in rows}
        precision_values = {str(row["precision"]) for row in rows}
        if batch_sizes != {expected_batch_size}:
            raise ValueError(f"Batch size differs for k={k}.")
        if len(hardware_values) != 1 or len(precision_values) != 1:
            raise ValueError(f"Hardware or precision differs for k={k}.")

        elapsed = [float(row["elapsed_seconds"]) for row in rows]
        amortized = [
            float(row["amortized_seconds_per_sample"]) for row in rows
        ]
        output_rows.append(
            {
                "k": k,
                "N": expected_examples,
                "P": prefix_length,
                "L": continuation_length,
                "batch_size": expected_batch_size,
                "hardware": next(iter(hardware_values)),
                "precision": next(iter(precision_values)),
                "timing_repeats": expected_repeats,
                "mean_elapsed_seconds": statistics.fmean(elapsed),
                "std_elapsed_seconds": sample_std(elapsed),
                "mean_amortized_seconds_per_sample": statistics.fmean(
                    amortized
                ),
                "std_amortized_seconds_per_sample": sample_std(amortized),
                "MAUVE": float(mauve_by_k[k]["mauve"]),
                "frontier_integral": float(
                    mauve_by_k[k]["frontier_integral"]
                ),
            }
        )

    all_hardware = {str(row["hardware"]) for row in timing_rows}
    all_precision = {str(row["precision"]) for row in timing_rows}
    all_batch_sizes = {int(row["batch_size"]) for row in timing_rows}
    if len(all_hardware) != 1 or len(all_precision) != 1:
        raise ValueError("Hardware or precision differs across k schedules.")
    if all_batch_sizes != {expected_batch_size}:
        raise ValueError("Batch size differs across k schedules.")

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else run_dir / "summary.csv"
    )
    return output_rows, output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows, output = summarize(args)
    atomic_write_csv(output, rows)
    print(f"[done] summary: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
