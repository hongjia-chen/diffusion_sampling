#!/usr/bin/env python3
"""Audit a completed Dream top-k run before computing metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--dataset-metadata", type=Path)
    parser.add_argument("--prefix-length", type=int)
    parser.add_argument("--continuation-length", type=int)
    parser.add_argument("--k-values", type=int, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.prefix_length is not None and args.prefix_length < 1:
        parser.error("--prefix-length must be at least 1.")
    if args.continuation_length is not None and args.continuation_length < 1:
        parser.error("--continuation-length must be at least 1.")
    if args.k_values is not None:
        if not args.k_values or any(k < 1 for k in args.k_values):
            parser.error("--k-values must contain positive integers.")
        if len(set(args.k_values)) != len(args.k_values):
            parser.error("--k-values must not contain duplicates.")
    return args


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
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
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def resolve_path(candidate: str | Path, *, base: Path) -> Path:
    path = Path(candidate).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def validate_dataset(
    dataset_path: Path,
    metadata_path: Path,
    *,
    expected_digest: str,
    prefix_length: int,
    continuation_length: int,
    expected_examples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = load_json(metadata_path)
    digest = sha256_file(dataset_path)
    if digest != expected_digest or digest != metadata.get("dataset_sha256"):
        raise ValueError("Dataset checksum mismatch.")
    if int(metadata["prefix_length"]) != prefix_length:
        raise ValueError("Dataset prefix length mismatch.")
    if int(metadata["continuation_length"]) != continuation_length:
        raise ValueError("Dataset continuation length mismatch.")

    rows = read_jsonl(dataset_path)
    if len(rows) != expected_examples:
        raise ValueError("Dataset record count mismatch.")
    if [int(row["example_id"]) for row in rows] != list(
        range(expected_examples)
    ):
        raise ValueError("Dataset example IDs are not contiguous.")
    document_ids = [str(row["document_id"]) for row in rows]
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Dataset document IDs are not unique.")
    for row in rows:
        if len(row["prefix_token_ids"]) != prefix_length:
            raise ValueError("Dataset contains a wrong-length prefix.")
        if (
            len(row["human_continuation_token_ids"])
            != continuation_length
        ):
            raise ValueError("Dataset contains a wrong-length continuation.")
    return rows, metadata


def validate_timings(
    path: Path,
    *,
    k_values: Sequence[int],
    timing_repeats: int,
    expected_examples: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    expected_pairs = {
        (k, repeat)
        for k in k_values
        for repeat in range(timing_repeats)
    }
    actual_pairs = {
        (int(row["k"]), int(row["timing_repeat"])) for row in rows
    }
    if actual_pairs != expected_pairs or len(rows) != len(expected_pairs):
        raise ValueError("Timing records do not form the expected k/repeat grid.")
    for row in rows:
        if int(row["completed_samples"]) != expected_examples:
            raise ValueError("A timing record has the wrong sample count.")
        if int(row["batch_size"]) != batch_size:
            raise ValueError("A timing record has the wrong batch size.")
        elapsed = float(row["elapsed_seconds"])
        amortized = float(row["amortized_seconds_per_sample"])
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("A timing elapsed value is invalid.")
        if not math.isclose(
            amortized,
            elapsed / expected_examples,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError("A timing amortized value is inconsistent.")
    if len({str(row["hardware"]) for row in rows}) != 1:
        raise ValueError("Hardware differs across timing records.")
    if len({str(row["precision"]) for row in rows}) != 1:
        raise ValueError("Precision differs across timing records.")
    return rows


def validate_generations(
    run_dir: Path,
    generation_artifacts: dict[str, Any],
    *,
    k_values: Sequence[int],
    expected_examples: int,
    continuation_length: int,
    mask_token_id: int,
) -> None:
    expected_ids = list(range(expected_examples))
    for k in k_values:
        relative = generation_artifacts.get(str(k))
        if not relative:
            raise ValueError(f"Metadata has no generation artifact for k={k}.")
        path = resolve_path(relative, base=run_dir)
        rows = read_jsonl(path)
        if len(rows) != expected_examples:
            raise ValueError(f"Generation count mismatch for k={k}.")
        if [int(row["example_id"]) for row in rows] != expected_ids:
            raise ValueError(f"Generation IDs are wrong for k={k}.")
        for row in rows:
            if int(row["k"]) != k:
                raise ValueError(f"Generation row has wrong k in {path}.")
            token_ids = [int(value) for value in row["generated_token_ids"]]
            if len(token_ids) != continuation_length:
                raise ValueError(f"Wrong generation length for k={k}.")
            if mask_token_id in token_ids:
                raise ValueError(f"MASK remains in a generation for k={k}.")
            if not isinstance(row.get("generated_text"), str):
                raise ValueError(f"Missing generated text for k={k}.")


def validate_trajectories(
    path: Path,
    *,
    k_values: Sequence[int],
    trajectory_ids: Sequence[int],
    prefix_length: int,
    continuation_length: int,
    mask_token_id: int,
) -> None:
    rows = read_jsonl(path)
    expected_keys = {
        (k, example_id, step)
        for k in k_values
        for example_id in trajectory_ids
        for step in range(math.ceil(continuation_length / k))
    }
    actual_keys = {
        (
            int(row["k"]),
            int(row["example_id"]),
            int(row["step_index"]),
        )
        for row in rows
    }
    if actual_keys != expected_keys or len(rows) != len(expected_keys):
        raise ValueError("Trajectory rows do not form the expected grid.")

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (int(row["k"]), int(row["example_id"])),
            [],
        ).append(row)

    for (k, _), group in grouped.items():
        group.sort(key=lambda row: int(row["step_index"]))
        expected_before = continuation_length
        committed_positions: set[int] = set()
        for row in group:
            before = int(row["remaining_masks_before"])
            after = int(row["remaining_masks_after"])
            expected_commit_count = min(k, before)
            canvas_positions = [
                int(value) for value in row["selected_canvas_positions"]
            ]
            sequence_positions = [
                int(value) for value in row["selected_sequence_positions"]
            ]
            token_ids = [int(value) for value in row["selected_token_ids"]]
            token_text = [str(value) for value in row["selected_token_text"]]
            confidences = [
                float(value) for value in row["selected_confidences"]
            ]
            if before != expected_before:
                raise ValueError("Trajectory remaining-mask progression is wrong.")
            if after != before - expected_commit_count:
                raise ValueError("Trajectory commitment count is wrong.")
            if not (
                len(canvas_positions)
                == len(sequence_positions)
                == len(token_ids)
                == len(token_text)
                == len(confidences)
                == expected_commit_count
            ):
                raise ValueError("Trajectory selected fields have wrong lengths.")
            if len(set(canvas_positions)) != len(canvas_positions):
                raise ValueError("A trajectory step repeats a canvas position.")
            if committed_positions.intersection(canvas_positions):
                raise ValueError("A trajectory recommits a prior position.")
            if any(
                position < 0 or position >= continuation_length
                for position in canvas_positions
            ):
                raise ValueError("A trajectory canvas position is out of range.")
            expected_sequence = [
                1 + prefix_length + position for position in canvas_positions
            ]
            if sequence_positions != expected_sequence:
                raise ValueError("Canvas and sequence positions disagree.")
            if mask_token_id in token_ids:
                raise ValueError("A trajectory committed the MASK token.")
            if any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in confidences
            ):
                raise ValueError("A trajectory confidence is invalid.")
            committed_positions.update(canvas_positions)
            expected_before = after
        if expected_before != 0 or len(committed_positions) != continuation_length:
            raise ValueError("A trajectory did not fill its complete canvas.")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    metadata_path = run_dir / "metadata.json"
    run_metadata = load_json(metadata_path)
    if run_metadata.get("status") != "complete":
        raise ValueError("Run metadata status is not complete.")

    experiment = run_metadata["experiment"]
    prefix_length = int(experiment["prefix_length"])
    continuation_length = int(experiment["continuation_length"])
    k_values = [int(k) for k in experiment["k_values"]]
    if args.prefix_length is not None and args.prefix_length != prefix_length:
        raise ValueError("--prefix-length disagrees with run metadata.")
    if (
        args.continuation_length is not None
        and args.continuation_length != continuation_length
    ):
        raise ValueError("--continuation-length disagrees with run metadata.")
    if args.k_values is not None and args.k_values != k_values:
        raise ValueError("--k-values disagrees with run metadata.")

    dataset_info = run_metadata["dataset"]
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

    artifacts = run_metadata["artifacts"]
    timing_rows = validate_timings(
        resolve_path(artifacts["timings"], base=run_dir),
        k_values=k_values,
        timing_repeats=int(experiment["timing_repeats"]),
        expected_examples=expected_examples,
        batch_size=int(experiment["batch_size"]),
    )
    mask_token_id = int(dataset_metadata["tokenizer"]["mask_token_id"])
    validate_generations(
        run_dir,
        artifacts["generations"],
        k_values=k_values,
        expected_examples=expected_examples,
        continuation_length=continuation_length,
        mask_token_id=mask_token_id,
    )
    trajectory_ids = [int(value) for value in experiment["trajectory_example_ids"]]
    validate_trajectories(
        resolve_path(artifacts["trajectories"], base=run_dir),
        k_values=k_values,
        trajectory_ids=trajectory_ids,
        prefix_length=prefix_length,
        continuation_length=continuation_length,
        mask_token_id=mask_token_id,
    )
    checks = run_metadata.get("determinism_checks", {})
    if any(checks.get(str(k)) is not True for k in k_values):
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
        "k_values": k_values,
        "timing_records": len(timing_rows),
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
    print(f"[verified] audit: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
