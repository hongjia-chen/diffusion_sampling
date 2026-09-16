#!/usr/bin/env python3
"""Merge independently timed one-schedule runs into auditable combined runs."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from audit_dream_topk import load_json, read_jsonl, resolve_path
from run_dream_topk import atomic_write_json, atomic_write_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--adaptive-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def prepare_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Merge output already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    (path / "generations").mkdir()


def load_verified_run(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    metadata = load_json(path / "metadata.json")
    audit = load_json(path / "audit.json")
    if metadata.get("status") != "complete":
        raise ValueError(f"Run is not complete: {path}")
    if audit.get("status") != "verified":
        raise ValueError(f"Run is not verified: {path}")
    return {"path": path, "metadata": metadata, "audit": audit}


def validate_shared(runs: Sequence[dict[str, Any]]) -> None:
    first = runs[0]["metadata"]
    for run in runs[1:]:
        metadata = run["metadata"]
        if metadata["dataset"]["sha256"] != first["dataset"]["sha256"]:
            raise ValueError("Schedule runs used different datasets.")
        for field in ("prefix_length", "continuation_length"):
            if metadata["experiment"][field] != first["experiment"][field]:
                raise ValueError(f"Schedule runs disagree on {field}.")
        for field in ("model_id", "revision", "dtype", "filter_special_tokens"):
            if metadata["model"].get(field) != first["model"].get(field):
                raise ValueError(f"Schedule runs disagree on model {field}.")
        if metadata.get("hardware") != first.get("hardware"):
            raise ValueError("Schedule runs used different GPU models.")


def source_record(run: dict[str, Any]) -> dict[str, Any]:
    metadata = run["metadata"]
    return {
        "run_dir": str(run["path"]),
        "physical_gpu_id": metadata.get("execution", {}).get(
            "physical_gpu_id"
        ),
        "cuda_visible_devices": metadata.get("execution", {}).get(
            "cuda_visible_devices"
        ),
    }


def merge_fixed(
    runs: Sequence[dict[str, Any]],
    output: Path,
    *,
    overwrite: bool,
) -> None:
    prepare_directory(output, overwrite)
    by_k: dict[int, dict[str, Any]] = {}
    for run in runs:
        values = [int(value) for value in run["metadata"]["experiment"]["k_values"]]
        if len(values) != 1:
            raise ValueError("Each fixed worker run must contain exactly one k.")
        k = values[0]
        if k in by_k:
            raise ValueError(f"Duplicate fixed schedule k={k}.")
        by_k[k] = run
    k_values = sorted(by_k)
    first = by_k[k_values[0]]["metadata"]
    timing_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    mauve_rows: list[dict[str, Any]] = []
    determinism: dict[str, bool] = {}
    generations: dict[str, str] = {}
    sources: dict[str, dict[str, Any]] = {}
    for k in k_values:
        run = by_k[k]
        metadata = run["metadata"]
        base = run["path"]
        timing_rows.extend(
            read_jsonl(resolve_path(metadata["artifacts"]["timings"], base=base))
        )
        trajectory_rows.extend(
            read_jsonl(
                resolve_path(metadata["artifacts"]["trajectories"], base=base)
            )
        )
        mauve_rows.extend(read_jsonl(base / "mauve.jsonl"))
        determinism[str(k)] = bool(metadata["determinism_checks"][str(k)])
        relative = f"generations/k_{k}.jsonl"
        source_generation = resolve_path(
            metadata["artifacts"]["generations"][str(k)],
            base=base,
        )
        shutil.copy2(source_generation, output / relative)
        generations[str(k)] = relative
        sources[str(k)] = source_record(run)

    atomic_write_jsonl(output / "timings.jsonl", timing_rows)
    atomic_write_jsonl(output / "trajectories.jsonl", trajectory_rows)
    atomic_write_jsonl(output / "mauve.jsonl", mauve_rows)
    experiment = dict(first["experiment"])
    experiment["k_values"] = k_values
    metadata = dict(first)
    metadata["started_at_utc"] = min(
        run["metadata"]["started_at_utc"] for run in runs
    )
    metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["experiment"] = experiment
    metadata["determinism_checks"] = determinism
    metadata["artifacts"] = {
        "timings": "timings.jsonl",
        "trajectories": "trajectories.jsonl",
        "generations": generations,
    }
    metadata["distributed_execution"] = {
        "parallelization": "one_complete_schedule_per_gpu",
        "schedule_sources": sources,
    }
    metadata["execution"] = {
        "mode": "four_gpu_schedule_parallel",
        "timing_metric": "per_schedule_single_gpu_elapsed_seconds_per_sample",
    }
    atomic_write_json(output / "metadata.json", metadata)


def merge_adaptive(
    runs: Sequence[dict[str, Any]],
    output: Path,
    *,
    overwrite: bool,
) -> None:
    prepare_directory(output, overwrite)
    by_label: dict[str, dict[str, Any]] = {}
    for run in runs:
        thresholds = run["metadata"]["experiment"]["thresholds"]
        if len(thresholds) != 1:
            raise ValueError(
                "Each adaptive worker run must contain exactly one threshold."
            )
        label = str(thresholds[0]["label"])
        if label in by_label:
            raise ValueError(f"Duplicate adaptive threshold {label}.")
        by_label[label] = run
    ordered_runs = sorted(
        by_label.values(),
        key=lambda run: float(
            run["metadata"]["experiment"]["thresholds"][0]["multiplier"]
        ),
    )
    first = ordered_runs[0]["metadata"]
    thresholds: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    mauve_rows: list[dict[str, Any]] = []
    determinism: dict[str, bool] = {}
    generations: dict[str, str] = {}
    sources: dict[str, dict[str, Any]] = {}
    for run in ordered_runs:
        metadata = run["metadata"]
        base = run["path"]
        threshold = metadata["experiment"]["thresholds"][0]
        label = str(threshold["label"])
        thresholds.append(threshold)
        timing_rows.extend(
            read_jsonl(resolve_path(metadata["artifacts"]["timings"], base=base))
        )
        trajectory_rows.extend(
            read_jsonl(
                resolve_path(metadata["artifacts"]["trajectories"], base=base)
            )
        )
        mauve_rows.extend(read_jsonl(base / "adaptive_mauve.jsonl"))
        determinism[label] = bool(metadata["determinism_checks"][label])
        relative = f"generations/{label}.jsonl"
        source_generation = resolve_path(
            metadata["artifacts"]["generations"][label],
            base=base,
        )
        shutil.copy2(source_generation, output / relative)
        generations[label] = relative
        sources[label] = source_record(run)

    atomic_write_jsonl(output / "timings.jsonl", timing_rows)
    atomic_write_jsonl(output / "trajectories.jsonl", trajectory_rows)
    atomic_write_jsonl(output / "adaptive_mauve.jsonl", mauve_rows)
    experiment = dict(first["experiment"])
    experiment["thresholds"] = thresholds
    metadata = dict(first)
    metadata["started_at_utc"] = min(
        run["metadata"]["started_at_utc"] for run in runs
    )
    metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["experiment"] = experiment
    metadata["determinism_checks"] = determinism
    metadata["artifacts"] = {
        "timings": "timings.jsonl",
        "trajectories": "trajectories.jsonl",
        "generations": generations,
    }
    metadata["distributed_execution"] = {
        "parallelization": "one_complete_schedule_per_gpu",
        "schedule_sources": sources,
    }
    metadata["execution"] = {
        "mode": "four_gpu_schedule_parallel",
        "timing_metric": "per_schedule_single_gpu_elapsed_seconds_per_sample",
    }
    atomic_write_json(output / "metadata.json", metadata)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    fixed = [load_verified_run(path) for path in args.fixed_runs]
    adaptive = [load_verified_run(path) for path in args.adaptive_runs]
    all_runs = fixed + adaptive
    validate_shared(all_runs)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    merge_fixed(
        fixed,
        output_root / "fixed",
        overwrite=args.overwrite,
    )
    merge_adaptive(
        adaptive,
        output_root / "adaptive",
        overwrite=args.overwrite,
    )
    print(f"[done] merged fixed run:    {output_root / 'fixed'}")
    print(f"[done] merged adaptive run: {output_root / 'adaptive'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
