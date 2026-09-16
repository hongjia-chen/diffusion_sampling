#!/usr/bin/env python3
"""Compute one corpus-level MAUVE score per adaptive threshold."""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from compute_dream_mauve import (
    atomic_write_jsonl,
    compute_one,
    load_json,
    package_version,
    read_jsonl,
    serializable,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--featurize-model-name", default="gpt2-large")
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=0,
        help="Zero uses min(500, floor(N/10)).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--kmeans-num-redo", type=int, default=5)
    parser.add_argument("--kmeans-max-iter", type=int, default=500)
    parser.add_argument("--mauve-scaling-factor", type=float, default=5.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hard-exit-after-success", action="store_true")
    args = parser.parse_args(argv)
    if args.device_id < -1:
        parser.error("--device-id must be -1 or nonnegative.")
    if args.max_text_length < 1 or args.batch_size < 1:
        parser.error("Text length and batch size must be positive.")
    if args.num_buckets < 0:
        parser.error("--num-buckets cannot be negative.")
    if args.kmeans_num_redo < 1 or args.kmeans_max_iter < 1:
        parser.error("K-means iteration settings must be positive.")
    if args.mauve_scaling_factor <= 0:
        parser.error("--mauve-scaling-factor must be positive.")
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive.")
    return args


def load_corpora(
    run_dir: Path,
    metadata: dict[str, Any],
    *,
    dataset_override: Path | None,
    labels: Sequence[str],
    expected_examples: int,
) -> tuple[list[str], dict[str, list[str]]]:
    dataset = (
        dataset_override.expanduser().resolve()
        if dataset_override
        else Path(metadata["dataset"]["path"]).expanduser().resolve()
    )
    human_rows = read_jsonl(dataset)
    if len(human_rows) != expected_examples:
        raise ValueError("Human corpus count is wrong.")
    if [int(row["example_id"]) for row in human_rows] != list(
        range(expected_examples)
    ):
        raise ValueError("Human corpus IDs are wrong.")
    human = [str(row["human_continuation_text"]) for row in human_rows]

    artifacts = metadata["artifacts"]["generations"]
    generated: dict[str, list[str]] = {}
    for label in labels:
        rows = read_jsonl((run_dir / artifacts[label]).resolve())
        if len(rows) != expected_examples:
            raise ValueError(f"Generated corpus count is wrong for {label}.")
        if [int(row["example_id"]) for row in rows] != list(
            range(expected_examples)
        ):
            raise ValueError(f"Generated corpus IDs are wrong for {label}.")
        generated[label] = [str(row["generated_text"]) for row in rows]
    return human, generated


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    metadata = load_json(run_dir / "metadata.json")
    audit = load_json(run_dir / "audit.json")
    if metadata.get("status") != "complete":
        raise ValueError("Adaptive run is not complete.")
    if audit.get("status") != "verified":
        raise ValueError("Adaptive audit has not passed.")
    if metadata["experiment"].get("schedule_type") != "adaptive":
        raise ValueError("Run is not an adaptive-scheduler experiment.")
    if str(audit["dataset_sha256"]) != str(metadata["dataset"]["sha256"]):
        raise ValueError("Audit and run dataset checksums disagree.")

    thresholds = metadata["experiment"]["thresholds"]
    labels = [str(value["label"]) for value in thresholds]
    by_label = {str(value["label"]): value for value in thresholds}
    expected_examples = int(metadata["dataset"]["num_examples"])
    prefix_length = int(metadata["experiment"]["prefix_length"])
    continuation_length = int(metadata["experiment"]["continuation_length"])
    output = (
        args.output.expanduser().resolve()
        if args.output
        else run_dir / "adaptive_mauve.jsonl"
    )
    existing: list[dict[str, Any]] = []
    if output.exists():
        if args.overwrite:
            existing = []
        elif args.resume:
            existing = read_jsonl(output)
        else:
            raise FileExistsError(
                f"Output exists: {output}. Use --resume or --overwrite."
            )
    completed = [str(row["threshold_label"]) for row in existing]
    if len(set(completed)) != len(completed) or not set(completed).issubset(
        set(labels)
    ):
        raise ValueError("Existing output has duplicate or unknown thresholds.")

    try:
        import mauve
    except ImportError as exc:
        raise RuntimeError("Install mauve-text before computing MAUVE.") from exc

    human, generated = load_corpora(
        run_dir,
        metadata,
        dataset_override=args.dataset,
        labels=labels,
        expected_examples=expected_examples,
    )
    settings = {
        "device_id": args.device_id,
        "featurize_model_name": args.featurize_model_name,
        "max_text_length": args.max_text_length,
        "num_buckets": (
            args.num_buckets
            or min(500, max(2, expected_examples // 10))
        ),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "kmeans_num_redo": args.kmeans_num_redo,
        "kmeans_max_iter": args.kmeans_max_iter,
        "mauve_scaling_factor": args.mauve_scaling_factor,
    }
    for row in existing:
        if row.get("settings") != settings:
            raise ValueError("Existing MAUVE settings differ.")
        if str(row.get("dataset_sha256")) != str(
            metadata["dataset"]["sha256"]
        ):
            raise ValueError("Existing MAUVE dataset checksum is wrong.")

    results = list(existing)
    completed_set = set(completed)
    for label in labels:
        if label in completed_set:
            print(f"[{label}] already present; skipping", flush=True)
            continue
        print(f"[{label}] computing MAUVE", flush=True)
        result = compute_one(
            mauve,
            human_text=human,
            generated_text=generated[label],
            settings=settings,
            verbose=args.verbose,
        )
        threshold = by_label[label]
        row = {
            "schedule_type": "adaptive",
            "threshold_label": label,
            "tau_multiplier": float(threshold["multiplier"]),
            "tau": float(threshold["tau"]),
            "num_examples": expected_examples,
            "prefix_length": prefix_length,
            "continuation_length": continuation_length,
            "mauve": float(result.mauve),
            "frontier_integral": float(result.frontier_integral),
            "mauve_star": serializable(getattr(result, "mauve_star", None)),
            "mauve_frontier_integral_star": serializable(
                getattr(result, "mauve_frontier_integral_star", None)
            ),
            "divergence_curve": serializable(result.divergence_curve),
            "p_hist": serializable(result.p_hist),
            "q_hist": serializable(result.q_hist),
            "settings": settings,
            "dataset_sha256": metadata["dataset"]["sha256"],
            "mauve_text_version": package_version("mauve-text"),
            "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if not math.isfinite(row["mauve"]) or not 0.0 <= row["mauve"] <= 1.0:
            raise RuntimeError(f"MAUVE returned an invalid score for {label}.")
        results.append(row)
        order = {value: index for index, value in enumerate(labels)}
        results.sort(key=lambda value: order[str(value["threshold_label"])])
        atomic_write_jsonl(output, results)
        print(f"[{label}] MAUVE={row['mauve']:.6f}", flush=True)

    if {str(row["threshold_label"]) for row in results} != set(labels):
        raise RuntimeError("Adaptive MAUVE output is incomplete.")
    print(f"[done] adaptive MAUVE: {output}", flush=True)
    if args.hard_exit_after_success:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
