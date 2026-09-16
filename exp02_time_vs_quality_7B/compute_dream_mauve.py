#!/usr/bin/env python3
"""Compute one fixed-setting corpus-level MAUVE score per Dream schedule."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--prefix-length", type=int)
    parser.add_argument("--continuation-length", type=int)
    parser.add_argument("--k-values", type=int, nargs="+")
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

    for name in (
        "prefix_length",
        "continuation_length",
    ):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1.")
    if args.k_values is not None:
        if any(k < 1 for k in args.k_values):
            parser.error("--k-values must contain positive integers.")
        if len(set(args.k_values)) != len(args.k_values):
            parser.error("--k-values must not contain duplicates.")
    if args.device_id < -1:
        parser.error("--device-id must be -1 for CPU or a nonnegative GPU index.")
    if args.max_text_length < 1:
        parser.error("--max-text-length must be at least 1.")
    if args.num_buckets < 0:
        parser.error("--num-buckets cannot be negative.")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.kmeans_num_redo < 1 or args.kmeans_max_iter < 1:
        parser.error("K-means iteration settings must be positive.")
    if args.mauve_scaling_factor <= 0:
        parser.error("--mauve-scaling-factor must be positive.")
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive.")
    return args


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}.")
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


def atomic_write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
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
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def validate_configuration(
    args: argparse.Namespace,
    run_metadata: dict[str, Any],
) -> tuple[int, int, list[int], int]:
    experiment = run_metadata["experiment"]
    prefix_length = int(experiment["prefix_length"])
    continuation_length = int(experiment["continuation_length"])
    k_values = [int(value) for value in experiment["k_values"]]
    num_examples = int(run_metadata["dataset"]["num_examples"])
    if args.prefix_length is not None and args.prefix_length != prefix_length:
        raise ValueError("--prefix-length disagrees with run metadata.")
    if (
        args.continuation_length is not None
        and args.continuation_length != continuation_length
    ):
        raise ValueError("--continuation-length disagrees with run metadata.")
    if args.k_values is not None and args.k_values != k_values:
        raise ValueError("--k-values disagrees with run metadata.")
    return prefix_length, continuation_length, k_values, num_examples


def load_text_corpora(
    run_dir: Path,
    run_metadata: dict[str, Any],
    *,
    dataset_override: Path | None,
    k_values: Sequence[int],
    expected_examples: int,
) -> tuple[list[str], dict[int, list[str]]]:
    dataset_path = (
        dataset_override.expanduser().resolve()
        if dataset_override
        else Path(run_metadata["dataset"]["path"]).expanduser().resolve()
    )
    dataset_rows = read_jsonl(dataset_path)
    if len(dataset_rows) != expected_examples:
        raise ValueError("Human corpus count is wrong.")
    human_text = [str(row["human_continuation_text"]) for row in dataset_rows]

    generation_artifacts = run_metadata["artifacts"]["generations"]
    generated: dict[int, list[str]] = {}
    for k in k_values:
        path = (run_dir / generation_artifacts[str(k)]).resolve()
        rows = read_jsonl(path)
        if len(rows) != expected_examples:
            raise ValueError(f"Generated corpus count is wrong for k={k}.")
        if [int(row["example_id"]) for row in rows] != list(
            range(expected_examples)
        ):
            raise ValueError(f"Generated corpus IDs are wrong for k={k}.")
        generated[k] = [str(row["generated_text"]) for row in rows]
    return human_text, generated


def serializable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return float(value)


def compute_one(
    mauve: Any,
    *,
    human_text: Sequence[str],
    generated_text: Sequence[str],
    settings: dict[str, Any],
    verbose: bool,
) -> Any:
    signature = inspect.signature(mauve.compute_mauve)
    requested = {
        "p_text": list(human_text),
        "q_text": list(generated_text),
        "device_id": settings["device_id"],
        "featurize_model_name": settings["featurize_model_name"],
        "max_text_length": settings["max_text_length"],
        "num_buckets": settings["num_buckets"],
        "batch_size": settings["batch_size"],
        "seed": settings["seed"],
        "kmeans_num_redo": settings["kmeans_num_redo"],
        "kmeans_max_iter": settings["kmeans_max_iter"],
        "mauve_scaling_factor": settings["mauve_scaling_factor"],
        "verbose": verbose,
    }
    required_settings = {
        "p_text",
        "q_text",
        "device_id",
        "featurize_model_name",
        "max_text_length",
        "num_buckets",
        "seed",
    }
    missing = required_settings - set(signature.parameters)
    if missing:
        raise RuntimeError(
            "Installed mauve.compute_mauve lacks required parameters: "
            f"{sorted(missing)}"
        )
    supported = {
        name: value
        for name, value in requested.items()
        if name in signature.parameters
    }
    return mauve.compute_mauve(**supported)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    run_metadata = load_json(run_dir / "metadata.json")
    audit = load_json(run_dir / "audit.json")
    if run_metadata.get("status") != "complete":
        raise ValueError("Run status is not complete.")
    if audit.get("status") != "verified":
        raise ValueError("Run audit has not passed.")

    prefix_length, continuation_length, k_values, num_examples = (
        validate_configuration(args, run_metadata)
    )
    if str(audit.get("dataset_sha256")) != str(
        run_metadata["dataset"]["sha256"]
    ):
        raise ValueError("Audit and run metadata dataset checksums disagree.")

    output = (
        args.output.expanduser().resolve()
        if args.output
        else run_dir / "mauve.jsonl"
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

    completed = {int(row["k"]) for row in existing}
    if len(completed) != len(existing) or not completed.issubset(set(k_values)):
        raise ValueError("Existing MAUVE output has duplicate or unknown k values.")

    try:
        import mauve
    except ImportError as exc:
        raise RuntimeError("Install mauve-text before computing MAUVE.") from exc

    human_text, generated = load_text_corpora(
        run_dir,
        run_metadata,
        dataset_override=args.dataset,
        k_values=k_values,
        expected_examples=num_examples,
    )
    num_buckets = args.num_buckets or min(500, max(2, num_examples // 10))
    settings = {
        "device_id": args.device_id,
        "featurize_model_name": args.featurize_model_name,
        "max_text_length": args.max_text_length,
        "num_buckets": num_buckets,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "kmeans_num_redo": args.kmeans_num_redo,
        "kmeans_max_iter": args.kmeans_max_iter,
        "mauve_scaling_factor": args.mauve_scaling_factor,
    }
    for row in existing:
        if row.get("settings") != settings:
            raise ValueError(
                "Existing MAUVE settings differ from the requested settings."
            )
        if str(row.get("dataset_sha256")) != str(
            run_metadata["dataset"]["sha256"]
        ):
            raise ValueError("Existing MAUVE dataset checksum is wrong.")
        if int(row.get("num_examples", -1)) != num_examples:
            raise ValueError("Existing MAUVE sample count is wrong.")
        if int(row.get("prefix_length", -1)) != prefix_length:
            raise ValueError("Existing MAUVE prefix length is wrong.")
        if int(row.get("continuation_length", -1)) != continuation_length:
            raise ValueError("Existing MAUVE continuation length is wrong.")

    results = list(existing)
    for k in k_values:
        if k in completed:
            print(f"[k={k}] already present; skipping", flush=True)
            continue
        print(f"[k={k}] computing MAUVE", flush=True)
        result = compute_one(
            mauve,
            human_text=human_text,
            generated_text=generated[k],
            settings=settings,
            verbose=args.verbose,
        )
        row = {
            "k": k,
            "num_examples": num_examples,
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
            "dataset_sha256": run_metadata["dataset"]["sha256"],
            "mauve_text_version": package_version("mauve-text"),
            "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if not math.isfinite(row["mauve"]) or not 0.0 <= row["mauve"] <= 1.0:
            raise RuntimeError(f"MAUVE returned an invalid score for k={k}.")
        results.append(row)
        results.sort(key=lambda value: int(value["k"]))
        atomic_write_jsonl(output, results)
        print(f"[k={k}] MAUVE={row['mauve']:.6f}", flush=True)

    if {int(row["k"]) for row in results} != set(k_values):
        raise RuntimeError("MAUVE output is incomplete.")
    print(f"[done] MAUVE results: {output}", flush=True)
    if args.hard_exit_after_success:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
