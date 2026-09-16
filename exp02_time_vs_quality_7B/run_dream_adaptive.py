#!/usr/bin/env python3
"""Generate and time Dream with the short-circuit adaptive scheduler."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from run_dream_topk import (
    atomic_write_json,
    atomic_write_jsonl,
    append_jsonl,
    decode_text,
    default_dataset_metadata_path,
    hardware_description,
    load_and_validate_dataset,
    package_version,
    prepare_output_directory,
    resolve_torch_dtype,
    synchronize,
    utc_now,
)


DEFAULT_MODEL = "Dream-org/Dream-v0-Base-7B"
DEFAULT_MULTIPLIERS = (1.01, 1.10, 1.25)


def threshold_label(multiplier: float) -> str:
    return f"tau_log_{multiplier:.2f}".replace(".", "p")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-metadata", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix-length", type=int, default=64)
    parser.add_argument("--continuation-length", type=int, default=128)
    parser.add_argument(
        "--tau-multipliers",
        type=float,
        nargs="+",
        default=list(DEFAULT_MULTIPLIERS),
        help="Natural-log thresholds are computed as tau=log(multiplier).",
    )
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--conditional-microbatch-size", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--warmup-examples", type=int, default=2)
    parser.add_argument(
        "--trajectory-example-ids",
        type=int,
        nargs="*",
        default=list(range(10)),
    )
    parser.add_argument("--determinism-check-examples", type=int, default=8)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-special-tokens", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.prefix_length < 1 or args.continuation_length < 1:
        parser.error("Prefix and continuation lengths must be positive.")
    if args.batch_size != 1:
        parser.error("Adaptive scheduler version one requires --batch-size 1.")
    if args.conditional_microbatch_size != 1:
        parser.error(
            "Version one requires --conditional-microbatch-size 1."
        )
    if args.max_candidates < 1:
        parser.error("--max-candidates must be positive.")
    if args.timing_repeats < 1:
        parser.error("--timing-repeats must be positive.")
    if args.warmup_examples < 0:
        parser.error("--warmup-examples cannot be negative.")
    if args.determinism_check_examples < 0:
        parser.error("--determinism-check-examples cannot be negative.")
    if (
        not args.tau_multipliers
        or any(not math.isfinite(x) or x <= 1.0 for x in args.tau_multipliers)
    ):
        parser.error("--tau-multipliers must be finite values greater than 1.")
    labels = [threshold_label(value) for value in args.tau_multipliers]
    if len(set(labels)) != len(labels):
        parser.error("Threshold multipliers produce duplicate labels.")
    if len(set(args.trajectory_example_ids)) != len(
        args.trajectory_example_ids
    ):
        parser.error("Trajectory example IDs must not contain duplicates.")
    return args


def make_states(
    adapter: Any,
    torch: Any,
    records: Sequence[dict[str, Any]],
    *,
    continuation_length: int,
) -> list[Any]:
    return [
        adapter.prepare_prefix_batch(
            torch.tensor(
                [record["prefix_token_ids"]],
                dtype=torch.long,
            ),
            canvas_length=continuation_length,
        )
        for record in records
    ]


def add_histograms(left: list[int], right: Sequence[int]) -> None:
    if len(left) != len(right):
        raise ValueError("Histogram lengths disagree.")
    for index, value in enumerate(right):
        left[index] += int(value)


def run_generation_pass(
    adapter: Any,
    torch: Any,
    device: Any,
    states: Sequence[Any],
    config: Any,
    *,
    timing_repeat: int,
    retain_outputs: bool,
) -> tuple[dict[str, Any], Any | None]:
    from adaptive_scheduler import generate_adaptive

    device_outputs: list[Any] = []
    decision_stats: list[tuple[int, int, tuple[int, ...]]] = []

    synchronize(torch, device)
    started = time.perf_counter()
    for state in states:
        result = generate_adaptive(adapter, state, config=config)
        device_outputs.append(result.continuation_token_ids)
        decision_stats.append(
            (
                result.scheduler_steps,
                result.conditional_probes,
                result.accepted_prefix_histogram,
            )
        )
    synchronize(torch, device)
    elapsed = time.perf_counter() - started

    histogram = [0] * config.max_candidates
    scheduler_steps = 0
    conditional_probes = 0
    for steps, probes, accepted_histogram in decision_stats:
        scheduler_steps += steps
        conditional_probes += probes
        add_histograms(histogram, accepted_histogram)

    canonical = None
    if retain_outputs:
        canonical = torch.cat(
            [value.detach().cpu() for value in device_outputs],
            dim=0,
        )
    completed = len(states)
    timing = {
        "schedule_type": "adaptive",
        "threshold_label": threshold_label(math.exp(config.tau)),
        "tau": config.tau,
        "tau_multiplier": math.exp(config.tau),
        "timing_repeat": timing_repeat,
        "completed_samples": completed,
        "elapsed_seconds": elapsed,
        "amortized_seconds_per_sample": elapsed / completed,
        "total_scheduler_steps": scheduler_steps,
        "total_conditional_probes": conditional_probes,
        "accepted_prefix_histogram": histogram,
        "mean_accepted_prefix_length": (
            completed
            * states[0].canvas_length
            / scheduler_steps
        ),
    }
    return timing, canonical


def save_generations(
    path: Path,
    tokenizer: Any,
    records: Sequence[dict[str, Any]],
    generated_ids: Any,
    threshold: dict[str, Any],
    *,
    continuation_length: int,
    mask_token_id: int,
) -> None:
    rows = generated_ids.tolist()
    output: list[dict[str, Any]] = []
    for record, token_ids in zip(records, rows):
        if len(token_ids) != continuation_length:
            raise RuntimeError("A generated continuation has the wrong length.")
        if mask_token_id in token_ids:
            raise RuntimeError("A generated continuation still contains MASK.")
        output.append(
            {
                "schedule_type": "adaptive",
                "threshold_label": threshold["label"],
                "tau_multiplier": threshold["multiplier"],
                "tau": threshold["tau"],
                "example_id": int(record["example_id"]),
                "generated_token_ids": [int(value) for value in token_ids],
                "generated_text": decode_text(tokenizer, token_ids),
            }
        )
    if len(output) != len(records):
        raise RuntimeError("Generated row count does not match the dataset.")
    atomic_write_jsonl(path, output)


def generate_subset(
    adapter: Any,
    torch: Any,
    records: Sequence[dict[str, Any]],
    config: Any,
    *,
    count: int,
    continuation_length: int,
) -> Any:
    from adaptive_scheduler import generate_adaptive

    outputs = []
    for record in records[:count]:
        state = adapter.prepare_prefix_batch(
            torch.tensor([record["prefix_token_ids"]], dtype=torch.long),
            canvas_length=continuation_length,
        )
        outputs.append(
            generate_adaptive(
                adapter,
                state,
                config=config,
            ).continuation_token_ids.detach().cpu()
        )
    if not outputs:
        return torch.empty((0, continuation_length), dtype=torch.long)
    return torch.cat(outputs, dim=0)


def collect_trajectories(
    adapter: Any,
    torch: Any,
    records: Sequence[dict[str, Any]],
    trajectory_ids: Sequence[int],
    thresholds: Sequence[dict[str, Any]],
    *,
    max_candidates: int,
    conditional_microbatch_size: int,
    continuation_length: int,
) -> list[dict[str, Any]]:
    from adaptive_scheduler import AdaptiveSchedulerConfig, adaptive_step

    by_id = {int(row["example_id"]): row for row in records}
    missing = [value for value in trajectory_ids if value not in by_id]
    if missing:
        raise ValueError(f"Unknown trajectory example IDs: {missing}")
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        config = AdaptiveSchedulerConfig(
            tau=threshold["tau"],
            max_candidates=max_candidates,
            conditional_microbatch_size=conditional_microbatch_size,
        )
        for example_id in trajectory_ids:
            state = adapter.prepare_prefix_batch(
                torch.tensor(
                    [by_id[example_id]["prefix_token_ids"]],
                    dtype=torch.long,
                ),
                canvas_length=continuation_length,
            )
            remaining = continuation_length
            step_index = 0
            while remaining:
                result = adaptive_step(
                    adapter,
                    state,
                    config=config,
                    remaining_count=remaining,
                )
                canvas = result.candidate_canvas_positions.detach().cpu().tolist()
                sequence = (
                    result.candidate_sequence_positions.detach().cpu().tolist()
                )
                tokens = result.candidate_token_ids.detach().cpu().tolist()
                confidence = (
                    result.candidate_confidences.detach().cpu().tolist()
                )
                accepted = result.accepted_prefix_length
                rows.append(
                    {
                        "schedule_type": "adaptive",
                        "threshold_label": threshold["label"],
                        "tau_multiplier": threshold["multiplier"],
                        "tau": threshold["tau"],
                        "example_id": example_id,
                        "step_index": step_index,
                        "remaining_masks_before": result.remaining_masks_before,
                        "candidate_canvas_positions": [
                            int(value) for value in canvas
                        ],
                        "candidate_sequence_positions": [
                            int(value) for value in sequence
                        ],
                        "candidate_token_ids": [
                            int(value) for value in tokens
                        ],
                        "candidate_token_text": [
                            decode_text(adapter.tokenizer, [value])
                            for value in tokens
                        ],
                        "candidate_confidences": [
                            float(value) for value in confidence
                        ],
                        "marginal_log_factors": list(
                            result.marginal_log_factors
                        ),
                        "conditional_log_factors": list(
                            result.conditional_log_factors
                        ),
                        "cumulative_log_marginals": list(
                            result.cumulative_log_marginals
                        ),
                        "cumulative_log_joints": list(
                            result.cumulative_log_joints
                        ),
                        "cumulative_errors": list(result.cumulative_errors),
                        "conditional_probes": result.conditional_probes,
                        "accepted_prefix_length": accepted,
                        "committed_canvas_positions": [
                            int(value) for value in canvas[:accepted]
                        ],
                        "committed_sequence_positions": [
                            int(value) for value in sequence[:accepted]
                        ],
                        "committed_token_ids": [
                            int(value) for value in tokens[:accepted]
                        ],
                        "remaining_masks_after": result.remaining_masks_after,
                    }
                )
                state = result.state
                remaining = result.remaining_masks_after
                step_index += 1
    return rows


def run(args: argparse.Namespace) -> None:
    try:
        import torch
        from adaptive_scheduler import AdaptiveSchedulerConfig, generate_adaptive
        from dream_adapter import DreamAdapter, ModelLoadOptions
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and dream_adapter.py beside this script are required."
        ) from exc

    dataset = args.dataset.expanduser().resolve()
    dataset_metadata = (
        args.dataset_metadata.expanduser().resolve()
        if args.dataset_metadata
        else default_dataset_metadata_path(dataset)
    )
    output_dir = args.output_dir.expanduser().resolve()
    records, _, dataset_digest = load_and_validate_dataset(
        dataset,
        dataset_metadata,
        prefix_length=args.prefix_length,
        continuation_length=args.continuation_length,
    )
    prepare_output_directory(output_dir, overwrite=args.overwrite)

    thresholds = [
        {
            "label": threshold_label(multiplier),
            "multiplier": multiplier,
            "tau": math.log(multiplier),
        }
        for multiplier in args.tau_multipliers
    ]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for requested device {device}.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    metadata_path = output_dir / "metadata.json"
    timing_path = output_dir / "timings.jsonl"
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": utc_now(),
        "dataset": {
            "path": str(dataset),
            "metadata_path": str(dataset_metadata),
            "sha256": dataset_digest,
            "num_examples": len(records),
        },
        "experiment": {
            "schedule_type": "adaptive",
            "policy": "short_circuit_frozen_proposals",
            "prefix_length": args.prefix_length,
            "continuation_length": args.continuation_length,
            "model_input_length": 1
            + args.prefix_length
            + args.continuation_length,
            "thresholds": thresholds,
            "max_candidates": args.max_candidates,
            "batch_size": args.batch_size,
            "conditional_microbatch_size": (
                args.conditional_microbatch_size
            ),
            "timing_repeats": args.timing_repeats,
            "warmup_examples": args.warmup_examples,
            "trajectory_example_ids": args.trajectory_example_ids,
            "determinism_check_examples": args.determinism_check_examples,
            "seed": args.seed,
            "timing_scope": {
                "included": [
                    "base model forwards",
                    "executed conditional model forwards",
                    "scheduler decisions and device synchronization",
                ],
                "excluded": [
                    "model loading",
                    "prefix-state construction",
                    "warm-up",
                    "device-to-host generation copy",
                    "decoding",
                    "file output",
                    "trajectory collection",
                ],
            },
            "numeric_policy": {
                "model_dtype": args.dtype,
                "log_probability_dtype": "float32",
                "cumulative_sum_and_threshold_dtype": "float64",
            },
        },
        "model": {
            "model_id": args.model,
            "revision": args.model_revision,
            "dtype": args.dtype,
            "device": str(device),
            "filter_special_tokens": not args.allow_special_tokens,
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
        },
        "execution": {
            "physical_gpu_id": os.environ.get("EXPERIMENT_PHYSICAL_GPU_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    atomic_write_json(metadata_path, metadata)

    try:
        adapter = DreamAdapter.from_pretrained(
            ModelLoadOptions(
                model_id=args.model,
                revision=args.model_revision,
                cache_dir=args.cache_dir,
                local_files_only=args.local_files_only,
                dtype=resolve_torch_dtype(torch, args.dtype),
                device=device,
            ),
            filter_special_tokens=not args.allow_special_tokens,
        )
        hardware = hardware_description(torch, device)
        metadata["hardware"] = hardware
        metadata["model"]["mask_token_id"] = adapter.mask_token_id
        atomic_write_json(metadata_path, metadata)
        states = make_states(
            adapter,
            torch,
            records,
            continuation_length=args.continuation_length,
        )

        determinism: dict[str, bool] = {}
        generation_artifacts: dict[str, str] = {}
        for threshold in thresholds:
            label = threshold["label"]
            config = AdaptiveSchedulerConfig(
                tau=threshold["tau"],
                max_candidates=args.max_candidates,
                conditional_microbatch_size=args.conditional_microbatch_size,
            )
            print(f"[{label}] warm-up", flush=True)
            for state in states[: args.warmup_examples]:
                generate_adaptive(adapter, state, config=config)
            synchronize(torch, device)

            canonical = None
            for repeat in range(args.timing_repeats):
                timing, generated = run_generation_pass(
                    adapter,
                    torch,
                    device,
                    states,
                    config,
                    timing_repeat=repeat,
                    retain_outputs=repeat == 0,
                )
                timing.update(
                    {
                        "threshold_label": label,
                        "tau_multiplier": threshold["multiplier"],
                        "tau": threshold["tau"],
                        "batch_size": args.batch_size,
                        "conditional_microbatch_size": (
                            args.conditional_microbatch_size
                        ),
                        "max_candidates": args.max_candidates,
                        "hardware": hardware,
                        "precision": args.dtype,
                    }
                )
                append_jsonl(timing_path, timing)
                print(
                    f"[{label} repeat={repeat}] "
                    f"{timing['elapsed_seconds']:.3f}s; "
                    f"{timing['total_conditional_probes']} probes",
                    flush=True,
                )
                if generated is not None:
                    canonical = generated
            if canonical is None:
                raise RuntimeError("Canonical generation corpus was not retained.")

            relative = f"generations/{label}.jsonl"
            generation_artifacts[label] = relative
            save_generations(
                output_dir / relative,
                adapter.tokenizer,
                records,
                canonical,
                threshold,
                continuation_length=args.continuation_length,
                mask_token_id=adapter.mask_token_id,
            )
            count = min(args.determinism_check_examples, len(records))
            repeated = generate_subset(
                adapter,
                torch,
                records,
                config,
                count=count,
                continuation_length=args.continuation_length,
            )
            determinism[label] = bool(torch.equal(repeated, canonical[:count]))
            if not determinism[label]:
                raise RuntimeError(f"Determinism check failed for {label}.")

        print("[trajectories] collecting untimed traces", flush=True)
        trajectories = collect_trajectories(
            adapter,
            torch,
            records,
            args.trajectory_example_ids,
            thresholds,
            max_candidates=args.max_candidates,
            conditional_microbatch_size=args.conditional_microbatch_size,
            continuation_length=args.continuation_length,
        )
        atomic_write_jsonl(output_dir / "trajectories.jsonl", trajectories)

        metadata["status"] = "complete"
        metadata["completed_at_utc"] = utc_now()
        metadata["determinism_checks"] = determinism
        metadata["artifacts"] = {
            "timings": "timings.jsonl",
            "trajectories": "trajectories.jsonl",
            "generations": generation_artifacts,
        }
        atomic_write_json(metadata_path, metadata)
        print(f"[done] adaptive run: {output_dir}", flush=True)
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["failed_at_utc"] = utc_now()
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(metadata_path, metadata)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
