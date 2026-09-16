#!/usr/bin/env python3
"""Deterministically recover ordered commitment widths for completed runs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dream_ablation.io import stream_jsonl, write_jsonl
from dream_ablation.model import DreamLogitsProvider, ModelOptions
from dream_ablation.scheduler import GenerationConfig, _remaining_positions, generation_step
from run_benchmark import append_jsonl, task_seed

BENCHMARKS = ("gsm8k", "mbpp_plus", "schema_json")
METHODS = tuple(
    "adaptive_delta_" + value
    for value in ("0p10", "0p15", "0p20", "0p30", "0p50", "0p75", "1p00", "1p50")
)


@dataclass(frozen=True)
class ReplayResult:
    token_ids: list[int]
    commitments: list[int]
    histogram: list[int]
    eos_position: int | None
    verifier_rows: int
    verifier_passes: int


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def config_from(metadata: dict[str, Any]) -> GenerationConfig:
    if metadata.get("method_type") != "adaptive":
        raise ValueError(f"not an adaptive run: {metadata.get('method')}")
    return GenerationConfig(
        method="adaptive",
        continuation_length=int(metadata["continuation_length"]),
        delta=float(metadata["delta"]),
        max_candidates=int(metadata["max_candidates"]),
        verifier_microbatch_size=int(metadata["verifier_microbatch_size"]),
        temperature=float(metadata["temperature"]),
        top_p=metadata.get("top_p"),
        confidence=str(metadata["confidence_metric"]),
    )


def generate_trajectory(
    provider: DreamLogitsProvider,
    prompt_ids: torch.Tensor,
    config: GenerationConfig,
) -> ReplayResult:
    canvas_start = int(prompt_ids.shape[1])
    canvas = torch.full(
        (1, config.continuation_length),
        provider.mask_token_id,
        dtype=torch.long,
        device=provider.device,
    )
    state = torch.cat((prompt_ids.to(provider.device), canvas), dim=1)
    commitments: list[int] = []
    histogram = [0] * config.max_candidates
    verifier_rows = 0
    verifier_passes = 0
    while True:
        remaining = _remaining_positions(
            state,
            canvas_start=canvas_start,
            canvas_length=config.continuation_length,
            mask_token_id=provider.mask_token_id,
        )
        if remaining.numel() == 0:
            break
        step = generation_step(
            provider,
            state,
            canvas_start=canvas_start,
            config=config,
            step_index=len(commitments),
            capture_trace=False,
        )
        state = step.state
        width = int(step.accepted_prefix_length)
        commitments.append(width)
        histogram[width - 1] += 1
        verifier_rows += step.verifier_rows
        verifier_passes += step.verifier_forward_passes

    full = [int(value) for value in state[0, canvas_start:].detach().cpu().tolist()]
    try:
        eos_position = full.index(provider.eos_token_id)
    except ValueError:
        eos_position = None
    token_ids = full if eos_position is None else full[:eos_position]
    return ReplayResult(
        token_ids, commitments, histogram, eos_position, verifier_rows, verifier_passes
    )


def validate(
    saved: dict[str, Any], replay: ReplayResult, provider: DreamLogitsProvider,
    continuation_length: int,
) -> None:
    label = f"{saved['method']}/{saved['example_id']}"
    checks = {
        "denoising_steps": (len(replay.commitments), int(saved["denoising_steps"])),
        "commitment_histogram": (replay.histogram, saved["commitment_histogram"]),
        "eos_position": (replay.eos_position, saved.get("eos_position")),
        "verifier_rows": (replay.verifier_rows, int(saved["verifier_rows"])),
        "verifier_forward_passes": (
            replay.verifier_passes, int(saved["verifier_forward_passes"])
        ),
        "raw_completion": (provider.decode(replay.token_ids), saved["raw_completion"]),
        "committed_tokens": (sum(replay.commitments), continuation_length),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise RuntimeError(
                f"replay mismatch for {label}: {name}: "
                f"replayed={actual!r}, saved={expected!r}"
            )


def repair(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows = list(stream_jsonl(path))
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        example_id = str(row["example_id"])
        widths = row.get("commitments_by_step")
        if (
            example_id not in seen
            and isinstance(widths, list)
            and len(widths) == int(row.get("denoising_steps", -1))
            and all(isinstance(value, int) and value >= 1 for value in widths)
        ):
            kept.append(row)
            seen.add(example_id)
    if len(kept) != len(rows):
        write_jsonl(path, kept)
    return seen


def discover(
    output_root: Path, benchmarks: list[str], methods: list[str]
) -> list[tuple[Path, dict[str, Any]]]:
    runs = []
    for benchmark in benchmarks:
        for method in methods:
            method_dir = output_root / benchmark / method
            metadata_path = method_dir / "metadata.json"
            records_path = method_dir / "records.jsonl"
            if not metadata_path.exists() or not records_path.exists():
                raise FileNotFoundError(f"missing run under {method_dir}")
            metadata = load_json(metadata_path)
            if metadata.get("status") != "complete":
                raise ValueError(f"run is not complete: {method_dir}")
            runs.append((method_dir, metadata))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "v3_matched")
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, help="at most N pending tasks per method")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    runs = discover(args.output_root, args.benchmarks, args.methods)
    models = {str(meta["model"]) for _, meta in runs}
    revisions = {meta.get("model_revision") for _, meta in runs}
    dtypes = {str(meta["dtype"]) for _, meta in runs}
    if len(models) != 1 or len(revisions) != 1 or len(dtypes) != 1:
        raise ValueError("selected runs do not share one model revision and dtype")
    provider = DreamLogitsProvider.from_pretrained(
        ModelOptions(
            model_id=models.pop(),
            revision=revisions.pop(),
            device=args.device,
            dtype=dtypes.pop(),
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            require_lean_logits=True,
        )
    )

    for method_dir, metadata in runs:
        config = config_from(metadata)
        output_path = method_dir / "trajectories.jsonl"
        completed = repair(output_path)
        pending = [
            row for row in stream_jsonl(method_dir / "records.jsonl")
            if str(row["example_id"]) not in completed
        ]
        if args.limit is not None:
            pending = pending[: args.limit]
        for index, saved in enumerate(pending, start=1):
            example_id = str(saved["example_id"])
            prompt_ids = provider.encode_prompt(str(saved["rendered_prompt"]))
            torch.manual_seed(task_seed(int(metadata["base_seed"]), config.label, example_id))
            replay = generate_trajectory(provider, prompt_ids, config)
            validate(saved, replay, provider, int(metadata["continuation_length"]))
            append_jsonl(
                output_path,
                {
                    "benchmark": saved["benchmark"],
                    "example_id": example_id,
                    "method": config.label,
                    "denoising_steps": len(replay.commitments),
                    "commitments_by_step": replay.commitments,
                },
            )
            print(
                f"[{saved['benchmark']}/{config.label}] {index}/{len(pending)} "
                f"{example_id}: {len(replay.commitments)} steps",
                flush=True,
            )


if __name__ == "__main__":
    main()
