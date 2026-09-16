#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from dream_ablation.io import percentile, sha256_file, stream_jsonl, write_json, write_jsonl
from dream_ablation.model import DEFAULT_MODEL, DreamLogitsProvider, ModelOptions
from dream_ablation.scheduler import GenerationConfig, generate
from dream_ablation.scoring import score_record


def parse_method(
    value: str,
    continuation_length: int,
    microbatch: int,
    *,
    max_candidates: int,
    sampling: dict[str, dict[str, Any]],
) -> GenerationConfig:
    """Build one method config.

    Sampling is per method kind, so the fixed arm can follow the Dream/ADJUST
    instruct recipe while the adaptive arm stays on greedy argmax.
    """
    try:
        kind, raw = value.split(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("methods must look like fixed:4 or adaptive:0.10") from error
    if kind not in sampling:
        raise argparse.ArgumentTypeError(f"unknown method kind: {kind}")
    shared = dict(
        continuation_length=continuation_length,
        max_candidates=max_candidates,
        verifier_microbatch_size=microbatch,
        **sampling[kind],
    )
    if kind == "fixed":
        return GenerationConfig(method="fixed", fixed_k=int(raw), **shared)
    return GenerationConfig(method="adaptive", delta=float(raw), **shared)


def task_seed(base_seed: int, label: str, example_id: str) -> int:
    """Stable per (seed, method, task) draw.

    Sampling is stochastic once temperature > 0, so the seed has to be a
    function of the task rather than of arrival order. That keeps timing
    repeats, trace replay, and resume-after-interruption all reproducing the
    same stream.
    """
    digest = hashlib.sha256(f"{base_seed}:{label}:{example_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


# A completion whose first EOS lands this early cannot contain a worked
# solution; it is the collapse signature, and it is k-dependent, so it belongs
# in the summary rather than being something you find by reading records.
EARLY_EOS_TOKENS = 16


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory(device: torch.device) -> int:
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def validate_records(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("Dataset is empty")
    benchmark = str(rows[0].get("benchmark"))
    seen: set[str] = set()
    for row in rows:
        if row.get("benchmark") != benchmark:
            raise ValueError("A dataset file must contain exactly one benchmark")
        example_id = str(row["example_id"])
        if example_id in seen:
            raise ValueError(f"Duplicate example_id: {example_id}")
        if not isinstance(row.get("prompt"), str) or not row["prompt"]:
            raise ValueError(f"Invalid prompt for {example_id}")
        seen.add(example_id)
    return benchmark


def select_sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    """Select one reproducible subset while retaining the frozen dataset order."""
    if size < 1:
        raise ValueError("sample size must be positive")
    if size >= len(rows):
        return rows
    indices = sorted(random.Random(seed).sample(range(len(rows)), size))
    return [rows[index] for index in indices]


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["example_id"]) for row in stream_jsonl(path)}


def repair_partial_artifacts(records_path: Path, traces_path: Path) -> set[str]:
    """Keep only task records with a complete, unique trace sequence."""
    records = list(stream_jsonl(records_path)) if records_path.exists() else []
    traces = list(stream_jsonl(traces_path)) if traces_path.exists() else []
    trace_counts: dict[str, int] = {}
    for trace in traces:
        example_id = str(trace["example_id"])
        trace_counts[example_id] = trace_counts.get(example_id, 0) + 1
    seen: set[str] = set()
    kept_records = []
    for record in records:
        example_id = str(record["example_id"])
        # Records written with --trace-limit have no traces by design. Demanding
        # a full trace sequence for them would discard every one on resume, so
        # the run would restart from zero forever. Records predating the flag
        # have no "traced" key and were always traced.
        if record.get("traced", True):
            complete = trace_counts.get(example_id, 0) == int(record["denoising_steps"])
        else:
            complete = example_id not in trace_counts
        if complete and example_id not in seen:
            kept_records.append(record)
            seen.add(example_id)
    kept_traces = [trace for trace in traces if str(trace["example_id"]) in seen]
    if len(kept_records) != len(records) or len(kept_traces) != len(traces):
        write_jsonl(records_path, kept_records)
        write_jsonl(traces_path, kept_traces)
    return seen


def run_method(
    provider: DreamLogitsProvider,
    rows: list[dict[str, Any]],
    config: GenerationConfig,
    output_root: Path,
    *,
    timing_repeats: int,
    trace_limit: int,
    use_chat_template: bool,
    overwrite: bool,
    base_seed: int = 0,
) -> None:
    method_dir = output_root / config.label
    records_path = method_dir / "records.jsonl"
    traces_path = method_dir / "decision_traces.jsonl"
    metadata_path = method_dir / "metadata.json"
    if overwrite:
        for path in (
            records_path,
            traces_path,
            metadata_path,
            method_dir / "evalplus_samples.jsonl",
            method_dir / "records.scored.jsonl",
            method_dir / "evalplus_results.json",
        ):
            if path.exists():
                path.unlink()
    completed = repair_partial_artifacts(records_path, traces_path)
    method_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "running",
        "method": config.label,
        "method_type": config.method,
        "fixed_k": config.fixed_k,
        "delta": config.delta,
        "tau_nats": config.tau_nats,
        "continuation_length": config.continuation_length,
        "eos_policy": "complete_canvas_then_truncate_at_first_eos",
        "max_candidates": config.max_candidates,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "confidence_metric": config.confidence,
        "base_seed": base_seed,
        "verifier_distribution": "unfiltered_log_softmax",
        "verifier_microbatch_size": config.verifier_microbatch_size,
        "timing_repeats": timing_repeats,
        "trace_limit": trace_limit,
        "model": provider.tokenizer.name_or_path,
        "model_revision": getattr(provider.model.config, "_commit_hash", None),
        "dtype": str(next(provider.model.parameters()).dtype).replace("torch.", ""),
        "proposer_is_verifier": True,
        "prompt_template": "tokenizer_chat_template" if use_chat_template else "raw",
    }
    write_json(metadata_path, metadata)

    # One untimed warmup keeps startup/JIT work out of the first task latency.
    first_pending = next((row for row in rows if str(row["example_id"]) not in completed), None)
    if first_pending is not None:
        rendered = provider.render_prompt(first_pending["prompt"], use_chat_template=use_chat_template)
        prompt_ids = provider.encode_prompt(rendered)
        torch.manual_seed(task_seed(base_seed, config.label, "__warmup__"))
        generate(provider, prompt_ids, config, capture_traces=False)
        synchronize(provider.device)

    for task_index, row in enumerate(rows):
        example_id = str(row["example_id"])
        if example_id in completed:
            continue
        rendered = provider.render_prompt(row["prompt"], use_chat_template=use_chat_template)
        prompt_ids = provider.encode_prompt(rendered)
        model_limit = getattr(provider.model.config, "max_position_embeddings", None)
        requested = int(prompt_ids.shape[1]) + config.continuation_length
        if model_limit is not None and requested > int(model_limit):
            raise ValueError(
                f"{example_id}: prompt+canvas length {requested} exceeds model limit {model_limit}"
            )

        latencies: list[float] = []
        peaks: list[int] = []
        canonical = None
        for repeat in range(timing_repeats):
            torch.manual_seed(task_seed(base_seed, config.label, example_id))
            reset_peak_memory(provider.device)
            synchronize(provider.device)
            started = time.perf_counter()
            result = generate(provider, prompt_ids, config, capture_traces=False)
            synchronize(provider.device)
            latencies.append(time.perf_counter() - started)
            peaks.append(peak_memory(provider.device))
            if canonical is None:
                canonical = result
            elif result.continuation_token_ids != canonical.continuation_token_ids:
                raise RuntimeError(f"Determinism check failed for {config.label}/{example_id}")
        assert canonical is not None
        # Tracing costs a whole extra generation, so only the first
        # `trace_limit` tasks pay it. Those tasks also carry the determinism
        # check below, which is the only one left when timing_repeats == 1.
        should_trace = task_index < trace_limit
        traced = None
        if should_trace:
            torch.manual_seed(task_seed(base_seed, config.label, example_id))
            traced = generate(provider, prompt_ids, config, capture_traces=True)
            synchronize(provider.device)
            if traced.continuation_token_ids != canonical.continuation_token_ids:
                raise RuntimeError(
                    f"Trace replay changed output for {config.label}/{example_id}"
                )
        completion = provider.decode(canonical.continuation_token_ids)
        scoring = score_record(row, completion)
        # Positional, not score-derived. The old flag reported "scoring failed
        # and an EOS exists", which conflated two things and named neither: it
        # would not have surfaced a canvas whose first EOS landed at position 3.
        early_eos = (
            canonical.eos_position is not None
            and canonical.eos_position < EARLY_EOS_TOKENS
        )
        decision_count = max(canonical.denoising_steps, 1)
        committed_count = sum(
            (index + 1) * count
            for index, count in enumerate(canonical.commitment_histogram)
        )
        output = {
            "benchmark": row["benchmark"],
            "example_id": example_id,
            "method": config.label,
            "raw_prompt": row["prompt"],
            "rendered_prompt": rendered,
            "raw_completion": completion,
            **scoring,
            "output_token_count": len(canonical.continuation_token_ids),
            "eos_position": canonical.eos_position,
            "denoising_steps": canonical.denoising_steps,
            "dream_proposal_passes": canonical.proposer_passes,
            "verifier_rows": canonical.verifier_rows,
            "verifier_forward_passes": canonical.verifier_forward_passes,
            "wall_time_seconds": statistics.median(latencies),
            "timing_repeats_seconds": latencies,
            "p95_repeat_seconds": percentile(latencies, 0.95),
            "peak_gpu_memory_bytes": max(peaks),
            "mean_committed_tokens_per_step": committed_count / decision_count,
            "commitment_histogram": canonical.commitment_histogram,
            "max_candidates": config.max_candidates,
            "maximum_length_termination": canonical.maximum_length_termination,
            "early_eos": early_eos,
            "traced": should_trace,
            "source": {key: value for key, value in row.items() if key not in {"prompt", "schema", "target"}},
        }
        append_jsonl(records_path, output)
        if traced is not None:
            for trace in traced.traces:
                append_jsonl(
                    traces_path,
                    {"benchmark": row["benchmark"], "example_id": example_id,
                     "method": config.label, **trace},
                )
        print(
            f"[{config.label}] {task_index + 1}/{len(rows)} {example_id}: "
            f"{statistics.median(latencies):.3f}s, steps={canonical.denoising_steps}",
            flush=True,
        )

    if rows[0]["benchmark"] == "mbpp_plus":
        all_records = list(stream_jsonl(records_path))
        write_jsonl(
            method_dir / "evalplus_samples.jsonl",
            [
                {"task_id": row["example_id"], "solution": row["parsed_output"]}
                for row in all_records
            ],
        )
    metadata["status"] = "complete"
    metadata["completed_examples"] = len(existing_ids(records_path))
    metadata["artifacts"] = {
        "records": "records.jsonl",
        "decision_traces": "decision_traces.jsonl",
        "evalplus_samples": "evalplus_samples.jsonl" if rows[0]["benchmark"] == "mbpp_plus" else None,
    }
    write_json(metadata_path, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic Dream ablation methods")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--continuation-length", type=int, required=True)
    parser.add_argument("--verifier-microbatch-size", type=int, default=8)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument(
        "--trace-limit",
        type=int,
        default=20,
        help="capture decision traces for the first N tasks of each method. "
        "Tracing costs one extra full generation per task, so this bounds "
        "that cost. 0 disables it, which also disables the audit's "
        "independent check of the acceptance rule and empties the "
        "error-curve figure.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--require-lean-logits", action="store_true",
                        help="fail at load if the memory-lean logits path is unavailable")
    parser.add_argument("--max-candidates", type=int, default=8,
                        help="cap on tokens proposed (and so committed) per step")
    for kind, default_conf in (("fixed", "logp"), ("adaptive", "logp")):
        parser.add_argument(f"--{kind}-temperature", type=float, default=0.0)
        parser.add_argument(f"--{kind}-top-p", type=float, default=None)
        parser.add_argument(f"--{kind}-confidence", choices=("logp", "entropy"),
                            default=default_conf)
    parser.add_argument(
        "--sample-size",
        type=int,
        help="deterministically sample this many tasks using --seed; unlike "
        "--limit, this does not bias the run toward the start of the dataset",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.timing_repeats < 1:
        parser.error("--timing-repeats must be positive")
    rows = list(stream_jsonl(args.dataset))
    benchmark = validate_records(rows)
    if args.limit is not None and args.sample_size is not None:
        parser.error("--limit and --sample-size are mutually exclusive")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        rows = rows[: args.limit]
    if args.sample_size is not None:
        if args.sample_size < 1:
            parser.error("--sample-size must be positive")
        rows = select_sample(rows, args.sample_size, args.seed)
    if args.max_candidates < 1:
        parser.error("--max-candidates must be positive")
    sampling = {
        kind: {
            "temperature": getattr(args, f"{kind}_temperature"),
            "top_p": getattr(args, f"{kind}_top_p"),
            "confidence": getattr(args, f"{kind}_confidence"),
        }
        for kind in ("fixed", "adaptive")
    }
    configs = [
        parse_method(
            value,
            args.continuation_length,
            args.verifier_microbatch_size,
            max_candidates=args.max_candidates,
            sampling=sampling,
        )
        for value in args.methods
    ]
    if len({config.label for config in configs}) != len(configs):
        parser.error("duplicate methods")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    provider = DreamLogitsProvider.from_pretrained(
        ModelOptions(
            model_id=args.model, revision=args.revision, device=args.device,
            dtype=args.dtype, cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            require_lean_logits=args.require_lean_logits,
        )
    )
    run_metadata = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "benchmark": benchmark,
        "number_of_tasks": len(rows),
        "example_ids": [str(row["example_id"]) for row in rows],
        "model": args.model,
        "revision": args.revision,
        "dtype": args.dtype,
        "device": args.device,
        "physical_gpu": os.environ.get("EXPERIMENT_PHYSICAL_GPU_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "host": platform.node(),
        "seed": args.seed,
        "methods": [config.label for config in configs],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    worker_label = os.environ.get("EXPERIMENT_PHYSICAL_GPU_ID", str(os.getpid()))
    write_json(args.output_root / f"run_metadata_worker_{worker_label}.json", run_metadata)
    for config in configs:
        run_method(
            provider, rows, config, args.output_root,
            timing_repeats=args.timing_repeats,
            trace_limit=args.trace_limit,
            use_chat_template=not args.no_chat_template,
            overwrite=args.overwrite,
            base_seed=args.seed,
        )


if __name__ == "__main__":
    main()
