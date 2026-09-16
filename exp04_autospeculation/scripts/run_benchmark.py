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

from autospec.io import percentile, sha256_file, stream_jsonl, write_json, write_jsonl
from autospec.autospeculation import AutospecConfig, generate_autospeculative
from autospec.blockwise import generate_blockwise
from autospec.model import FAMILIES, MaskedDiffusionProvider, ModelOptions
from autospec.timing import StageTimer
from autospec.scheduler import GenerationConfig, generate
from autospec.scoring import CODE_BENCHMARKS, score_record


def parse_method(
    value: str,
    continuation_length: int,
    microbatch: int,
    *,
    max_candidates: int,
    sampling: dict[str, dict[str, Any]],
    autospec_max_batch_rows: int | None = None,
) -> GenerationConfig | AutospecConfig:
    """Build one method config.

    Sampling is per method kind, so each arm can carry its own decoding recipe.
    Keeping them identical is what makes the arms comparable; see README.
    """
    try:
        kind, raw = value.split(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "methods must look like fixed:4, adaptive:0.10 or autospec:<rho>"
        ) from error
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
    if kind == "adaptive":
        return GenerationConfig(method="adaptive", delta=float(raw), **shared)
    # Algorithm 3. `raw` is rho; "auto" takes the paper's 1/log2(n). It ignores
    # the shared sampling flags: AutospecConfig fixes T=1.0 with no top-p, and
    # it has no confidence metric because sigma decides its order (DESIGN.md
    # D1, D2).
    rho = None if raw in ("auto", "default", "") else float(raw)
    return AutospecConfig(
        continuation_length=continuation_length,
        rho=rho,
        # Default to the whole canvas, so the root block is one physical round
        # wherever memory allows it (DESIGN.md D5).
        max_batch_rows=autospec_max_batch_rows or continuation_length,
    )


def task_seed(base_seed: int, label: str, example_id: str) -> int:
    """Stable per (seed, method, task) draw.

    Sampling is stochastic once temperature > 0, so the seed has to be a
    function of the task rather than of arrival order. That keeps timing
    repeats, trace replay, and resume-after-interruption all reproducing the
    same stream.
    """
    digest = hashlib.sha256(f"{base_seed}:{label}:{example_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def sigma_seed(base_seed: int, example_id: str) -> int:
    """Seed for the position permutation.

    Deliberately NOT a function of the method label, unlike `task_seed`, so
    that any two sigma-consuming configurations of Algorithm 3 (a rho sweep,
    say) walk the SAME permutation on the same task and stay a paired
    comparison rather than a comparison of two different random orders.

    Only Algorithm 3 uses this. Fixed-k and adaptive order their positions by
    confidence and never see a permutation.
    """
    digest = hashlib.sha256(f"sigma:{base_seed}:{example_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def sigma_rank_for_task(
    base_seed: int, example_id: str, length: int, device: torch.device
) -> torch.Tensor:
    """`rank[p]` is the step at which canvas position `p` is visited.

    Drawn on a CPU generator so the permutation is identical across devices and
    does not consume from the global RNG stream the token draws use.
    """
    generator = torch.Generator(device="cpu").manual_seed(sigma_seed(base_seed, example_id))
    permutation = torch.randperm(length, generator=generator)
    rank = torch.empty(length, dtype=torch.long)
    rank[permutation] = torch.arange(length)
    return rank.to(device)


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


def sample(
    provider: MaskedDiffusionProvider,
    prompt_ids: torch.Tensor,
    config: Any,
    *,
    capture_traces: bool = False,
    sigma_rank: torch.Tensor | None = None,
    timer: StageTimer | None = None,
):
    """One generation, routed by checkpoint kind first, then method.

    Under a block diffusion checkpoint EVERY method runs inside a block, with
    the canvas a loop over blocks and a KV cache carrying the committed prefix
    -- fixed-k and adaptive through the scheduler's step loop, Algorithm 3
    through its recursion. The block loop owns the cache, the block-causal
    slices and the commits either way, so the dispatch is on the checkpoint.

    Without block structure the canvas is one piece: Algorithm 3 gets its own
    full-canvas entry point, and the other two share `scheduler.generate`.
    """
    if getattr(provider.conventions, "block_length", None) is not None:
        return generate_blockwise(
            provider, prompt_ids, config,
            capture_traces=capture_traces, sigma_rank=sigma_rank, timer=timer,
        )
    if isinstance(config, AutospecConfig):
        return generate_autospeculative(
            provider, prompt_ids, config,
            capture_traces=capture_traces, sigma_rank=sigma_rank,
        )
    return generate(
        provider, prompt_ids, config, capture_traces=capture_traces, timer=timer
    )


def run_method(
    provider: MaskedDiffusionProvider,
    rows: list[dict[str, Any]],
    config: GenerationConfig,
    output_root: Path,
    *,
    timing_repeats: int,
    trace_limit: int,
    profile_limit: int = 0,
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
    uses_sigma = isinstance(config, AutospecConfig)
    completed = repair_partial_artifacts(records_path, traces_path)
    method_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "running",
        "method": config.label,
        "method_type": config.method,
        "fixed_k": getattr(config, "fixed_k", None),
        "delta": getattr(config, "delta", None),
        "tau_nats": getattr(config, "tau_nats", None),
        "continuation_length": config.continuation_length,
        "eos_policy": "complete_canvas_then_truncate_at_first_eos",
        "max_candidates": getattr(config, "max_candidates", None),
        "temperature": getattr(config, "temperature", None),
        "top_p": getattr(config, "top_p", None),
        "confidence_metric": getattr(config, "confidence", None),
        "position_order": "sigma" if uses_sigma else "confidence",
        "sigma_seed_basis": "sha256(sigma:<seed>:<example_id>)" if uses_sigma else None,
        "block_length": getattr(provider.conventions, "block_length", None),
        "base_seed": base_seed,
        "verifier_distribution": "unfiltered_log_softmax",
        "verifier_microbatch_size": getattr(config, "verifier_microbatch_size", None),
        # Algorithm 3's rho, reported as CONFIGURED rather than as resolved.
        # `effective_rho` would be wrong here under block diffusion: it is
        # 1/log2(continuation_length), but `generate_blockwise` re-derives rho
        # per block from that block's masked count, so a canvas of 256 records
        # 0.125 while every block actually runs at 1/log2(64) = 0.167. There is
        # no single right number to record -- the last block and the block
        # holding the prompt remainder are narrower than the rest -- so the
        # knob is recorded as set, null for the paper's default, and the rule
        # that resolves it is recorded beside it.
        "rho": getattr(config, "rho", None),
        "rho_rule": (
            None
            if not isinstance(config, AutospecConfig)
            else "configured"
            if config.rho is not None
            else "1/log2(masked positions in the block)"
            if getattr(provider.conventions, "block_length", None) is not None
            else "1/log2(canvas)"
        ),
        "timing_repeats": timing_repeats,
        "trace_limit": trace_limit,
        "profile_limit": profile_limit,
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
        sample(
            provider, prompt_ids, config, capture_traces=False,
            sigma_rank=(
                sigma_rank_for_task(
                    base_seed, "__warmup__", config.continuation_length, provider.device
                )
                if uses_sigma else None
            ),
        )
        synchronize(provider.device)

    for task_index, row in enumerate(rows):
        example_id = str(row["example_id"])
        if example_id in completed:
            continue
        sigma_rank = (
            sigma_rank_for_task(
                base_seed, example_id, config.continuation_length, provider.device
            )
            if uses_sigma else None
        )
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
            result = sample(
                provider, prompt_ids, config,
                capture_traces=False, sigma_rank=sigma_rank,
            )
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
            traced = sample(
                provider, prompt_ids, config,
                capture_traces=True, sigma_rank=sigma_rank,
            )
            synchronize(provider.device)
            if traced.continuation_token_ids != canonical.continuation_token_ids:
                raise RuntimeError(
                    f"Trace replay changed output for {config.label}/{example_id}"
                )
        # Stage profiling is its own generation for the same reason tracing is:
        # CUDA events are cheap but not free, and `totals()` synchronizes. The
        # timed repeats above must stay exactly as they were, so nothing that
        # measures the parts is allowed to run inside the part being measured.
        stage_seconds: dict[str, float] = {}
        profiled_wall_seconds = None
        if task_index < profile_limit:
            timer = StageTimer(provider.device, enabled=True)
            torch.manual_seed(task_seed(base_seed, config.label, example_id))
            synchronize(provider.device)
            began = time.perf_counter()
            profiled = sample(
                provider, prompt_ids, config,
                capture_traces=False, sigma_rank=sigma_rank, timer=timer,
            )
            synchronize(provider.device)
            profiled_wall_seconds = time.perf_counter() - began
            if profiled.continuation_token_ids != canonical.continuation_token_ids:
                raise RuntimeError(
                    f"Profiling changed output for {config.label}/{example_id}"
                )
            stage_seconds = profiled.stage_seconds

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
        # Throughput, following the convention every serving stack and the
        # diffusion-LM papers use: tokens delivered divided by wall time.
        #   vLLM          total_output_tokens / elapsed_time
        #   Fast-dLLM     (generated ids != eos).sum() / elapsed_time
        # `tokens_per_second` is that number. But a masked diffusion model with
        # a fixed canvas does the SAME work whether it emits 20 tokens or 250,
        # and this project's early-EOS rate ranges from 0.00 to 0.68 across
        # methods -- so on its own that metric rates an early-EOS method as
        # catastrophically slow for a reason that is about output length, not
        # speed. `canvas_tokens_per_second` divides the work actually done by
        # the same wall time and is the one to compare schedulers on.
        latency = statistics.median(latencies)
        canvas_token_count = len(canonical.full_continuation_token_ids)
        # Commit forwards used to be folded into `proposer_passes`; they are
        # reported separately now, so add them back explicitly. This total is
        # the NFE every earlier run recorded and must not drift.
        forward_passes = (
            canonical.proposer_passes
            + canonical.commit_passes
            + canonical.verifier_forward_passes
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
            # Keeps its original meaning -- denoising forwards PLUS the commit
            # forwards -- because runs already on disk recorded it that way and
            # a field must not change meaning underneath them. The split lives
            # in the two fields below.
            "dream_proposal_passes": canonical.proposer_passes + canonical.commit_passes,
            "denoising_forward_passes": canonical.proposer_passes,
            "verifier_rows": canonical.verifier_rows,
            "verifier_forward_passes": canonical.verifier_forward_passes,
            "wall_time_seconds": latency,
            "canvas_token_count": canvas_token_count,
            "tokens_per_second": (
                len(canonical.continuation_token_ids) / latency if latency > 0 else float("nan")
            ),
            "canvas_tokens_per_second": (
                canvas_token_count / latency if latency > 0 else float("nan")
            ),
            # NFE in the diffusion-LM literature: model forward calls. Tokens
            # per forward is 1 for an autoregressive model by construction, so
            # it is the metric that says what the parallelism bought.
            "model_forward_passes": forward_passes,
            "tokens_per_forward": (
                canvas_token_count / forward_passes if forward_passes else float("nan")
            ),
            # Per-stage wall clock, and the wall time of the run they were
            # measured in -- the residual against that number is the Python and
            # sampling overhead outside the model. Empty unless profiled.
            "stage_seconds": stage_seconds,
            "profiled_wall_seconds": profiled_wall_seconds,
            "commit_passes": canonical.commit_passes,
            "timing_repeats_seconds": latencies,
            "p95_repeat_seconds": percentile(latencies, 0.95),
            "peak_gpu_memory_bytes": max(peaks),
            "mean_committed_tokens_per_step": committed_count / decision_count,
            "commitment_histogram": canonical.commitment_histogram,
            "max_candidates": getattr(config, "max_candidates", None),
            "maximum_length_termination": canonical.maximum_length_termination,
            "early_eos": early_eos,
            "traced": should_trace,
            "position_order": "sigma" if uses_sigma else "confidence",
            "sigma_seed": sigma_seed(base_seed, example_id) if uses_sigma else None,
            # Algorithm 3 only. `oracle_rounds_physical` is the sequential
            # depth the card actually executed -- the paper's round complexity,
            # counted as run rather than as scheduled (DESIGN.md D4).
            **{
                name: getattr(canonical, name)
                for name in (
                    "oracle_rounds_physical",
                    "oracle_rows", "blocks_opened", "speculations_accepted",
                    "speculations_rejected", "fallback_draws",
                    "max_fallback_draws_in_call", "max_pooled_request_rows",
                    "singleton_blocks",
                )
                if hasattr(canonical, name)
            },
            "source": {key: value for key, value in row.items() if key not in {"prompt", "target"}},
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

    if rows[0]["benchmark"] in CODE_BENCHMARKS:
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
        "evalplus_samples": (
            "evalplus_samples.jsonl" if rows[0]["benchmark"] in CODE_BENCHMARKS else None
        ),
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
    parser.add_argument("--model", required=True,
                        help="HF checkpoint id. Required: this experiment has no "
                             "default model, so a stale one cannot leak into a run.")
    parser.add_argument("--block-length", type=int,
                        help="block size for a block diffusion checkpoint. MUST "
                             "match the checkpoint (SDAR ships -b8 ... -b64); a "
                             "mismatch gives an attention pattern the weights "
                             "were never trained with.")
    parser.add_argument("--family", default="dream", choices=sorted(FAMILIES),
                        help="calling conventions for the checkpoint (logits "
                             "alignment and attention mask). Getting this wrong "
                             "still produces fluent text; see autospec/model.py.")
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
    parser.add_argument(
        "--profile-stages",
        type=int,
        default=0,
        metavar="N",
        help="decompose latency into proposer / verifier / commit (oracle, for "
        "autospec) for the first N tasks of each method. Runs one extra "
        "generation per profiled task and is OFF by default, because the "
        "measurement must never run inside the timed repeats it explains.",
    )
    parser.add_argument("--max-candidates", type=int, default=8,
                        help="cap on tokens proposed (and so committed) per step")
    parser.add_argument("--autospec-max-batch-rows", type=int,
                        help="most state rows in one autospec forward. Defaults "
                        "to the canvas length so Algorithm 3's root block is a "
                        "single physical round. Changes cost only, never the "
                        "sample; lower it if the root block will not fit.")
    for kind, default_conf in (("fixed", "logp"), ("adaptive", "logp"), ("autospec", "logp")):
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
        for kind in ("fixed", "adaptive", "autospec")
    }
    configs = [
        parse_method(
            value,
            args.continuation_length,
            args.verifier_microbatch_size,
            max_candidates=args.max_candidates,
            sampling=sampling,
            autospec_max_batch_rows=args.autospec_max_batch_rows,
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
    provider = MaskedDiffusionProvider.from_pretrained(
        ModelOptions(
            model_id=args.model, family=args.family,
            block_length=args.block_length,
            revision=args.revision, device=args.device,
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
        "family": args.family,
        "block_length": args.block_length,
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
            profile_limit=args.profile_stages,
            use_chat_template=not args.no_chat_template,
            overwrite=args.overwrite,
            base_seed=args.seed,
        )


if __name__ == "__main__":
    main()
