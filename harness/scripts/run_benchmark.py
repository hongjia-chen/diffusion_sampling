#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
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

# Runnable from any directory: the package lives next to scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sdar_harness.io import percentile, sha256_file, stream_jsonl, write_json, write_jsonl
from sdar_harness.blockwise import generate_blockwise
from sdar_harness.cross import CrossConfig, generate_cross
from sdar_harness.vgb import VGBConfig, generate_vgb, parse_vgb_spec
from sdar_harness.model import (
    FAMILIES, MaskedDiffusionProvider, ModelOptions, assert_shared_vocabulary, trained_block_length,
)
from sdar_harness.timing import StageTimer
from sdar_harness.scheduler import GenerationConfig, generate
from sdar_harness.scoring import CODE_BENCHMARKS, score_record


def parse_method(
    value: str,
    continuation_length: int,
    microbatch: int,
    *,
    max_candidates: int,
    sampling: dict[str, dict[str, Any]],
    early_stop: bool = True,
    veto: bool = False,
    vgb_budget: int | None = 512,
    vgb_row_chunk: int = 16,
) -> GenerationConfig | VGBConfig:
    """Build one method config.

    Sampling is per method kind, so each arm can carry its own decoding recipe.
    Keeping them identical is what makes the arms comparable; see README.
    """
    try:
        kind, raw = value.split(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "methods must look like fixed:4, dynamic:0.90, adaptive:0.10, "
            "xadaptive:0.10 or xadaptive-remask:0.10"
        ) from error
    if kind == "vgb":
        # Density-ratio VGB, SDAR-8B target / SDAR-1.7B reference (DESIGN.md).
        # e.g. vgb:lam0.5:mom:sl8x8 or vgb:lam1:mom:T0:conf:draft:exc4
        return parse_vgb_spec(raw, continuation_length=continuation_length,
                              budget=vgb_budget, early_stop=early_stop,
                              row_chunk=vgb_row_chunk)
    cross = kind in ("xadaptive", "xadaptive-remask")
    if cross:
        # Dream-verified adaptive (DESIGN.md F1-F4). Same proposer recipe as
        # the SDAR-only adaptive arms, so the verifier is the only difference.
        return CrossConfig(
            method="adaptive", delta=float(raw), continuation_length=continuation_length,
            max_candidates=max_candidates, verifier_microbatch_size=microbatch,
            early_stop=early_stop, remask=(kind == "xadaptive-remask"), veto=veto,
            **sampling["adaptive"],
        )
    if kind not in sampling:
        raise argparse.ArgumentTypeError(f"unknown method kind: {kind}")
    shared = dict(
        continuation_length=continuation_length,
        max_candidates=max_candidates,
        verifier_microbatch_size=microbatch,
        early_stop=early_stop,
        **sampling[kind],
    )
    if kind == "fixed":
        return GenerationConfig(method="fixed", fixed_k=int(raw), **shared)
    if kind == "dynamic":
        # SDAR's low_confidence_dynamic: raw is the probability threshold.
        return GenerationConfig(method="dynamic", threshold=float(raw), **shared)
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


def sample(
    provider: MaskedDiffusionProvider,
    prompt_ids: torch.Tensor,
    config: Any,
    *,
    capture_traces: bool = False,
    timer: StageTimer | None = None,
    verifier: MaskedDiffusionProvider | None = None,
    verifier_prompt_ids: torch.Tensor | None = None,
):
    """One generation: VGB, cross-model, else the block loop, else the full canvas."""
    if isinstance(config, VGBConfig):
        if config.needs_reference and verifier is None:
            raise ValueError(f"{config.label} needs --verifier-model (the 1.7B reference)")
        if verifier_prompt_ids is not None and not torch.equal(
            prompt_ids.cpu(), verifier_prompt_ids.cpu()
        ):
            raise ValueError("target and reference tokenize the prompt differently")
        # run_method seeds torch per (seed, method, task) right before this
        # call; deriving the sampler's generator from it keeps runs resumable
        # and repeatable.
        generator = torch.Generator(device=provider.device).manual_seed(torch.initial_seed())
        # VGB traces through its config; tracing reads what each move computed
        # and draws nothing, so the traced walk takes the same moves.
        if capture_traces and not config.trace:
            config = dataclasses.replace(config, trace=True)
        return generate_vgb(
            provider, verifier if config.needs_reference else None, prompt_ids, config,
            generator=generator, timer=timer,
        )
    if isinstance(config, CrossConfig):
        if verifier is None or verifier_prompt_ids is None:
            raise ValueError(f"{config.label} needs --verifier-model")
        return generate_cross(
            provider, verifier, prompt_ids, verifier_prompt_ids, config,
            capture_traces=capture_traces, timer=timer,
        )
    if getattr(provider.conventions, "block_length", None) is not None:
        return generate_blockwise(
            provider, prompt_ids, config,
            capture_traces=capture_traces, timer=timer,
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
    verifier: MaskedDiffusionProvider | None = None,
) -> None:
    cross = isinstance(config, CrossConfig)
    vgb = isinstance(config, VGBConfig)
    verifier = verifier if cross or (vgb and config.needs_reference) else None
    # Every card that holds weights, not just the one the sampler reads from.
    # A pipeline-split model's last stage cannot finish before its first, but
    # a timer that syncs only the output card still stops early whenever the
    # host has not blocked on the handoff (non_blocking copies). Synchronizing
    # all of them before and after is what keeps tokens/s a wall-clock number.
    proposer_devices = list(getattr(provider, "devices", [provider.device]))
    verifier_devices = (
        list(getattr(verifier, "devices", [verifier.device])) if verifier is not None else []
    )
    devices = proposer_devices + verifier_devices

    def sync() -> None:
        for device in devices:
            synchronize(device)

    def verifier_prompt(row_prompt: str) -> torch.Tensor | None:
        if verifier is None:
            return None
        return verifier.encode_prompt(
            verifier.render_prompt(row_prompt, use_chat_template=use_chat_template)
        )

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
        "fixed_k": getattr(config, "fixed_k", None),
        "delta": getattr(config, "delta", None),
        "tau_nats": getattr(config, "tau_nats", None),
        "threshold": getattr(config, "threshold", None),
        "continuation_length": config.continuation_length,
        "eos_policy": (
            "early_stop_at_first_stop_token"
            if getattr(config, "early_stop", False)
            else "complete_canvas_then_truncate_at_first_stop_token"
        ),
        "early_stop": getattr(config, "early_stop", False),
        "stop_token_ids": sorted(getattr(provider, "stop_token_ids", ())),
        "vocab_limit": getattr(provider, "vocab_limit", None),
        "max_candidates": getattr(config, "max_candidates", None),
        "temperature": getattr(config, "temperature", None),
        "top_p": getattr(config, "top_p", None),
        "confidence_metric": getattr(config, "confidence", None),
        "position_order": getattr(config, "forward_selection", "confidence"),
        "block_length": getattr(provider.conventions, "block_length", None),
        "base_seed": base_seed,
        "verifier_distribution": "unfiltered_log_softmax",
        "verifier_microbatch_size": getattr(config, "verifier_microbatch_size", None),
        "timing_repeats": timing_repeats,
        "trace_limit": trace_limit,
        "profile_limit": profile_limit,
        "model": provider.tokenizer.name_or_path,
        "model_revision": getattr(provider.model.config, "_commit_hash", None),
        "dtype": str(next(provider.model.parameters()).dtype).replace("torch.", ""),
        "proposer_is_verifier": verifier is None,
        "devices": [str(device) for device in proposer_devices],
        "layer_split": getattr(provider, "layer_split", None),
        "verifier_model": verifier.tokenizer.name_or_path if verifier is not None else None,
        "acceptance_ratio": "dream_only" if cross else "same_model",
        "veto": getattr(config, "veto", None),
        "remask": getattr(config, "remask", None),
        "remask_bottom": getattr(config, "remask_bottom", None) if cross else None,
        "verifier_group_rows": getattr(config, "group_rows", None),
        "prompt_template": "tokenizer_chat_template" if use_chat_template else "raw",
        "vgb": dataclasses.asdict(config) if vgb else None,
    }
    write_json(metadata_path, metadata)

    # One untimed warmup keeps startup/JIT work out of the first task latency.
    first_pending = next((row for row in rows if str(row["example_id"]) not in completed), None)
    # A VGB warmup is a whole extra task (minutes), and its latency is not a
    # headline number; skip it.
    if first_pending is not None and not vgb:
        rendered = provider.render_prompt(first_pending["prompt"], use_chat_template=use_chat_template)
        prompt_ids = provider.encode_prompt(rendered)
        torch.manual_seed(task_seed(base_seed, config.label, "__warmup__"))
        sample(provider, prompt_ids, config, capture_traces=False, verifier=verifier,
               verifier_prompt_ids=verifier_prompt(first_pending["prompt"]))
        sync()

    for task_index, row in enumerate(rows):
        example_id = str(row["example_id"])
        if example_id in completed:
            continue
        rendered = provider.render_prompt(row["prompt"], use_chat_template=use_chat_template)
        prompt_ids = provider.encode_prompt(rendered)
        vprompt_ids = verifier_prompt(row["prompt"])
        model_limit = getattr(provider.model.config, "max_position_embeddings", None)
        requested = int(prompt_ids.shape[1]) + config.continuation_length
        if model_limit is not None and requested > int(model_limit):
            raise ValueError(
                f"{example_id}: prompt+canvas length {requested} exceeds model limit {model_limit}"
            )

        latencies: list[float] = []
        peaks: list[int] = []
        device_peaks: list[list[int]] = []
        verifier_peaks: list[int] = []
        for_both = dict(verifier=verifier, verifier_prompt_ids=vprompt_ids)
        canonical = None
        for repeat in range(timing_repeats):
            torch.manual_seed(task_seed(base_seed, config.label, example_id))
            for device in devices:
                reset_peak_memory(device)
            sync()
            started = time.perf_counter()
            result = sample(
                provider, prompt_ids, config,
                capture_traces=False, **for_both,
            )
            # Both cards: the answer is not done until the verifier's last
            # forward is, and timing only one device would undercount.
            sync()
            latencies.append(time.perf_counter() - started)
            # A split model's footprint is the sum over its cards; the
            # per-card list is what says whether each one fits.
            per_device = [peak_memory(device) for device in proposer_devices]
            device_peaks.append(per_device)
            peaks.append(sum(per_device))
            if verifier is not None:
                verifier_peaks.append(sum(peak_memory(device) for device in verifier_devices))
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
                capture_traces=True, **for_both,
            )
            sync()
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
            timer = StageTimer(
                provider.device, enabled=True,
                sync_devices=devices if len(devices) > 1 else None,
            )
            torch.manual_seed(task_seed(base_seed, config.label, example_id))
            sync()
            began = time.perf_counter()
            profiled = sample(
                provider, prompt_ids, config,
                capture_traces=False, timer=timer, **for_both,
            )
            sync()
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
        # `tokens_per_second` is that number and is the headline (DESIGN.md F1).
        # With early stop the work done is no longer the canvas: it is the
        # positions actually committed, `decoded_positions`, which is what
        # `tokens_per_forward` now divides.
        latency = statistics.median(latencies)
        canvas_token_count = len(canonical.full_continuation_token_ids)
        decoded_positions = canonical.decoded_positions
        # Commit forwards used to be folded into `proposer_passes`; they are
        # reported separately now, so add them back explicitly. This total is
        # the NFE every earlier run recorded and must not drift.
        forward_passes = (
            canonical.proposer_passes
            + canonical.commit_passes
            + canonical.verifier_forward_passes
            + int(canonical.extras.get("remask_forward_passes", 0))
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
            "decoded_positions": decoded_positions,
            "decoded_positions_per_second": (
                decoded_positions / latency if latency > 0 else float("nan")
            ),
            "stop_token_id": canonical.stop_token_id,
            "early_stopped": canonical.early_stopped,
            # NFE in the diffusion-LM literature: model forward calls. Tokens
            # per forward is 1 for an autoregressive model by construction, so
            # it is the metric that says what the parallelism bought.
            "model_forward_passes": forward_passes,
            "tokens_per_forward": (
                decoded_positions / forward_passes if forward_passes else float("nan")
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
            "peak_gpu_memory_bytes_per_device": [
                max(values) for values in zip(*device_peaks)
            ],
            "verifier_peak_gpu_memory_bytes": max(verifier_peaks) if verifier_peaks else None,
            **canonical.extras,
            "mean_committed_tokens_per_step": committed_count / decision_count,
            "commitment_histogram": canonical.commitment_histogram,
            "max_candidates": getattr(config, "max_candidates", None),
            "maximum_length_termination": canonical.maximum_length_termination,
            "early_eos": early_eos,
            "traced": should_trace,
            "position_order": getattr(config, "forward_selection", "confidence"),
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
    parser.add_argument("--allow-block-mismatch", action="store_true",
                        help="decode an SDAR checkpoint at a block length other "
                        "than the one its name says it was trained with "
                        "(out-of-distribution ablations only)")
    parser.add_argument("--family", default="dream", choices=sorted(FAMILIES),
                        help="calling conventions for the checkpoint (logits "
                             "alignment and attention mask). Getting this wrong "
                             "still produces fluent text; see sdar_harness/model.py.")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", nargs="+",
                        help="pipeline-split the model over these cards, e.g. "
                        "cuda:0 cuda:1 (overrides --device). For checkpoints "
                        "that do not fit one card, such as SDAR-8B in BF16 on "
                        "a 16 GB A4000.")
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
        help="decompose latency into proposer / verifier / commit "
        "for the first N tasks of each method. Runs one extra "
        "generation per profiled task and is OFF by default, because the "
        "measurement must never run inside the timed repeats it explains.",
    )
    parser.add_argument("--max-candidates", type=int, default=8,
                        help="cap on tokens proposed (and so committed) per step")
    for kind, default_conf in (("fixed", "logp"), ("adaptive", "logp"), ("dynamic", "logp")):
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
    parser.add_argument("--vocab-limit", type=int,
                        help="read logits only over token IDs below this; 151665 "
                        "is the range SDAR and Dream share (DESIGN.md F4)")
    parser.add_argument("--stop-token-ids", type=int, nargs="+",
                        help="tokens that end the answer. Default: the "
                        "checkpoint's generation_config eos_token_id plus the "
                        "tokenizer EOS (DESIGN.md F2)")
    parser.add_argument("--verifier-model",
                        help="checkpoint that verifies and remasks for xadaptive "
                        "methods (Dream-v0-Instruct-7B in exp05)")
    parser.add_argument("--verifier-family", default="dream", choices=sorted(FAMILIES))
    parser.add_argument("--verifier-device", default="cuda:1")
    parser.add_argument("--verifier-devices", nargs="+",
                        help="pipeline-split the verifier over these cards "
                        "(SDAR-8B as the xadaptive verifier); overrides --verifier-device")
    parser.add_argument("--veto", action="store_true",
                        help="xadaptive only: let Dream veto SDAR's top candidate. "
                        "Off by default -- exp05 runs without it (decided 2026-09-19)")
    parser.add_argument("--vgb-budget", type=int, default=512,
                        help="vgb only: moves per block before forward-only completion")
    parser.add_argument("--vgb-row-chunk", type=int, default=16,
                        help="vgb only: leave-one-out rows per forward (memory bound, not math)")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="fill the whole canvas before truncating, as exp04 did")
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
        for kind in ("fixed", "adaptive", "dynamic")
    }
    configs = [
        parse_method(
            value,
            args.continuation_length,
            args.verifier_microbatch_size,
            max_candidates=args.max_candidates,
            sampling=sampling,
            early_stop=not args.no_early_stop,
            veto=args.veto,
            vgb_budget=args.vgb_budget,
            vgb_row_chunk=args.vgb_row_chunk,
        )
        for value in args.methods
    ]
    if len({config.label for config in configs}) != len(configs):
        parser.error("duplicate methods")
    needs_verifier = any(
        isinstance(config, CrossConfig)
        or (isinstance(config, VGBConfig) and config.needs_reference)
        for config in configs
    )
    if needs_verifier and (args.verifier_model is None or args.vocab_limit is None):
        parser.error("xadaptive and VGB methods need --verifier-model and --vocab-limit")
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
            devices=tuple(args.devices) if args.devices else None,
            dtype=args.dtype, cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            require_lean_logits=args.require_lean_logits,
            vocab_limit=args.vocab_limit,
            stop_token_ids=tuple(args.stop_token_ids) if args.stop_token_ids else None,
            allow_block_mismatch=args.allow_block_mismatch,
        )
    )
    verifier = None
    if needs_verifier:
        verifier = MaskedDiffusionProvider.from_pretrained(
            ModelOptions(
                model_id=args.verifier_model, family=args.verifier_family,
                device=args.verifier_device, dtype=args.dtype,
                devices=tuple(args.verifier_devices) if args.verifier_devices else None,
                # A block-diffusion verifier must use the proposer's block grid.
                block_length=(args.block_length
                              if FAMILIES[args.verifier_family].block_length is not None else None),
                cache_dir=args.cache_dir, local_files_only=args.local_files_only,
                require_lean_logits=True, vocab_limit=args.vocab_limit,
                allow_block_mismatch=args.allow_block_mismatch,
            )
        )
        assert_shared_vocabulary(provider.tokenizer, verifier.tokenizer, args.vocab_limit)
    run_metadata = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "benchmark": benchmark,
        "number_of_tasks": len(rows),
        "example_ids": [str(row["example_id"]) for row in rows],
        "model": args.model,
        "family": args.family,
        "block_length": provider.conventions.block_length,
        "trained_block_length": trained_block_length(args.model),
        "revision": args.revision,
        "dtype": args.dtype,
        "device": args.device,
        "devices": [str(device) for device in provider.devices],
        "layer_split": provider.layer_split,
        "physical_gpu": os.environ.get("EXPERIMENT_PHYSICAL_GPU_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "host": platform.node(),
        "seed": args.seed,
        "methods": [config.label for config in configs],
        "stop_token_ids": sorted(provider.stop_token_ids),
        "vocab_limit": provider.vocab_limit,
        "early_stop": not args.no_early_stop,
        "verifier_model": args.verifier_model if needs_verifier else None,
        "verifier_device": args.verifier_device if needs_verifier else None,
        "verifier_family": args.verifier_family if needs_verifier else None,
        "verifier_devices": [str(device) for device in verifier.devices] if verifier is not None else None,
        "verifier_layer_split": verifier.layer_split if verifier is not None else None,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    worker_label = os.environ.get("EXPERIMENT_PHYSICAL_GPU_ID", str(os.getpid()))
    # One file per (worker, method set): a worker that runs several invocations
    # into the same benchmark directory must not overwrite its earlier record.
    method_tag = "+".join(config.label for config in configs)
    write_json(
        args.output_root / f"run_metadata_worker_{worker_label}__{method_tag}.json", run_metadata
    )
    for config in configs:
        run_method(
            provider, rows, config, args.output_root,
            timing_repeats=args.timing_repeats,
            trace_limit=args.trace_limit,
            profile_limit=args.profile_stages,
            use_chat_template=not args.no_chat_template,
            overwrite=args.overwrite,
            base_seed=args.seed,
            verifier=verifier,
        )


if __name__ == "__main__":
    main()
