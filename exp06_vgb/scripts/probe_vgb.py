#!/usr/bin/env python3
"""Feasibility probe for density-ratio VGB on SDAR-8B (target) / SDAR-1.7B (reference).

Part A (--timing): wall time of one leave-one-out scorer call at several row
counts, per model, against a real GSM8K prefix. This is the per-step cost.

Part B (--walk): run the chain on the first decode blocks of a few GSM8K
prompts under several configs and record how it behaves -- steps to the first
leaf, remasks, budget hits, and the per-step log(B~/F~) = log mean rho^lambda
that sets the depth drift.

    CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=.:../harness:../harness/vendor_env/tf452 \
        python scripts/probe_vgb.py --timing --walk
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from sdar_harness.blockwise import BlockCacheProvider
from sdar_harness.io import stream_jsonl
from sdar_harness.model import MaskedDiffusionProvider, ModelOptions, block_causal_mask
from sdar_harness.scoring import score_record
from sdar_harness.vgb import block_scorer, generate_vgb, parse_vgb_spec

HERE = Path(__file__).resolve().parent.parent


def load(path: Path, devices: list[str], vocab_limit: int) -> MaskedDiffusionProvider:
    return MaskedDiffusionProvider.from_pretrained(ModelOptions(
        model_id=str(path), family="sdar", block_length=64, devices=tuple(devices),
        dtype="bfloat16", local_files_only=True, require_lean_logits=True,
        vocab_limit=vocab_limit,
    ))


def sync(*models) -> None:
    for model in models:
        for device in model.devices:
            torch.cuda.synchronize(device)


def timing(target, reference, prompt_ids, rows_list, repeats=3) -> list[dict]:
    from transformers.cache_utils import DynamicCache

    block = 64
    length = int(prompt_ids.shape[1])
    total = ((length // block) + 1) * block
    canvas = torch.full((total,), target.mask_token_id, dtype=torch.long, device=target.device)
    canvas[:length] = prompt_ids[0].to(target.device)
    low = (length // block) * block
    attention = block_causal_mask(total, block, device=target.device)
    positions = torch.arange(total, device=target.device)
    out = []
    for name, model in (("target_8b", target), ("reference_1p7b", reference)):
        cache = DynamicCache()
        if low:
            model.commit(canvas[:low].unsqueeze(0).to(model.device),
                         attention[:low, :low].to(model.device), positions[:low].to(model.device), cache)
        scope = BlockCacheProvider(model, cache, attention[low:, :].to(model.device),
                                   positions[low:].to(model.device))
        score = block_scorer(scope, model.excluded_token_ids)
        state = canvas[low:].clone()
        for rows in rows_list:
            states = state.unsqueeze(0).repeat(rows, 1)
            index = torch.arange(rows, device=state.device)
            pos = (index % block).clamp(min=length - low)
            score(states, index, pos)  # warm
            times = []
            for _ in range(repeats):
                sync(target, reference)
                start = time.perf_counter()
                score(states, index, pos)
                sync(target, reference)
                times.append(time.perf_counter() - start)
            row = {"model": name, "rows": rows, "prefix": low, "seconds": statistics.median(times),
                   "peak_gb": [torch.cuda.max_memory_allocated(d) / 2**30 for d in model.devices]}
            print(json.dumps(row), flush=True)
            out.append(row)
    return out


def walk(target, reference, prompts, configs, seed, rows, references, small_references, sink) -> list[dict]:
    out = []
    for config in configs:
        for index, prompt_ids in enumerate(prompts):
            generator = torch.Generator(device=target.device).manual_seed(seed + index)
            sync(target, reference)
            start = time.perf_counter()
            result = generate_vgb(target, reference if config.needs_reference else None,
                                  prompt_ids, config, generator=generator)
            sync(target, reference)
            seconds = time.perf_counter() - start
            drift = [r["log_backward"] - r["log_forward"] for r in result.traces
                     if r["log_backward"] != float("-inf") and r["k"] > 0]
            row = {
                "config": config.label, "prompt": index, "seconds": seconds,
                "totals": result.extras["vgb"],
                "blocks": [{key: b[key] for key in ("block_index", "n", "steps", "backward_moves",
                                                    "stays", "switches", "forced", "target_rows",
                                                    "reference_rows", "excursions", "changed_vs_draft")}
                           for b in result.extras["vgb_blocks"]],
                "log_mean_rho_lam": {
                    "median": statistics.median(drift) if drift else None,
                    "p10": sorted(drift)[len(drift) // 10] if drift else None,
                    "p90": sorted(drift)[9 * len(drift) // 10] if drift else None,
                },
                "log_value_end": [r["log_value"] for r in result.traces][-1:] if config.tracks_value else None,
                "remasked": [b["remasked"] for b in result.extras["vgb_blocks"]],
                "text": target.decode(result.continuation_token_ids),
            }
            task = rows[index]
            row["example_id"] = str(task["example_id"])
            row["score"] = score_record(task, row["text"]).get("score")
            ref = references.get(row["example_id"])
            if ref is not None:
                row["fixed_k1_score"] = ref.get("score")
                row["matches_fixed_k1"] = row["text"] == ref.get("raw_completion")
            small_ref = small_references.get(row["example_id"])
            if small_ref is not None:
                row["small_fixed_k1_score"] = small_ref.get("score")
                row["matches_small_fixed_k1"] = row["text"] == small_ref.get("raw_completion")
            print(json.dumps({k: v for k, v in row.items() if k not in ("remasked", "text")}), flush=True)
            with open(sink, "a") as handle:
                handle.write(json.dumps(row) + "\n")
            out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--walk", action="store_true")
    parser.add_argument("--target", default=str(HERE / "../models/SDAR-8B-Chat-b64-local"))
    parser.add_argument("--reference", default=str(HERE / "../models/SDAR-1.7B-Chat-b64-local"))
    parser.add_argument("--target-devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--reference-device", default="cuda:2")
    parser.add_argument("--dataset", default=str(HERE / "../data/gsm8k/test.jsonl"))
    parser.add_argument("--prompts", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--budget", type=int, default=400)
    parser.add_argument("--configs", nargs="+",
                        default=["fwdonly", "lam0.5", "lam1", "lam0.5:mom", "lam1:mom", "lam0.5:exact"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--from-records",
                        help="records.jsonl of a finished arm: take its first --prompts task IDs, "
                        "in order, and compare against its completions")
    parser.add_argument("--small-records",
                        help="the 1.7B arm's records.jsonl, to compare against as well")
    parser.add_argument("--task-ids", nargs="+", help="run exactly these tasks, in this order")
    parser.add_argument("--output", default=str(HERE / "outputs/vgb_probe/probe.json"))
    args = parser.parse_args()

    target = load(Path(args.target), args.target_devices, 151665)
    reference = load(Path(args.reference), [args.reference_device], 151665)
    rows = list(stream_jsonl(Path(args.dataset)))
    references: dict = {}
    if args.from_records:
        references = {str(r["example_id"]): r for r in stream_jsonl(Path(args.from_records))}
        by_id = {str(r["example_id"]): r for r in rows}
        rows = [by_id[eid] for eid in references]
    small_references: dict = {}
    if args.small_records:
        small_references = {str(r["example_id"]): r for r in stream_jsonl(Path(args.small_records))}
    if args.task_ids:
        by_id = {str(r["example_id"]): r for r in stream_jsonl(Path(args.dataset))}
        rows = [by_id[eid] for eid in args.task_ids]
    rows = rows[: args.prompts]
    prompts = [target.encode_prompt(target.render_prompt(r["prompt"])) for r in rows]
    for row, ids in zip(rows, prompts):
        ref_ids = reference.encode_prompt(reference.render_prompt(row["prompt"]))
        assert torch.equal(ids.cpu(), ref_ids.cpu()), "target and reference prompts differ"

    report: dict = {"args": vars(args)}
    if args.timing:
        report["timing"] = timing(target, reference, prompts[0], [1, 2, 4, 8, 16, 32, 64])
    if args.walk:
        configs = [
            parse_vgb_spec(spec, continuation_length=64 * args.blocks, budget=args.budget,
                           trace=True, early_stop=True, row_chunk=64)
            for spec in args.configs
        ]
        sink = Path(args.output).with_suffix(".jsonl")
        sink.parent.mkdir(parents=True, exist_ok=True)
        report["walk"] = walk(target, reference, prompts, configs, args.seed, rows, references,
                              small_references, sink)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
