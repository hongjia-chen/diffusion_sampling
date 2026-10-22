"""Per-call cost of the MI verifier vs rows per call and context length.

Settles the group-size question for the B=32 MI run (README "Deferred",
2026-10-15): at B=64 the 1.7B verifier scores 4 rows of 64 tokens per call;
at B=32 the same 256 tokens per call is 8 rows of 32. Whether that costs the
same depends on what grows with context: SDAR's attention concatenates the
cached prefix onto every row (`ExpandedCache` is a stride-0 view, the `cat` is
not), so part of the cost may scale with rows rather than tokens.

Times exactly the `verify_forward` stage of `cross.dream_verify`: one
`_forward_slices` call through `BlockCacheProvider` on a committed prefix of
`context` tokens, with the first group's two gathers (conditional rows and the
row-0 marginals over every candidate). Token values do not affect cost, so the
prefix is random ids. Cells are interleaved over rounds so slow drift (host
load, clocks) spreads over all of them; each timed call is bracketed by CUDA
syncs. The contention guard watches the card throughout.

    CUDA_VISIBLE_DEVICES=3 python bench_verifier_rows.py \
        --model ../models/SDAR-1.7B-Chat-b32-local ../models/SDAR-1.7B-Chat-b64-local
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path

import torch

from sdar_harness.blockwise import BlockCacheProvider
from sdar_harness.contention import ContentionGuard
from sdar_harness.model import MaskedDiffusionProvider, ModelOptions, block_causal_mask, trained_block_length
from sdar_harness.scheduler import _forward_slices


def build_prefix(provider, block: int, context: int, generator: torch.Generator, chunk: int):
    """A verifier view of one block after `context` committed tokens.

    The prefix is committed `chunk` tokens at a time, as the decoder commits block by block: one
    4096-token call runs the 8B out of memory in attention on a 22 GiB card. Only the cache is
    built this way; the timed call is unchanged."""
    device = provider.device
    total = context + block
    ids = torch.randint(0, 150000, (1, total), generator=generator).to(device)
    attention = block_causal_mask(total, block, device=device)
    positions = torch.arange(total, device=device)
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    for low in range(0, context, chunk):
        high = min(low + chunk, context)
        provider.commit(ids[:, low:high], attention[low:high, :high], positions[low:high], cache)
    return BlockCacheProvider(provider, cache, attention[context:total, :total], positions[context:total])


def build_probes(provider, block: int, rows: int, generator: torch.Generator):
    """A first group of `rows` probes: row r fills the first r candidates of an
    all-masked block, as dream_verify's probe rows do, plus its two gathers."""
    device = provider.device
    state = torch.full((1, block), provider.mask_token_id, dtype=torch.long, device=device)
    candidates = torch.randperm(block, generator=generator)[:rows].to(device)
    tokens = torch.randint(0, 150000, (rows,), generator=generator).to(device)
    probes = state.repeat(rows, 1)
    for row in range(1, rows):
        probes[row, candidates[:row]] = tokens[:row]
    gathers = [
        (torch.arange(rows, device=device), candidates),
        (torch.zeros_like(candidates), candidates),
    ]
    return probes, gathers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 512, 1024, 2048, 4096])
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--reps", type=int, default=8, help="timed calls per cell per round")
    parser.add_argument("--warmup", type=int, default=3, help="untimed calls per cell per round")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--vocab-limit", type=int, default=151936)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--prefill-chunk", type=int, default=512,
                        help="tokens per commit call when building each prefix (a multiple of the block)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = "cuda:0"
    uuid = f"GPU-{torch.cuda.get_device_properties(0).uuid}"
    guard = ContentionGuard([uuid])
    print(f"card {os.environ.get('CUDA_VISIBLE_DEVICES')} {uuid}", flush=True)
    guard.wait_until_clear()
    results = []
    with guard.watch() as watch:
        for model_id in args.model:
            block = trained_block_length(model_id)
            provider = MaskedDiffusionProvider.from_pretrained(ModelOptions(
                model_id=model_id, family="sdar", block_length=block, device=device,
                dtype=args.dtype, local_files_only=True, require_lean_logits=True,
                vocab_limit=args.vocab_limit,
            ))
            generator = torch.Generator().manual_seed(args.seed)
            cells = [(c, r) for c in args.contexts for r in args.rows if r <= block]
            if args.prefill_chunk % block:
                raise ValueError(f"--prefill-chunk must be a multiple of the block ({block})")
            views = {c: build_prefix(provider, block, c, generator, args.prefill_chunk) for c in args.contexts}
            cases = {(c, r): (views[c], *build_probes(provider, block, r, generator)) for c, r in cells}
            times = {cell: [] for cell in cells}
            loads = []
            order = random.Random(args.seed)
            for round_index in range(args.rounds):
                order.shuffle(cells)
                for cell in cells:
                    view, probes, gathers = cases[cell]
                    with torch.inference_mode():
                        for _ in range(args.warmup):
                            _forward_slices(view, probes, gathers)
                        torch.cuda.synchronize()
                        for _ in range(args.reps):
                            torch.cuda.synchronize()
                            began = time.perf_counter()
                            _forward_slices(view, probes, gathers)
                            torch.cuda.synchronize()
                            times[cell].append(time.perf_counter() - began)
                loads.append(os.getloadavg()[0])
                print(f"{Path(model_id).name} round {round_index + 1}/{args.rounds} done, load {loads[-1]:.0f}", flush=True)
            for (context, rows), values in sorted(times.items()):
                ms = [1000 * v for v in values]
                results.append({
                    "model": Path(model_id).name, "block": block, "context": context, "rows": rows,
                    "tokens_per_call": rows * block, "n": len(ms),
                    "median_ms": statistics.median(ms), "p10_ms": sorted(ms)[len(ms) // 10],
                    "p90_ms": sorted(ms)[(9 * len(ms)) // 10], "min_ms": min(ms),
                })
                r = results[-1]
                print(f"  B={block:2d} ctx={context:5d} rows={rows:2d} ({rows * block:4d} tok): "
                      f"median {r['median_ms']:6.1f} ms  p10 {r['p10_ms']:6.1f}  p90 {r['p90_ms']:6.1f}", flush=True)
            del cases, views, provider
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "card": uuid, "host": os.uname().nodename,
        "torch": torch.__version__, "started_load": loads[0] if loads else None,
        **watch.record(), "results": results,
    }, indent=1))
    print(f"contended: {watch.contended}  host load {watch.load_before:.0f} -> {watch.load_after:.0f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
