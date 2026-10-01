#!/usr/bin/env python3
"""Single-GPU gate: will the real run fit in memory, and did the lean path engage?

Loads the model once and exercises the worst-case memory moment -- the first
denoising step, where every canvas position is still masked and the proposer
scores all of them -- for both a fixed and an adaptive config. It does not run a
full generation, so it finishes in about a minute instead of hours while still
touching peak memory.

Run this before committing three GPUs to a smoke test:

    CUDA_VISIBLE_DEVICES=0 python3 scripts/preflight.py

Exit status is 0 only if the model loaded, the memory-lean read path engaged,
and both configs completed a step without running out of memory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

# Deliberately NOT setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
# Expandable segments allocate through CUDA's virtual-memory APIs, which run a
# peer-to-peer access check that queries NVML. On a host whose NVML is broken
# -- the same fault that stops nvidia-smi from running -- that check fails and
# aborts `model.to(device)` in PeerToPeerAccess.cpp before sampling starts.
# Set it explicitly in the environment if NVML is healthy and fragmentation is
# a problem; it must stay off otherwise.

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive_re.model import FAMILIES, MaskedDiffusionProvider, ModelOptions
from adaptive_re.scheduler import GenerationConfig, generation_step


GB = 1024 ** 3


def report_memory(device: torch.device, label: str) -> None:
    if device.type != "cuda":
        return
    free, total = torch.cuda.mem_get_info(device)
    print(
        f"  {label:<26} free {free / GB:6.2f} GB / total {total / GB:6.2f} GB"
        f"   (used {(total - free) / GB:6.2f} GB)"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--family", default="dream", choices=sorted(FAMILIES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", nargs="+",
                        help="pipeline-split the model over these cards (overrides --device)")
    parser.add_argument("--block-length", type=int)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--continuation-length", type=int, default=512)
    parser.add_argument("--prompt-length", type=int, default=76)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument(
        "--microbatch-sweep",
        default="8,4,2,1",
        help="verifier microbatch sizes to try, largest first",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="frozen dataset jsonl; when given, sizes the probe from its "
        "longest prompt instead of --prompt-length",
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    all_devices = [torch.device(name) for name in (args.devices or [args.device])]
    # Sampler state lives where the model's output comes out: the last card.
    device = all_devices[-1]
    print(f"\ndevice        {device}  ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'})")
    print(f"torch         {torch.__version__}")
    print(f"canvas        {args.continuation_length} tokens")
    print(f"alloc conf    {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}\n")

    for card in all_devices:
        report_memory(card, f"before load {card}")

    print("\nloading model (this is the slow part)...")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        provider = MaskedDiffusionProvider.from_pretrained(
            ModelOptions(
                model_id=args.model,
                family=args.family,
                device=args.device,
                devices=tuple(args.devices) if args.devices else None,
                block_length=args.block_length,
                dtype=args.dtype,
                cache_dir=args.cache_dir,
                local_files_only=args.local_files_only,
            )
        )
        lean_warnings = [str(w.message) for w in caught if "lean" in str(w.message).lower()]

    for card in all_devices:
        report_memory(card, f"after load {card}")
    if provider.layer_split:
        print(f"  layer split          {provider.layer_split}")

    lean = provider.supports_lean_logits
    print(f"\nmemory-lean read path: {'ENGAGED' if lean else 'NOT ENGAGED'}")
    if hasattr(provider, "lean_path_relative_gap"):
        print(f"  self-check relative gap {provider.lean_path_relative_gap:.3g}")
    for message in lean_warnings:
        print(f"  ! {message}")
    if not lean:
        print(
            "\nFAIL: the sampler will materialize [batch, sequence, vocabulary]\n"
            "logits and almost certainly run out of memory. Investigate the\n"
            "warning above before starting a run."
        )
        return 1

    prompt_length = args.prompt_length
    if args.dataset is not None:
        # A long prompt costs memory in every activation, so the gate is only
        # meaningful at the longest prompt the run will actually see.
        lengths = []
        with args.dataset.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                text = provider.render_prompt(json.loads(line)["prompt"])
                lengths.append(int(provider.encode_prompt(text).shape[1]))
        if lengths:
            prompt_length = max(lengths)
            print(
                f"\ndataset {args.dataset.name}: {len(lengths)} prompts, "
                f"longest {prompt_length} tokens (median {sorted(lengths)[len(lengths) // 2]})"
            )

    block = provider.conventions.block_length
    if block is not None:
        # A block-causal forward needs a whole number of blocks. The full-canvas
        # step below is a conservative bound for block decoding, which only
        # ever runs one block against a cached prefix.
        total = -(-(prompt_length + args.continuation_length) // block) * block
        args.continuation_length = total - prompt_length

    # Worst-case step: the whole canvas is still masked.
    prompt = torch.randint(0, 1000, (1, prompt_length), dtype=torch.long, device=device)
    canvas = torch.full(
        (1, args.continuation_length), provider.mask_token_id, dtype=torch.long, device=device
    )
    state = torch.cat((prompt, canvas), dim=1)

    def attempt(config: GenerationConfig) -> tuple[bool, str]:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            for card in all_devices:
                torch.cuda.reset_peak_memory_stats(card)
                torch.cuda.synchronize(card)
        try:
            step = generation_step(
                provider, state, canvas_start=prompt_length,
                config=config, step_index=0, capture_trace=False,
            )
        except RuntimeError as error:
            # Where NVML is broken, a genuine OOM surfaces as an NVML assert
            # rather than OutOfMemoryError, because the allocator's own OOM
            # reporting path calls into NVML and that call fails too. But an
            # NVML assert on its own does NOT mean OOM -- the same assert fires
            # from PeerToPeerAccess.cpp when expandable segments are enabled on
            # such a host. Classify by call site, not by the NVML mention.
            text = str(error)
            lowered = text.lower()
            if "out of memory" in lowered or "cudacachingallocator" in lowered:
                kind = "OUT OF MEMORY"
            elif "nvml" in lowered:
                kind = "NVML FAILURE (not an OOM)"
            else:
                kind = "FAILED"
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return False, f"{kind}: {text.splitlines()[0][:90]}"
        if device.type == "cuda":
            parts = []
            for card in all_devices:
                torch.cuda.synchronize(card)
                peak = torch.cuda.max_memory_allocated(card) / GB
                free, _ = torch.cuda.mem_get_info(card)
                parts.append(f"{card}: peak {peak:5.2f} GB, free {free / GB:5.2f} GB")
            detail = "; ".join(parts)
        else:
            detail = "no cuda memory stats"
        return True, f"OK   committed {step.accepted_prefix_length} token(s), {detail}"

    print(f"\nworst-case first denoising step ({args.continuation_length} masked positions,"
          f" {prompt_length}-token prompt):")

    ok, detail = attempt(
        GenerationConfig(
            method="fixed", continuation_length=args.continuation_length, fixed_k=1,
            max_candidates=args.max_candidates, verifier_microbatch_size=1,
        )
    )
    print(f"  {'fixed_k_1':<24} {detail}")
    if not ok:
        print("\nFAIL: the proposer alone does not fit. This model does not fit on\n"
              "this card with this canvas length; nothing below will help.")
        return 1

    print("\nverifier microbatch sweep (adaptive only; fixed-k never verifies):")
    sizes = [int(v) for v in args.microbatch_sweep.split(",") if v.strip()]
    largest_ok = None
    for size in sizes:
        ok, detail = attempt(
            GenerationConfig(
                method="adaptive", continuation_length=args.continuation_length, delta=0.01,
                max_candidates=args.max_candidates, verifier_microbatch_size=size,
            )
        )
        print(f"  microbatch {size:<12} {detail}")
        if ok and largest_ok is None:
            largest_ok = size
            break

    if largest_ok is None:
        print("\nFAIL: the verifier does not fit at any microbatch size tried.\n"
              "Options: shorten --continuation-length, or shard the model.\n"
              "If the rows above say NVML FAILURE rather than OUT OF MEMORY, the\n"
              "problem is the driver, not memory -- check PYTORCH_CUDA_ALLOC_CONF\n"
              "is unset, since expandable_segments needs a working NVML.")
        return 1

    print(f"\nPASS. Use verifier microbatch {largest_ok}:")
    print(f"  MICROBATCH_GSM8K={largest_ok} TASKS=1 bash run_all.sh")
    print("\nKeep this value identical for every adaptive config and for the final\n"
          "run -- it changes the reported verifier_forward_passes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
