#!/usr/bin/env python3
"""Build a local, loadable copy of an SDAR checkpoint without flash_attn.

SDAR's `modeling_sdar.py` has an unguarded top-level

    from flash_attn.ops.triton.layer_norm import rms_norm_fn as flash_rms_norm

so `trust_remote_code` refuses to load it at all when flash_attn is absent --
and flash_attn cannot be built here (no nvcc, and torch 2.13+cu130 has no
prebuilt wheel).

This writes a directory that symlinks the weights and applies two minimal,
documented patches to that one file. Both fall back to code the authors
themselves supply:

  1. RMSNorm. The pure-PyTorch implementation is already in the file, commented
     out directly beneath the flash call, with the authors' own note that it is
     equivalent. The patch guards the import and uses it when flash is absent.

  2. Attention. The file already has both paths -- `flash_attn_func` for an
     all-ones mask and `F.scaled_dot_product_attention` otherwise -- and they
     compute the same exact attention, differing only in kernel and layout. The
     patch takes the flash branch only when flash is importable, so an all-ones
     mask falls through to SDPA with an all-True `attn_mask`, which is the same
     function of the same inputs.

Nothing else is touched: the block-causal masking, the `store_kv` cache
discipline and the logits alignment are all the checkpoint's own.

    python scripts/prepare_sdar_local.py JetLM/SDAR-1.7B-Chat-b64
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

RMS_IMPORT_OLD = (
    "from flash_attn.ops.triton.layer_norm import rms_norm_fn as flash_rms_norm"
)
RMS_IMPORT_NEW = """try:
    from flash_attn.ops.triton.layer_norm import rms_norm_fn as flash_rms_norm
except Exception:  # PATCHED: flash_attn is optional here
    flash_rms_norm = None"""

RMS_FORWARD_OLD = """    def forward(self, hidden_states):
        return flash_rms_norm(
            hidden_states, weight=self.weight, bias=None, eps=self.variance_epsilon)"""
RMS_FORWARD_NEW = """    def forward(self, hidden_states):
        if flash_rms_norm is not None:
            return flash_rms_norm(
                hidden_states, weight=self.weight, bias=None, eps=self.variance_epsilon)
        # PATCHED: the authors' own equivalent, verbatim from the block below.
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * \\
            torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)"""

ATTN_OLD = "        if torch.all(attention_mask):  # decoding"
ATTN_NEW = (
    "        # PATCHED: take the flash path only when flash_attn imported. An\n"
    "        # all-ones mask otherwise falls through to SDPA below, which is the\n"
    "        # same exact attention with attn_mask all True.\n"
    "        if \"flash_attn_func\" in globals() and torch.all(attention_mask):  # decoding"
)

PATCHES = [
    ("rms import", RMS_IMPORT_OLD, RMS_IMPORT_NEW),
    ("rms forward", RMS_FORWARD_OLD, RMS_FORWARD_NEW),
    ("attention branch", ATTN_OLD, ATTN_NEW),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="e.g. JetLM/SDAR-1.7B-Chat-b64")
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "models")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    source = Path(snapshot_download(args.repo, local_files_only=True)).resolve()
    target = args.output_root / (args.repo.split("/")[-1] + "-local")
    if target.exists():
        if not args.force:
            print(f"{target} exists; use --force to rebuild")
            return
        shutil.rmtree(target)
    target.mkdir(parents=True)

    patched_name = "modeling_sdar.py"
    for entry in sorted(source.iterdir()):
        if entry.name == patched_name:
            continue
        # Symlink the weights rather than copying several GB.
        os.symlink(entry.resolve(), target / entry.name)

    text = (source / patched_name).read_text(encoding="utf-8")
    for label, old, new in PATCHES:
        if old not in text:
            raise SystemExit(
                f"patch {label!r} did not match; the checkpoint's modeling file "
                f"has changed and the patches must be re-derived before trusting "
                f"any number produced with it."
            )
        text = text.replace(old, new, 1)
    (target / patched_name).write_text(text, encoding="utf-8")

    (target / "PATCHES.md").write_text(
        f"# Local patched copy of {args.repo}\n\n"
        f"Source snapshot: {source}\n"
        f"Everything is a symlink to that snapshot except `{patched_name}`, which\n"
        f"carries {len(PATCHES)} patches so the model loads without flash_attn:\n\n"
        + "".join(f"{i+1}. {label}\n" for i, (label, _, _) in enumerate(PATCHES))
        + "\nBoth fallbacks are the authors' own code: the pure-PyTorch RMSNorm is\n"
          "commented out in their file directly beneath the flash call, and the SDPA\n"
          "attention path is the branch they already use for prefill. Neither changes\n"
          "the function being computed.\n\n"
          "Regenerate with `python scripts/prepare_sdar_local.py " + args.repo + " --force`.\n",
        encoding="utf-8",
    )
    print(f"wrote {target}")
    for label, _, _ in PATCHES:
        print(f"  patched: {label}")


if __name__ == "__main__":
    main()
