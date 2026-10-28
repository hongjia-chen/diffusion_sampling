"""Attribute the verifier's rows x context cost in SDAR block-diffusion decoding.

`bench_verifier_rows.py` measured it on an A10G (ETA_STUDY.md section 10):
8B-b16 ms ~ 40 + 2.8*rows + 13.3*rows*ctx/1000; 1.7B-b16 ~36 ms launch floor
plus ~6.1 ms per row per 1k ctx. This script asks where the rows x ctx term goes:

  (a) the prefix `torch.cat`. In every denoising forward (store_kv=False), every
      layer concatenates the cached prefix onto the block's K/V. The prefix
      arrives through `ExpandedCache`, a stride-0 view, so the cat materialises
      rows x ctx of K and V per layer per step.
  (b) the attention kernel. SDAR calls SDPA with an all-ones bool mask and
      enable_gqa=True. Per the torch 2.13 docs, GQA runs only in the flash and
      math kernels, and flash takes no mask, so this is presumably the fp32
      math kernel.
  (c) everything else.

How: `SDARAttention.forward`, taken from the loaded model's own module, is
replaced at runtime by a transcription of it with two switches (plus one for
RMSNorm). Nothing in harness/ or models/ is edited, and the original is
restored on exit.

  attention  sdpa_mask    the current behaviour (SDAR's SDPA branch, exact)
             sdpa_nomask  attn_mask=None when the mask is all-ones, still enable_gqa=True
             flash_sync   flash_attn_func exactly as SDAR calls it, including its
                          per-layer `torch.all(mask)` host sync
             flash        the same call, all-ones decided without the sync
  prefix     cat          the current torch.cat
             buffer       per-block K/V buffers [rows_max, kv, ctx+B, hd]: the
                          prefix is copied once per block, and each step writes
                          only the block's B tokens into the tail. Same values
                          and layout as the cat, so same-kernel outputs are
                          bit-identical (checked; the run aborts if not).
  norm       torch        the pure-PyTorch RMSNorm
             triton       flash_attn's triton rms_norm_fn (installing flash_attn
                          switches SDAR to it)
  original                the module's own forward, unpatched (a control for
                          the patch; with flash_attn installed it is
                          flash_sync/cat/triton)

"All-ones" without a sync: under the harness's protocol, a forward whose query
length equals the block length is one aligned block whose keys are sliced to
exactly [0, high). That holds for denoising forwards (`BlockCacheProvider`) and
for single-block commits. Prefill and chunked commits are longer and keep their
mask. The correctness phase asserts this rule against `torch.all` on every
forward it runs.

Phases, per model:
  1. correctness. At --check-cells, a replay of three denoising steps on one
     block, a commit, and a step on the next block. Each variant is compared
     with sdpa_mask/cat/torch (max |logit diff|, argmax agreement). Each buffer
     variant must be torch.equal to its cat twin, and the unpatched module must
     equal its mirror variant bit for bit. The run aborts otherwise.
  2. timing. rows x ctx grid, every (variant, cell) interleaved across rounds as
     in bench_verifier_rows.py. Each call is one `_forward_slices` verify
     forward, bracketed by syncs. Reports median ms per call, the first call of
     each visit (for buffer variants it includes the prefix copy), and peak
     allocated memory over the static baseline.
  3. torch.profiler at --profile-cells. Device time per call is split into the
     labelled prefix and attention regions and the rest. Also reported: which
     SDPA kernel ran, aten::cat time, and the top ops and kernels.

Output: outputs/attention_profile/<tag>.json plus <tag>.txt, a summary table.
CPU works for smoke tests (--device cpu). Run it on one card:

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../harness/vendor_env/tf452:../harness:.. \\
        python profile_attention.py \\
        --model ../models/SDAR-1.7B-Chat-b16-local ../models/SDAR-8B-Chat-b16-local \\
        --tag ec2_card0_b16
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import socket
import statistics
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn.functional as F

from bench_verifier_rows import build_prefix, build_probes
from sdar_harness.blockwise import BlockCacheProvider
from sdar_harness.contention import ContentionGuard, NullGuard
from sdar_harness.model import MaskedDiffusionProvider, ModelOptions, block_causal_mask, trained_block_length
from sdar_harness.scheduler import _forward_slices

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "outputs" / "attention_profile"
ATTENTION = ("sdpa_mask", "sdpa_nomask", "flash_sync", "flash")
PREFIX = ("cat", "buffer")
NORM = ("torch", "triton")
# Cells with rows*ctx at or above this are past the 1.7B's launch floor on an A10G
# (README / ETA_STUDY section 10); the slope fit over them is the GPU-bound one.
GPU_BOUND_ROW_TOKENS = 8192
TOKEN_RANGE = 150000          # bench_verifier_rows' random-id range


# ----------------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Variant:
    attention: str = "sdpa_mask"
    prefix: str = "cat"
    norm: str = "torch"
    original: bool = False

    def __post_init__(self) -> None:
        if self.original:
            return
        if self.attention not in ATTENTION:
            raise ValueError(f"attention {self.attention!r} not in {ATTENTION}")
        if self.prefix not in PREFIX:
            raise ValueError(f"prefix {self.prefix!r} not in {PREFIX}")
        if self.norm not in NORM:
            raise ValueError(f"norm {self.norm!r} not in {NORM}")

    @classmethod
    def parse(cls, text: str) -> "Variant":
        if text == "original":
            return cls(original=True)
        parts = text.split("/")
        if not 2 <= len(parts) <= 3:
            raise ValueError(f"variant {text!r}: expected attention/prefix[/norm] or 'original'")
        return cls(*parts)

    @property
    def name(self) -> str:
        return "original" if self.original else f"{self.attention}/{self.prefix}/{self.norm}"

    @property
    def short(self) -> str:
        if self.original:
            return "original"
        attention = {"sdpa_mask": "mask", "sdpa_nomask": "nomask", "flash_sync": "fsync", "flash": "flash"}
        text = f"{attention[self.attention]}/{'buf' if self.prefix == 'buffer' else 'cat'}"
        return text + ("/tri" if self.norm == "triton" else "")

    @property
    def needs_flash(self) -> bool:
        return not self.original and self.attention.startswith("flash")

    @property
    def needs_triton_norm(self) -> bool:
        return not self.original and self.norm == "triton"

    def twin(self) -> "Variant":
        """The cat variant a buffer variant must equal bit for bit."""
        return replace(self, prefix="cat")


REFERENCE = Variant()  # sdpa_mask/cat/torch: what every run so far used
DEFAULT_VARIANTS = (
    "original",
    "sdpa_mask/cat/torch",
    "sdpa_mask/buffer/torch",
    "sdpa_nomask/cat/torch",
    "sdpa_nomask/buffer/torch",
    "flash_sync/cat/torch",
    "flash/cat/torch",
    "flash/buffer/torch",
    "flash_sync/cat/triton",
)


@dataclass
class Kernels:
    """The optional flash_attn pieces, or why they are missing."""

    flash: Callable | None = None
    rms_norm: Callable | None = None
    flash_error: str | None = None
    rms_norm_error: str | None = None
    version: str | None = None

    def missing(self, variant: Variant) -> str | None:
        if variant.needs_flash and self.flash is None:
            return f"flash_attn_func unavailable ({self.flash_error})"
        if variant.needs_triton_norm and self.rms_norm is None:
            return f"flash_attn triton rms_norm_fn unavailable ({self.rms_norm_error})"
        return None


def load_kernels(module: Any | None = None) -> Kernels:
    """flash_attn_func and rms_norm_fn, preferring the objects SDAR's module imported."""
    kernels = Kernels()
    namespace = vars(module) if module is not None else {}
    kernels.flash = namespace.get("flash_attn_func")
    kernels.rms_norm = namespace.get("flash_rms_norm")
    try:
        import flash_attn

        kernels.version = getattr(flash_attn, "__version__", "unknown")
        if kernels.flash is None:
            kernels.flash = flash_attn.flash_attn_func
    except Exception as error:  # noqa: BLE001 - any import failure means "absent"
        kernels.flash_error = f"{type(error).__name__}: {error}"
    if kernels.rms_norm is None:
        try:
            from flash_attn.ops.triton.layer_norm import rms_norm_fn

            kernels.rms_norm = rms_norm_fn
        except Exception as error:  # noqa: BLE001
            kernels.rms_norm_error = f"{type(error).__name__}: {error}"
    return kernels


def sdar_attention(model: Any) -> tuple[type, Any]:
    """SDARAttention and its module, both taken from the loaded model itself."""
    found = [m for m in model.modules() if type(m).__name__ == "SDARAttention"]
    if not found:
        raise RuntimeError("no SDARAttention module in this model")
    cls = type(found[0])
    if any(type(m) is not cls for m in found):
        raise RuntimeError("SDARAttention instances of more than one class")
    return cls, sys.modules[cls.__module__]


# ----------------------------------------------------------------------------
# The buffer prototype
# ----------------------------------------------------------------------------

@dataclass
class _Slot:
    sources: tuple[torch.Tensor, torch.Tensor]  # held so their addresses cannot be reused
    context: int
    rows: int
    shared: bool
    keys: torch.Tensor
    values: torch.Tensor


class PrefixBuffers:
    """Per-layer K/V buffers that replace the per-step prefix cat.

    The first denoising forward of a block allocates, for each layer,
    [rows_max, kv_heads, ctx + B, head_dim] contiguous K and V. It copies the
    prefix into [:, :, :ctx] once. Every step then writes only the block's B
    new keys into [:rows, :, ctx:] and hands attention the slice [:rows]. That
    slice has the values, shape and strides of `torch.cat([past, new], -2)`, so
    a kernel gets identical inputs.

    A slot belongs to one prefix, identified by the cache tensors' addresses
    and length. The slot holds a reference to them, so the addresses cannot be
    reused while it lives. A DynamicCache commit replaces the layer tensors (a
    cat, never in place), so the next block's first step sees a new address
    and copies again: once per block.

    `verify=True` also computes the cat and raises unless values and strides
    match. It is used by the correctness phase and the tests, not by timing.
    """

    def __init__(self, rows_max: int | None = None, *, verify: bool = False):
        self.rows_max = rows_max
        self.verify = verify
        self.slots: dict[int, _Slot] = {}
        self.prefix_copies = 0        # one per layer per block
        self.peak_bytes = 0

    def nbytes(self) -> int:
        return sum(
            s.keys.numel() * s.keys.element_size() + s.values.numel() * s.values.element_size()
            for s in self.slots.values()
        )

    def clear(self) -> None:
        self.slots.clear()

    @staticmethod
    def _holds(slot: _Slot, past_k: torch.Tensor, past_v: torch.Tensor, new_k: torch.Tensor) -> bool:
        rows, heads, width, dim = new_k.shape
        return (
            slot.sources[0].data_ptr() == past_k.data_ptr()
            and slot.sources[1].data_ptr() == past_v.data_ptr()
            and slot.context == past_k.shape[-2]
            and tuple(slot.keys.shape[1:]) == (heads, slot.context + width, dim)
            and slot.keys.dtype == new_k.dtype
            and slot.keys.device == new_k.device
            and (slot.rows >= rows if slot.shared else slot.rows == rows)
        )

    def _fill(self, layer: int, past_k, past_v, new_k) -> _Slot:
        rows, heads, width, dim = new_k.shape
        context = int(past_k.shape[-2])
        if past_k.dtype != new_k.dtype or past_v.dtype != new_k.dtype:
            raise TypeError("prefix and block K/V dtypes differ; the cat would type-promote")
        shared = past_k.shape[0] == 1 or past_k.stride(0) == 0
        if not shared and past_k.shape[0] != rows:
            raise ValueError(f"per-row prefix of batch {past_k.shape[0]} for {rows} rows")
        total = max(rows, self.rows_max or 0) if shared else rows
        shape = (total, heads, context + width, dim)
        keys = torch.empty(shape, dtype=new_k.dtype, device=new_k.device)
        values = torch.empty(shape, dtype=new_k.dtype, device=new_k.device)
        source_k = past_k[:1].expand(total, -1, -1, -1) if shared else past_k
        source_v = past_v[:1].expand(total, -1, -1, -1) if shared else past_v
        keys[:, :, :context].copy_(source_k)
        values[:, :, :context].copy_(source_v)
        slot = _Slot((past_k, past_v), context, total, shared, keys, values)
        self.slots[layer] = slot
        self.prefix_copies += 1
        self.peak_bytes = max(self.peak_bytes, self.nbytes())
        return slot

    def assemble(self, layer: int, past_k, past_v, new_k, new_v) -> tuple[torch.Tensor, torch.Tensor]:
        rows = int(new_k.shape[0])
        slot = self.slots.get(layer)
        if slot is None or not self._holds(slot, past_k, past_v, new_k):
            self.slots.pop(layer, None)   # free the old block's buffer before allocating
            slot = self._fill(layer, past_k, past_v, new_k)
        keys, values = slot.keys[:rows], slot.values[:rows]
        keys[:, :, slot.context:].copy_(new_k)
        values[:, :, slot.context:].copy_(new_v)
        if self.verify:
            reference_k = torch.cat([past_k, new_k], dim=-2)
            reference_v = torch.cat([past_v, new_v], dim=-2)
            if not (torch.equal(keys, reference_k) and torch.equal(values, reference_v)):
                raise AssertionError(f"layer {layer}: buffer K/V differ from the cat")
            if keys.stride() != reference_k.stride() or values.stride() != reference_v.stride():
                raise AssertionError(
                    f"layer {layer}: buffer strides {keys.stride()} != cat strides {reference_k.stride()}"
                )
        return keys, values


# ----------------------------------------------------------------------------
# The patched forward
# ----------------------------------------------------------------------------

def _no_region(name: str):
    return nullcontext()


def make_forward(
    module: Any,
    variant: Variant,
    *,
    block: int,
    flash: Callable | None,
    buffers: PrefixBuffers | None,
    check_mask: bool = False,
    label: bool = False,
) -> Callable:
    """`SDARAttention.forward` (models/*-local/modeling_sdar.py), transcribed, with switches.

    Everything not named by the variant is the original line for line: the
    projections, q/k norms, RoPE (the module's own `apply_rotary_pos_emb`), the
    store_kv branches, the `.bool()`, the SDPA call and its transpose, the
    reshape and o_proj. `label` wraps the prefix assembly and the attention call
    in profiler ranges; it is used only under torch.profiler.
    """
    apply_rotary = module.apply_rotary_pos_emb
    attention = variant.attention
    region = torch.profiler.record_function if label else _no_region
    if variant.needs_flash and flash is None:
        raise RuntimeError(f"{variant.name} needs flash_attn_func")

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_value=None, cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        bsz, q_len = input_shape
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary(query_states, key_states, cos, sin)

        if past_key_value is not None and kwargs.get("store_kv", False):
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
        elif past_key_value is not None and not kwargs.get("store_kv", False) and len(past_key_value) > self.layer_idx:
            past_key_states, past_value_states = past_key_value[self.layer_idx]
            with region("pa::prefix"):
                if buffers is None:
                    key_states = torch.cat([past_key_states, key_states], dim=-2)
                    value_states = torch.cat([past_value_states, value_states], dim=-2)
                else:
                    key_states, value_states = buffers.assemble(
                        self.layer_idx, past_key_states, past_value_states, key_states, value_states)

        attention_mask = attention_mask.bool() if attention_mask is not None else None
        # One aligned block against exactly its attendable keys (module docstring).
        all_ones = q_len == block
        if check_mask and attention_mask is not None:
            actual = bool(torch.all(attention_mask))
            if actual != all_ones:
                raise AssertionError(
                    f"shape rule says all-ones={all_ones} but the mask says {actual} "
                    f"(q_len {q_len}, mask {tuple(attention_mask.shape)}, block {block})")

        with region("pa::attention"):
            if attention == "flash_sync":
                use_flash = bool(torch.all(attention_mask))      # SDAR's own test: a host sync
            elif attention == "flash":
                use_flash = all_ones
            else:
                use_flash = False
            if use_flash:
                query_states = query_states.transpose(1, 2)
                key_states = key_states.transpose(1, 2)
                value_states = value_states.transpose(1, 2)
                attn_output = flash(
                    query_states,
                    key_states,
                    value_states,
                    causal=False,
                    softmax_scale=self.scaling,
                )
            else:
                mask = None if (attention == "sdpa_nomask" and all_ones) else attention_mask
                attn_output = F.scaled_dot_product_attention(
                    query=query_states,
                    key=key_states,
                    value=value_states,
                    attn_mask=mask,
                    is_causal=False,
                    scale=self.scaling,
                    enable_gqa=True,
                )
                attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    forward.profile_variant = variant.name
    return forward


_ACTIVE: set[type] = set()
_UNSET = object()


@contextmanager
def patched(
    model: Any,
    variant: Variant,
    *,
    block: int,
    kernels: Kernels | None = None,
    buffers: PrefixBuffers | None = None,
    rows_max: int | None = None,
    check_mask: bool = False,
    label: bool = False,
) -> Iterator[PrefixBuffers | None]:
    """Run `model` under `variant`, then restore the original forward and RMSNorm switch.

    Yields the PrefixBuffers in use (None for cat variants and `original`). A
    store this function created is cleared on exit. A store passed in is left
    as it is, for inspection.
    """
    if variant.original:
        yield None
        return
    cls, module = sdar_attention(model)
    if cls in _ACTIVE:
        raise RuntimeError(f"{cls.__name__} is already patched; patches do not nest")
    kernels = kernels if kernels is not None else load_kernels(module)
    missing = kernels.missing(variant)
    if missing:
        raise RuntimeError(f"{variant.name}: {missing}")
    owned = buffers is None and variant.prefix == "buffer"
    store = PrefixBuffers(rows_max) if owned else (buffers if variant.prefix == "buffer" else None)
    forward = make_forward(
        module, variant, block=block, flash=kernels.flash, buffers=store,
        check_mask=check_mask, label=label,
    )
    original_forward = cls.__dict__["forward"]
    original_norm = vars(module).get("flash_rms_norm", _UNSET)
    _ACTIVE.add(cls)
    try:
        cls.forward = forward
        module.flash_rms_norm = kernels.rms_norm if variant.norm == "triton" else None
        yield store
    finally:
        cls.forward = original_forward
        if original_norm is _UNSET:
            vars(module).pop("flash_rms_norm", None)
        else:
            module.flash_rms_norm = original_norm
        _ACTIVE.discard(cls)
        if owned:
            store.clear()


def mirror_of_original(model: Any) -> Variant:
    """The variant the unpatched module computes, given what its module imported."""
    _, module = sdar_attention(model)
    namespace = vars(module)
    return Variant(
        attention="flash_sync" if "flash_attn_func" in namespace else "sdpa_mask",
        prefix="cat",
        norm="triton" if namespace.get("flash_rms_norm") is not None else "torch",
    )


# ----------------------------------------------------------------------------
# Correctness: a replay of one block, a commit, and the next block
# ----------------------------------------------------------------------------

@dataclass
class Replay:
    context: int
    block: int
    before: list[torch.Tensor]      # probe batches on block n
    commit_ids: torch.Tensor        # [1, B], block n as committed
    after: list[torch.Tensor]       # probe batches on block n+1


def make_replay(provider: Any, block: int, context: int, rows: int, generator: torch.Generator) -> Replay:
    """Three steps on a block (rows R, R//2, R, with more of the block filled
    each step), its commit, and one step on the next block."""
    def batch(count: int, filled: int) -> torch.Tensor:
        probes, _ = build_probes(provider, block, count, generator)
        if filled:
            probes[:, :filled] = torch.randint(0, TOKEN_RANGE, (1, filled), generator=generator).to(probes.device)
        return probes

    before = [batch(rows, 0), batch(max(1, rows // 2), block // 4), batch(rows, block // 2)]
    commit_ids = torch.randint(0, TOKEN_RANGE, (1, block), generator=generator).to(provider.device)
    return Replay(context, block, before, commit_ids, [batch(rows, 0)])


def clone_cache(cache: Any) -> Any:
    """A deep copy of a DynamicCache that keeps each layer's strides (the first
    commit stores RoPE's non-contiguous output as it is)."""
    from transformers.cache_utils import DynamicCache

    copy = DynamicCache()
    for layer in range(len(cache)):
        keys, values = cache[layer]
        copy.update(keys.clone(), values.clone(), layer)
    return copy


@torch.inference_mode()
def block_logits(view: BlockCacheProvider, probes: torch.Tensor) -> torch.Tensor:
    """Logits at every block position of every row, [rows, B, V'], on the CPU."""
    hidden = view.hidden_states(probes)
    return view._provider.project(hidden).cpu()


def replay(provider: Any, plan: Replay, prefix_cache: Any) -> list[torch.Tensor]:
    """Run the plan on a copy of `prefix_cache` under whatever forward is installed."""
    cache = clone_cache(prefix_cache)
    device = provider.device
    low, block = plan.context, plan.block
    total = low + 2 * block
    mask = block_causal_mask(total, block, device=device)
    positions = torch.arange(total, device=device)
    here = BlockCacheProvider(provider, cache, mask[low:low + block, :low + block], positions[low:low + block])
    outputs = [block_logits(here, probes) for probes in plan.before]
    provider.commit(plan.commit_ids, mask[low:low + block, :low + block], positions[low:low + block], cache)
    high = low + block
    there = BlockCacheProvider(provider, cache, mask[high:high + block, :high + block], positions[high:high + block])
    outputs += [block_logits(there, probes) for probes in plan.after]
    return outputs


def compare(outputs: list[torch.Tensor], reference: list[torch.Tensor], plan: Replay, mask_id: int) -> dict:
    diffs, agree, total, agree_masked, total_masked = [], 0, 0, 0, 0
    probes = plan.before + plan.after
    for out, ref, batch in zip(outputs, reference, probes):
        diffs.append(float((out.float() - ref.float()).abs().max()))
        same = out.argmax(-1) == ref.argmax(-1)
        masked = batch.cpu() == mask_id
        agree += int(same.sum())
        total += same.numel()
        agree_masked += int(same[masked].sum())
        total_masked += int(masked.sum())
    return {
        "max_abs_logit_diff": max(diffs),
        "max_abs_logit_diff_per_step": diffs,
        "argmax_agree": agree / total,
        "argmax_disagree_count": total - agree,
        "argmax_agree_masked": agree_masked / total_masked if total_masked else None,
        "positions": total,
        "masked_positions": total_masked,
    }


def bitwise(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


# ----------------------------------------------------------------------------
# One loaded model
# ----------------------------------------------------------------------------

def parse_cell(text: str) -> tuple[int, int]:
    rows, _, context = text.lower().partition("x")
    return int(rows), int(context)


class Session:
    """One loaded model, its prefix caches and probe cases, and the phases."""

    def __init__(self, provider: Any, block: int, args: argparse.Namespace, kernels: Kernels, log=print):
        self.provider = provider
        self.model = provider.model
        self.block = block
        self.args = args
        self.kernels = kernels
        self.log = log
        self.device = provider.device
        self.cuda = self.device.type == "cuda"
        self.generator = torch.Generator().manual_seed(args.seed)
        config = self.model.config
        self.shape = {
            "layers": int(config.num_hidden_layers),
            "q_heads": int(config.num_attention_heads),
            "kv_heads": int(config.num_key_value_heads),
            "head_dim": int(getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads),
        }
        self.element_size = next(self.model.parameters()).element_size()
        self.views: dict[int, BlockCacheProvider] = {}
        self.cases: dict[tuple[int, int], tuple] = {}

    # -- setup --------------------------------------------------------------

    def sync(self) -> None:
        if self.cuda:
            torch.cuda.synchronize(self.device)

    def build(self, contexts: list[int], cells: list[tuple[int, int]]) -> None:
        for context in sorted(set(contexts)):
            if context not in self.views:
                self.views[context] = build_prefix(
                    self.provider, self.block, context, self.generator, self.args.prefill_chunk)
        for rows, context in sorted(set(cells)):
            if (rows, context) not in self.cases:
                self.cases[(rows, context)] = (self.views[context], *build_probes(
                    self.provider, self.block, rows, self.generator))
        self.sync()

    def allocated(self) -> int | None:
        return torch.cuda.memory_allocated(self.device) if self.cuda else None

    def buffer_bytes(self, rows: int, context: int) -> int:
        s = self.shape
        rows = max(rows, self.args.buffer_rows_max or 0)
        return 2 * s["layers"] * rows * s["kv_heads"] * (context + self.block) * s["head_dim"] * self.element_size

    def buffer_fits(self, variant: Variant, rows: int, context: int) -> tuple[bool, dict]:
        """Whether a buffer variant's K/V buffers plus a working margin fit on the card.
        Under sdpa_mask the margin bounds the math kernel's fp32 GQA-expanded K and V;
        otherwise one more cat-sized K/V pair. An OOM past this check is caught too."""
        need = self.buffer_bytes(rows, context)
        if not self.cuda:
            return True, {"buffer_bytes": need}
        s = self.shape
        keys = context + self.block
        working = (2 * rows * s["q_heads"] * keys * s["head_dim"] * 4 if variant.attention == "sdpa_mask"
                   else 2 * rows * s["kv_heads"] * keys * s["head_dim"] * self.element_size)
        margin = (512 << 20) + working
        free, _ = torch.cuda.mem_get_info(self.device)
        cached = torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
        return need + margin <= free + cached, {
            "buffer_bytes": need, "margin_bytes": margin, "available_bytes": free + cached}

    def patch(self, variant: Variant, **kwargs):
        return patched(self.model, variant, block=self.block, kernels=self.kernels,
                       rows_max=self.args.buffer_rows_max, **kwargs)

    def release(self) -> None:
        gc.collect()
        if self.cuda:
            torch.cuda.empty_cache()

    # -- 1. correctness -----------------------------------------------------

    def correctness(self, variants: list[Variant], cells: list[tuple[int, int]]) -> list[dict]:
        records = []
        mirror = mirror_of_original(self.model)
        needed = [REFERENCE, mirror]
        for variant in variants:
            if variant.original:
                continue
            if variant.prefix == "buffer":
                needed.append(variant.twin())
            needed.append(variant)
        order = list(dict.fromkeys(needed))
        order.sort(key=lambda v: v.prefix == "buffer")      # cat twins before buffers
        for rows, context in cells:
            plan = make_replay(self.provider, self.block, context, rows, self.generator)
            cache = self.views[context]._cache
            outputs: dict[str, list[torch.Tensor]] = {}
            stats: dict[str, dict] = {}
            for variant in order:
                if variant.prefix == "buffer":
                    fits, info = self.buffer_fits(variant, rows, context + self.block)
                    if not fits:
                        stats[variant.name] = {"status": "skipped_memory", **info}
                        continue
                    store = PrefixBuffers(self.args.buffer_rows_max, verify=True)
                    with self.patch(variant, buffers=store, check_mask=True):
                        outputs[variant.name] = replay(self.provider, plan, cache)
                    stats[variant.name] = {"status": "ok", "prefix_copies": store.prefix_copies,
                                           "buffer_bytes": store.peak_bytes}
                    store.clear()
                else:
                    with self.patch(variant, check_mask=True):
                        outputs[variant.name] = replay(self.provider, plan, cache)
                    stats[variant.name] = {"status": "ok"}
                self.release()
            outputs["original"] = replay(self.provider, plan, cache)   # nothing patched
            stats["original"] = {"status": "ok", "mirror": mirror.name}
            reference = outputs[REFERENCE.name]
            failures = []
            for variant in [Variant(original=True)] + order:
                record = {"variant": variant.name, "rows": rows, "context": context, **stats[variant.name]}
                if variant == REFERENCE:
                    record["reference"] = True
                if record["status"] != "ok":
                    records.append(record)
                    self.log(f"  check {rows:2d}x{context:<5d} {variant.name:26s} {record['status']}")
                    continue
                record.update(compare(outputs[variant.name], reference, plan, self.provider.mask_token_id))
                note = ""
                if variant.original:
                    record["bitwise_equal_mirror"] = bitwise(outputs[variant.name], outputs[mirror.name])
                    note = f"  ==mirror({mirror.name}) {record['bitwise_equal_mirror']}"
                    if not record["bitwise_equal_mirror"]:
                        failures.append(f"original != its mirror {mirror.name} at {rows}x{context}")
                if variant.prefix == "buffer" and not variant.original:
                    record["bitwise_equal_cat"] = bitwise(outputs[variant.name], outputs[variant.twin().name])
                    note = f"  ==cat {record['bitwise_equal_cat']}"
                    if not record["bitwise_equal_cat"]:
                        failures.append(f"{variant.name} != {variant.twin().name} at {rows}x{context}")
                records.append(record)
                self.log(f"  check {rows:2d}x{context:<5d} {variant.name:26s} "
                         f"max|d| {record['max_abs_logit_diff']:.4g}  argmax {100 * record['argmax_agree']:.2f}%"
                         + note)
            del outputs, reference
            self.release()
            if failures:
                raise CorrectnessError("; ".join(failures), records)
        return records

    # -- 2. timing ----------------------------------------------------------

    def run_job(self, variant: Variant, cell: tuple[int, int], warmup: int, reps: int) -> dict:
        rows, context = cell
        view, probes, gathers = self.cases[cell]
        if variant.prefix == "buffer" and not variant.original:
            fits, info = self.buffer_fits(variant, rows, context)
            if not fits:
                return {"status": "skipped_memory", **info}
        result: dict[str, Any] = {"status": "ok"}
        try:
            with self.patch(variant) as store:
                if self.cuda:
                    torch.cuda.reset_peak_memory_stats(self.device)
                static = self.allocated()
                times, first = [], None
                with torch.inference_mode():
                    for index in range(warmup):
                        self.sync()
                        began = time.perf_counter()
                        _forward_slices(view, probes, gathers)
                        self.sync()
                        if index == 0:
                            first = time.perf_counter() - began
                    for _ in range(reps):
                        self.sync()
                        began = time.perf_counter()
                        _forward_slices(view, probes, gathers)
                        self.sync()
                        times.append(time.perf_counter() - began)
                result.update(times=times, first=first, static_bytes=static,
                              peak_over_static_bytes=(torch.cuda.max_memory_allocated(self.device) - static)
                              if self.cuda else None,
                              buffer_bytes=store.peak_bytes if store is not None else 0)
        except torch.OutOfMemoryError as error:
            result = {"status": "oom", "error": str(error).splitlines()[0][:300]}
        finally:
            if variant.prefix == "buffer" or result["status"] != "ok":
                self.release()
        return result

    def timing(self, variants: list[Variant], cells: list[tuple[int, int]]) -> tuple[list[dict], dict]:
        args = self.args
        jobs = [(v, c) for v in variants for c in cells]
        acc = {(v.name, c): {"times": [], "first": [], "peak": [], "static": [], "buffer": 0, "status": "ok"}
               for v, c in jobs}
        dropped: dict = {}
        began = time.perf_counter()
        warm = 0.0
        for variant, cell in jobs:      # untimed pass: one-time costs (JIT, cuBLAS heuristics, allocator)
            result = self.run_job(variant, cell, warmup=1, reps=0)
            if result["status"] != "ok":
                dropped[(variant.name, cell)] = result
                self.log(f"  drop {variant.name} {cell[0]}x{cell[1]}: {result['status']}")
            else:
                warm += result["first"]
        estimate = warm * args.rounds * (args.warmup + args.reps)
        self.log(f"  warm pass {time.perf_counter() - began:.0f} s; timing estimate ~{estimate / 60:.1f} min "
                 f"({len(jobs) - len(dropped)} jobs x {args.rounds} rounds x {args.warmup}+{args.reps} calls)")
        order = random.Random(args.seed)
        active = [job for job in jobs if (job[0].name, job[1]) not in dropped]
        loads = []
        for round_index in range(args.rounds):
            order.shuffle(active)
            for variant, cell in list(active):
                result = self.run_job(variant, cell, args.warmup, args.reps)
                key = (variant.name, cell)
                if result["status"] != "ok":
                    dropped[key] = result
                    active.remove((variant, cell))
                    self.log(f"  drop {variant.name} {cell[0]}x{cell[1]} in round {round_index + 1}: {result['status']}")
                    continue
                entry = acc[key]
                entry["times"] += result["times"]
                entry["first"].append(result["first"])
                entry["peak"].append(result["peak_over_static_bytes"])
                entry["static"].append(result["static_bytes"])
                entry["buffer"] = max(entry["buffer"], result["buffer_bytes"])
            loads.append(os.getloadavg()[0])
            self.log(f"  round {round_index + 1}/{args.rounds} done, {time.perf_counter() - began:.0f} s, "
                     f"load {loads[-1]:.1f}")
        records = []
        for variant, cell in jobs:
            key = (variant.name, cell)
            rows, context = cell
            base = {"variant": variant.name, "rows": rows, "context": context, "tokens_per_call": rows * self.block}
            if key in dropped:
                records.append({**base, **{k: v for k, v in dropped[key].items() if k != "times"}})
                continue
            entry = acc[key]
            ms = sorted(1000 * t for t in entry["times"])
            first = [1000 * t for t in entry["first"]]
            peaks = [p for p in entry["peak"] if p is not None]
            records.append({
                **base, "status": "ok", "n": len(ms),
                "median_ms": statistics.median(ms), "p10_ms": ms[len(ms) // 10],
                "p90_ms": ms[(9 * len(ms)) // 10], "min_ms": ms[0],
                "first_call_median_ms": statistics.median(first),
                "first_call_over_median_ms": statistics.median(first) - statistics.median(ms),
                "peak_over_static_bytes": max(peaks) if peaks else None,
                "static_bytes": entry["static"][0],
                "buffer_bytes": entry["buffer"],
            })
        return records, {"seconds": time.perf_counter() - began, "warm_pass_estimate_seconds": estimate,
                         "host_load_per_round": loads}

    # -- 3. profiler --------------------------------------------------------

    def profile(self, variants: list[Variant], cells: list[tuple[int, int]]) -> list[dict]:
        from torch.profiler import ProfilerActivity, profile, record_function

        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if self.cuda else [])
        records = []
        calls = self.args.profile_calls
        for variant in variants:
            for cell in cells:
                rows, context = cell
                base = {"variant": variant.name, "rows": rows, "context": context, "calls": calls}
                if variant.prefix == "buffer" and not variant.original:
                    fits, info = self.buffer_fits(variant, rows, context)
                    if not fits:
                        records.append({**base, "status": "skipped_memory", **info})
                        continue
                view, probes, gathers = self.cases[cell]
                try:
                    with self.patch(variant, label=True), torch.inference_mode():
                        for _ in range(2):
                            _forward_slices(view, probes, gathers)
                        self.sync()
                        with profile(activities=activities) as prof:
                            for _ in range(calls):
                                with record_function("pa::call"):
                                    _forward_slices(view, probes, gathers)
                                    self.sync()
                    summary = summarize_profile(prof, calls)
                except torch.OutOfMemoryError as error:
                    records.append({**base, "status": "oom", "error": str(error).splitlines()[0][:300]})
                    self.release()
                    continue
                records.append({**base, "status": "ok", **summary})
                self.log(f"  profile {variant.name:26s} {rows:2d}x{context:<5d} device {summary['device_ms']:.1f} ms "
                         f"(prefix {fmt(summary['prefix_ms'])}, attention {fmt(summary['attention_ms'])}) "
                         f"kernel {summary['attention_kernel']}")
                if variant.prefix == "buffer":
                    self.release()
        return records


class CorrectnessError(RuntimeError):
    def __init__(self, message: str, records: list[dict]):
        super().__init__(message)
        self.records = records


# ----------------------------------------------------------------------------
# Profiler summary
# ----------------------------------------------------------------------------

SDPA_OPS = {
    "aten::_scaled_dot_product_flash_attention": "torch_flash",
    "aten::_scaled_dot_product_efficient_attention": "torch_mem_efficient",
    "aten::_scaled_dot_product_cudnn_attention": "torch_cudnn",
    "aten::_scaled_dot_product_attention_math": "math",
    "aten::_scaled_dot_product_flash_attention_for_cpu": "cpu_flash",
}


def _device_total(event) -> float:
    return float(getattr(event, "device_time_total", 0.0) or 0.0)


def _self_device(event) -> float:
    return float(getattr(event, "self_device_time_total", 0.0) or 0.0)


def _subtree_self_device(events: list, into: dict) -> None:
    for event in events:
        if not event.name.startswith("pa::"):
            slot = into.setdefault(event.name, [0, 0.0])
            slot[0] += 1
            slot[1] += _self_device(event)
        _subtree_self_device(list(getattr(event, "cpu_children", []) or []), into)


def summarize_profile(prof: Any, calls: int, top: int = 15) -> dict:
    """Per-call device time split by the labelled regions, the SDPA dispatch, top ops and kernels.

    Times are summed from the kernels each CPU op launched (Kineto
    correlation), so "device_ms" is GPU busy time, not wall time; "wall_ms" is
    the CPU span of each call including its closing sync, inflated by profiler
    overhead.
    """
    from torch.autograd import DeviceType

    events = list(prof.events())
    cpu = [e for e in events if e.device_type == DeviceType.CPU]
    device = [e for e in events if e.device_type != DeviceType.CPU
              and not getattr(e, "is_user_annotation", False) and not e.name.startswith("pa::")]

    def region(name: str) -> tuple[float | None, list]:
        hits = [e for e in cpu if e.name == name]
        if not hits:
            return None, hits
        return sum(_device_total(e) for e in hits) / calls / 1000, hits

    call_ms, call_events = region("pa::call")
    prefix_ms, prefix_events = region("pa::prefix")
    attention_ms, attention_events = region("pa::attention")
    wall_ms = sum(e.cpu_time_total for e in call_events) / calls / 1000 if call_events else None

    ops: dict[str, list] = {}
    for event in cpu:
        if event.name.startswith("pa::") or getattr(event, "is_user_annotation", False):
            continue
        slot = ops.setdefault(event.name, [0, 0.0, 0.0])
        slot[0] += 1
        slot[1] += _self_device(event)
        slot[2] += _device_total(event)
    kernels: dict[str, list] = {}
    for event in device:
        slot = kernels.setdefault(event.name, [0, 0.0])
        slot[0] += 1
        slot[1] += _device_total(event)

    dispatch = {kind: ops[name][0] / calls for name, kind in SDPA_OPS.items() if name in ops}
    flash_ops = {name: slot[0] / calls for name, slot in ops.items() if name.startswith("flash_attn")}
    if flash_ops:
        dispatch["flash_attn"] = sum(flash_ops.values())
    elif any("flash_fwd" in name and "pytorch_flash" not in name for name in kernels):
        dispatch["flash_attn"] = sum(slot[0] for name, slot in kernels.items()
                                     if "flash_fwd" in name and "pytorch_flash" not in name) / calls

    def breakdown(hits: list) -> list:
        into: dict[str, list] = {}
        _subtree_self_device(hits, into)
        ranked = sorted(into.items(), key=lambda kv: -kv[1][1])
        return [{"op": name, "per_call": count / calls, "device_ms": us / calls / 1000}
                for name, (count, us) in ranked if us > 0][:8]

    device_ms = call_ms if call_ms is not None else sum(slot[1] for slot in kernels.values()) / calls / 1000
    cat = ops.get("aten::cat")
    return {
        "wall_ms": wall_ms,
        "device_ms": device_ms,
        "prefix_ms": prefix_ms,
        "attention_ms": attention_ms,
        "other_ms": (device_ms - (prefix_ms or 0.0) - (attention_ms or 0.0)) if prefix_ms is not None else None,
        "prefix_regions_per_call": len(prefix_events) / calls,
        "attention_regions_per_call": len(attention_events) / calls,
        "aten_cat_ms": cat[1] / calls / 1000 if cat else 0.0,
        "aten_cat_per_call": cat[0] / calls if cat else 0,
        "sdpa_dispatch_per_call": dispatch,
        "attention_kernel": "+".join(sorted(dispatch)) or "none",
        "prefix_breakdown": breakdown(prefix_events),
        "attention_breakdown": breakdown(attention_events),
        "top_ops": [{"op": name, "per_call": c / calls, "self_device_ms": s / calls / 1000,
                     "device_ms": t / calls / 1000}
                    for name, (c, s, t) in sorted(ops.items(), key=lambda kv: -kv[1][1])[:top]],
        "top_kernels": [{"kernel": name[:160], "per_call": c / calls, "device_ms": t / calls / 1000}
                        for name, (c, t) in sorted(kernels.items(), key=lambda kv: -kv[1][1])[:top]],
    }


# ----------------------------------------------------------------------------
# Fits and the text summary
# ----------------------------------------------------------------------------

def fit(points: list[tuple[int, int, float]]) -> dict | None:
    """Least squares ms = a + b*rows + c*rows*ctx/1000 over (rows, ctx, ms)."""
    if len({(r, c) for r, c, _ in points}) < 4:
        return None
    x = torch.tensor([[1.0, r, r * c / 1000] for r, c, _ in points], dtype=torch.float64)
    y = torch.tensor([[m] for _, _, m in points], dtype=torch.float64)
    solution = torch.linalg.lstsq(x, y).solution.flatten().tolist()
    residual = (x @ torch.tensor(solution, dtype=torch.float64).unsqueeze(1) - y).flatten()
    return {"a": solution[0], "b": solution[1], "c": solution[2], "n": len(points),
            "rmse_ms": float(residual.pow(2).mean().sqrt())}


def fits(timing: list[dict], variants: list[Variant]) -> dict:
    ok = [r for r in timing if r.get("status") == "ok"]
    cells = {(r["rows"], r["context"]) for r in ok}
    common = {cell for cell in cells
              if all(any(r["variant"] == v.name and (r["rows"], r["context"]) == cell for r in ok) for v in variants)}
    out: dict[str, Any] = {"common_cells": sorted(common), "gpu_bound_row_tokens": GPU_BOUND_ROW_TOKENS}
    for v in variants:
        mine = [(r["rows"], r["context"], r["median_ms"]) for r in ok if r["variant"] == v.name]
        shared = [p for p in mine if (p[0], p[1]) in common]
        out[v.name] = {
            "own": fit(mine),
            "common": fit(shared),
            "common_gpu_bound": fit([p for p in shared if p[0] * p[1] >= GPU_BOUND_ROW_TOKENS]),
        }
    return out


def fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def attribution(fit_table: dict) -> list[tuple[str, float]]:
    """Differences of the GPU-bound rows*ctx slope c (ms per row per 1k ctx)."""
    def c(name: str) -> float | None:
        entry = fit_table.get(name, {}).get("common_gpu_bound")
        return entry["c"] if entry else None

    base = "sdpa_mask/cat/torch"
    slopes = [("total slope, sdpa_mask/cat (current)", base)]
    differences = [
        ("(a) prefix cat, math kernel kept: mask/cat - mask/buf", base, "sdpa_mask/buffer/torch"),
        ("(a) prefix cat, SDPA no-mask kept: nomask/cat - nomask/buf",
         "sdpa_nomask/cat/torch", "sdpa_nomask/buffer/torch"),
        ("(a) prefix cat, flash_attn kept: flash/cat - flash/buf", "flash/cat/torch", "flash/buffer/torch"),
        ("(b) kernel, cat kept: mask/cat - nomask/cat", base, "sdpa_nomask/cat/torch"),
        ("(b) kernel, cat kept: mask/cat - flash/cat", base, "flash/cat/torch"),
    ]
    rests = [("(c) rest: nomask/buf slope", "sdpa_nomask/buffer/torch"),
             ("(c) rest: flash/buf slope", "flash/buffer/torch")]
    lines = [(label, c(name)) for label, name in slopes if c(name) is not None]
    lines += [(label, c(first) - c(second)) for label, first, second in differences
              if c(first) is not None and c(second) is not None]
    lines += [(label, c(name)) for label, name in rests if c(name) is not None]
    return lines


def summary_text(record: dict) -> str:
    s = record["shape"]
    variants = [Variant.parse(name) for name in record["variants"]]
    out = [f"== {record['model']}  B={record['block']}  {s['layers']} layers  {s['q_heads']} q / {s['kv_heads']} kv "
           f"heads  hd {s['head_dim']}  {record['dtype']}  flash_attn {record['flash_attn'] or 'absent'} =="]
    if record.get("skipped_variants"):
        for name, why in record["skipped_variants"].items():
            out.append(f"  skipped {name}: {why}")
    checks = record.get("correctness") or []
    if checks:
        out.append(f"correctness vs {REFERENCE.name} (3 steps, commit, 1 step):")
        out.append(f"  {'variant':26s} {'cell':>8s} {'max|dlogit|':>11s} {'argmax':>8s} {'masked':>8s}  bitwise")
        for r in checks:
            if r.get("status") != "ok":
                out.append(f"  {r['variant']:26s} {r['rows']:>3d}x{r['context']:<4d} {r.get('status')}")
                continue
            tag = ("==cat " + str(r["bitwise_equal_cat"])) if "bitwise_equal_cat" in r else (
                ("==mirror " + str(r["bitwise_equal_mirror"])) if "bitwise_equal_mirror" in r else "")
            out.append(f"  {r['variant']:26s} {r['rows']:>3d}x{r['context']:<4d} {r['max_abs_logit_diff']:11.4g} "
                       f"{100 * r['argmax_agree']:7.2f}% {100 * (r['argmax_agree_masked'] or 0):7.2f}%  {tag}")
    timing = [r for r in record.get("timing") or []]
    if timing:
        by = {(r["variant"], r["rows"], r["context"]): r for r in timing}
        cells = sorted({(r["rows"], r["context"]) for r in timing}, key=lambda c: (c[1], c[0]))
        out.append("median ms per verify call (_forward_slices); '-' = skipped/oom:")
        out.append("  " + f"{'ctx':>5s} {'rows':>4s} " + " ".join(f"{v.short:>10s}" for v in variants))
        for rows, context in cells:
            cols = []
            for v in variants:
                r = by.get((v.name, rows, context))
                cols.append(f"{r['median_ms']:10.1f}" if r and r.get("status") == "ok" else f"{'-':>10s}")
            out.append(f"  {context:5d} {rows:4d} " + " ".join(cols))
        table = record.get("fits") or {}
        out.append(f"fit ms = a + b*rows + c*rows*ctx/1k over the {len(table.get('common_cells', []))} cells every "
                   f"variant completed; c_gb over those with rows*ctx >= {GPU_BOUND_ROW_TOKENS}:")
        out.append(f"  {'variant':26s} {'a':>7s} {'b':>6s} {'c':>6s} {'c_gb':>6s} {'a_gb':>6s} {'rmse':>6s}")
        for v in variants:
            entry = table.get(v.name, {})
            common, gb = entry.get("common"), entry.get("common_gpu_bound")
            out.append(f"  {v.name:26s} {fmt(common and common['a']):>7s} {fmt(common and common['b']):>6s} "
                       f"{fmt(common and common['c'], 2):>6s} {fmt(gb and gb['c'], 2):>6s} "
                       f"{fmt(gb and gb['a']):>6s} {fmt(common and common['rmse_ms']):>6s}")
        lines = attribution(table)
        if lines:
            out.append("attribution of the rows*ctx slope (c_gb, ms per row per 1k ctx):")
            for label, value in lines:
                out.append(f"  {label:62s} {value:7.2f}")
        out.append("memory: peak allocated over static (GiB, max over cells); buffer GiB and first-call "
                   "extra ms at the largest buffer cell:")
        for v in variants:
            mine = [r for r in timing if r["variant"] == v.name and r.get("status") == "ok"]
            if not mine:
                continue
            peak = max((r["peak_over_static_bytes"] or 0) for r in mine) / 2**30
            largest = max(mine, key=lambda r: r["rows"] * r["context"])
            extra = (f"  buffer {largest['buffer_bytes'] / 2**30:.2f} GiB, first call +"
                     f"{largest['first_call_over_median_ms']:.1f} ms at {largest['rows']}x{largest['context']}"
                     if largest["buffer_bytes"] else "")
            out.append(f"  {v.name:26s} {peak:6.2f}{extra}")
    profiles = record.get("profile") or []
    if profiles:
        out.append("profile, device ms per call (wall* = profiled CPU span incl. sync, inflated):")
        out.append(f"  {'variant':26s} {'cell':>8s} {'wall*':>7s} {'device':>7s} {'prefix':>7s} {'attn':>7s} "
                   f"{'other':>7s} {'cat':>6s}  kernel")
        for r in profiles:
            if r.get("status") != "ok":
                out.append(f"  {r['variant']:26s} {r['rows']:>3d}x{r['context']:<4d} {r.get('status')}")
                continue
            out.append(f"  {r['variant']:26s} {r['rows']:>3d}x{r['context']:<4d} {fmt(r['wall_ms']):>7s} "
                       f"{fmt(r['device_ms']):>7s} {fmt(r['prefix_ms']):>7s} {fmt(r['attention_ms']):>7s} "
                       f"{fmt(r['other_ms']):>7s} {fmt(r['aten_cat_ms']):>6s}  {r['attention_kernel']}")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", required=True, help="local SDAR checkpoint dirs, run in order")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 512, 1024, 2048, 4096])
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS),
                        help="attention/prefix[/norm] or 'original'; attention in "
                             f"{ATTENTION}, prefix in {PREFIX}, norm in {NORM} (default torch)")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--reps", type=int, default=4, help="timed calls per job per round")
    parser.add_argument("--warmup", type=int, default=1,
                        help="untimed calls per job per round, >= 1; the first is recorded as the first-call time")
    parser.add_argument("--check-cells", nargs="*", default=["1x512", "8x2048"], help="ROWSxCTX")
    parser.add_argument("--profile-cells", nargs="*", default=["1x2048", "8x2048"], help="ROWSxCTX")
    parser.add_argument("--profile-calls", type=int, default=3)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--buffer-rows-max", type=int, default=None,
                        help="rows the buffers are allocated for (default: the call's rows)")
    parser.add_argument("--vocab-limit", type=int, default=151936)
    parser.add_argument("--prefill-chunk", type=int, default=512,
                        help="tokens per commit call when building each prefix (a multiple of the block)")
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--contention-guard", choices=["wait", "off"], default="wait")
    parser.add_argument("--tag", default=None, help="output name (default: host_YYYYmmdd_HHMMSS)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.warmup < 1:
        parser.error("--warmup must be >= 1")
    if args.tag is None:
        args.tag = f"{socket.gethostname().split('.')[0]}_{time.strftime('%Y%m%d_%H%M%S')}"
    return args


def load(model_id: str, args: argparse.Namespace) -> tuple[Any, int]:
    block = trained_block_length(model_id)
    if block is None:
        raise ValueError(f"cannot read a block length from {model_id!r}")
    if args.prefill_chunk % block:
        raise ValueError(f"--prefill-chunk must be a multiple of the block ({block})")
    provider = MaskedDiffusionProvider.from_pretrained(ModelOptions(
        model_id=model_id, family="sdar", block_length=block, device=args.device,
        dtype=args.dtype, local_files_only=True, require_lean_logits=True,
        vocab_limit=args.vocab_limit,
    ))
    return provider, block


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    tmp.replace(path)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    requested = [Variant.parse(text) for text in args.variants]
    check_cells = [parse_cell(c) for c in args.check_cells]
    profile_cells = [parse_cell(c) for c in args.profile_cells]
    output = args.output_dir / f"{args.tag}.json"
    if cuda:
        uuid = f"GPU-{torch.cuda.get_device_properties(device).uuid}"
        card = {"uuid": uuid, "name": torch.cuda.get_device_name(device),
                "visible": os.environ.get("CUDA_VISIBLE_DEVICES")}
        guard = ContentionGuard([uuid]) if args.contention_guard == "wait" else NullGuard()
    else:
        card = {"uuid": None, "name": "cpu", "visible": os.environ.get("CUDA_VISIBLE_DEVICES")}
        guard = NullGuard()
    print(f"device {args.device} {card['name']} {card['uuid']}", flush=True)
    import transformers

    payload: dict[str, Any] = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "host": socket.gethostname(), "card": card, "torch": torch.__version__,
        "transformers": transformers.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sdp_backends": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        },
        "models": [],
    }
    log = lambda message: print(message, flush=True)  # noqa: E731
    guard.wait_until_clear()
    began = time.perf_counter()
    with guard.watch() as watch:
        for model_id in args.model:
            loaded = time.perf_counter()
            provider, block = load(model_id, args)
            _, module = sdar_attention(provider.model)
            kernels = load_kernels(module)
            variants, skipped = [], {}
            for variant in requested:
                why = kernels.missing(variant)
                if why:
                    skipped[variant.name] = why
                    log(f"skip {variant.name}: {why}")
                else:
                    variants.append(variant)
            rows = [r for r in args.rows if r <= block]
            cells = [(r, c) for c in args.contexts for r in rows]
            for r, c in check_cells + profile_cells:
                if not 1 <= r <= block or c < 1:
                    raise ValueError(f"cell {r}x{c}: rows must be in [1, {block}] and ctx >= 1")
            session = Session(provider, block, args, kernels, log=log)
            session.build(args.contexts + [c for _, c in check_cells + profile_cells],
                          cells + profile_cells)
            record: dict[str, Any] = {
                "model": Path(model_id).name, "model_path": str(model_id), "block": block,
                "dtype": args.dtype, "shape": session.shape, "flash_attn": kernels.version,
                "flash_attn_error": kernels.flash_error, "rms_norm_error": kernels.rms_norm_error,
                "mirror_of_original": mirror_of_original(provider.model).name,
                "variants": [v.name for v in variants], "skipped_variants": skipped,
                "static_allocated_bytes": session.allocated(),
                "load_seconds": time.perf_counter() - loaded,
            }
            payload["models"].append(record)
            log(f"{record['model']}: B={block} {session.shape}, loaded in {record['load_seconds']:.0f} s, "
                f"static {fmt((record['static_allocated_bytes'] or 0) / 2**30, 2)} GiB")
            try:
                if not args.skip_check:
                    log("correctness")
                    record["correctness"] = session.correctness(variants, check_cells)
            except CorrectnessError as error:
                record["correctness"] = error.records
                record["correctness_failure"] = str(error)
                write(output, payload)
                raise SystemExit(f"CORRECTNESS FAILURE ({record['model']}): {error}; wrote {output}")
            write(output, payload)
            if not args.skip_timing:
                log(f"timing: {len(variants)} variants x {len(cells)} cells")
                record["timing"], record["timing_meta"] = session.timing(variants, cells)
                record["fits"] = fits(record["timing"], variants)
                write(output, payload)
            if not args.skip_profile:
                log("profile")
                record["profile"] = session.profile(variants, profile_cells)
            record["seconds"] = time.perf_counter() - loaded
            write(output, payload)
            text = summary_text(record)
            print(text, flush=True)
            del session, provider
            gc.collect()
            if cuda:
                torch.cuda.empty_cache()
    payload.update(watch.record())
    payload["contended"] = watch.contended
    payload["seconds"] = time.perf_counter() - began
    write(output, payload)
    summary = "\n\n".join(summary_text(record) for record in payload["models"])
    output.with_suffix(".txt").write_text(summary + "\n")
    print(f"contended: {watch.contended}  host load {fmt(watch.load_before)} -> {fmt(watch.load_after)}  "
          f"total {payload['seconds'] / 60:.1f} min")
    print(f"wrote {output} and {output.with_suffix('.txt')}")
    return payload


if __name__ == "__main__":
    main()
