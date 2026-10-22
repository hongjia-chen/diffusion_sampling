"""Dependency-transfer measurement: the η study (MI_decoding/ETA_STUDY.md).

At a proposal state h, the block has k masked slots. The proposer ranks them by
confidence (slots i_1..i_k, argmax candidates y_1..y_k). Probe row j fills
y_1..y_{j-1} and is read at slot i_j; row 1 fills nothing and is also read at
every slot. Both models read the same rows, so for each rank j:

    q_j   = p(. | h)          at i_j      row 1
    p_j   = p(. | h, y_<j)    at i_j      row j
    D_j   = TV(p_j, q_j)                  per model: D^P (proposer), D^V (reference)
    eta_j = ½ ‖(p_j^P − q_j^P) − (p_j^V − q_j^V)‖₁

so TV(p_j^P, q_j^P) ≤ D_j^V + eta_j. These are the same rows the MI verifier
builds (`cross.dream_verify`), run through the proposer as well.

`generate_measured` decodes exactly as `blockwise.generate_blockwise` at
fixed:1 (one token per step, greedy, logp confidence), keeps the reference
model's KV cache in step block by block, and at chosen steps runs the probe
rows through both models. The probes are read-only forwards (store_kv=False):
nothing they compute feeds back into the decode, so the output must equal the
fixed:1 baseline's.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sdar_harness.blockwise import BlockCacheProvider
from sdar_harness.model import block_causal_mask
from sdar_harness.scheduler import (
    _forward_slices,
    _propose,
    finished_at,
    stop_token_ids,
    truncate_at_stop,
)


def probe_rows(state: torch.Tensor, positions: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """[k, B] rows: row j (0-based) holds candidates 0..j-1 at their slots."""
    count = int(positions.numel())
    rows = state.repeat(count, 1)
    for row in range(1, count):
        rows[row, positions[:row]] = tokens[:row]
    return rows


def read_probe_logits(
    provider: Any, rows: torch.Tensor, positions: torch.Tensor, group_rows: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(conditional [k, V], marginal [k, V], forwards) for one model.

    conditional[j] is row j read at positions[j]; marginal[j] is row 0 read at
    positions[j]. Row 0 is always in the first group, as in `dream_verify`.
    """
    if group_rows < 1:
        raise ValueError("group_rows must be positive")
    device = provider.device
    rows = rows.to(device)
    positions = positions.to(device)
    count = int(rows.shape[0])
    conditional: list[torch.Tensor] = []
    marginal: torch.Tensor | None = None
    forwards = 0
    for start in range(0, count, group_rows):
        chunk = rows[start : start + group_rows]
        width = int(chunk.shape[0])
        gathers = [(torch.arange(width, device=device), positions[start : start + width])]
        if start == 0:
            gathers.append((torch.zeros_like(positions), positions))
        outputs = _forward_slices(provider, chunk, gathers)
        forwards += 1
        conditional.append(outputs[0].float())
        if start == 0:
            marginal = outputs[1].float()
    assert marginal is not None
    return torch.cat(conditional), marginal, forwards


def _log_probs64(logits: torch.Tensor, excluded: set[int]) -> torch.Tensor:
    """`scheduler._filtered_log_probs` in float64: the same support, finer sums."""
    scoring = logits.to(torch.float64, copy=True)
    banned = sorted(index for index in excluded if 0 <= index < scoring.shape[-1])
    if banned:
        scoring.index_fill_(-1, torch.tensor(banned, device=scoring.device), -torch.inf)
    return torch.log_softmax(scoring, dim=-1)


def _entropy(probs: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
    return -torch.where(probs > 0, probs * log_probs, torch.zeros_like(probs)).sum(-1)


# Per-rank fields `dependency_metrics` returns, in record order.
METRIC_FIELDS = (
    "D_P", "D_V", "eta", "tv_PV",
    "conf_P", "conf_V", "gap_P", "gap_V", "ent_P", "ent_V",
    "lr_P", "lr_V",
    "argmax_qP", "argmax_pP", "argmax_qV", "argmax_pV",
)
# Rows per float64 chunk: each [rows, V] float64 tensor is 1.2 MB per row at
# V=151936, and about a dozen are alive at once.
METRIC_CHUNK = 4


@torch.inference_mode()
def dependency_metrics(
    conditional_p: torch.Tensor,
    marginal_p: torch.Tensor,
    conditional_v: torch.Tensor,
    marginal_v: torch.Tensor,
    tokens: torch.Tensor,
    excluded: set[int],
) -> dict[str, list]:
    """Per-rank TVs, η and diagnostics from the four [k, V] logit tensors.

    All inputs must be on one device. `tokens` are the proposer's candidates
    y_j. Everything is computed in float64 over the whole (filtered) vocabulary.
    """
    out: dict[str, list] = {name: [] for name in METRIC_FIELDS}
    count = int(tokens.numel())
    for start in range(0, count, METRIC_CHUNK):
        stop = start + METRIC_CHUNK
        lq_p = _log_probs64(marginal_p[start:stop], excluded)
        lp_p = _log_probs64(conditional_p[start:stop], excluded)
        lq_v = _log_probs64(marginal_v[start:stop], excluded)
        lp_v = _log_probs64(conditional_v[start:stop], excluded)
        q_p, p_p, q_v, p_v = lq_p.exp(), lp_p.exp(), lq_v.exp(), lp_v.exp()
        d_p, d_v = p_p - q_p, p_v - q_v
        y = tokens[start:stop].to(q_p.device).reshape(-1, 1)
        top_p = q_p.topk(2, dim=-1).values
        top_v = q_v.topk(2, dim=-1).values
        values = {
            "D_P": 0.5 * d_p.abs().sum(-1),
            "D_V": 0.5 * d_v.abs().sum(-1),
            "eta": 0.5 * (d_p - d_v).abs().sum(-1),
            "tv_PV": 0.5 * (q_p - q_v).abs().sum(-1),
            "conf_P": q_p.gather(1, y).squeeze(1),
            "conf_V": q_v.gather(1, y).squeeze(1),
            "gap_P": top_p[:, 0] - top_p[:, 1],
            "gap_V": top_v[:, 0] - top_v[:, 1],
            "ent_P": _entropy(q_p, lq_p),
            "ent_V": _entropy(q_v, lq_v),
            "lr_P": (lp_p.gather(1, y) - lq_p.gather(1, y)).squeeze(1),
            "lr_V": (lp_v.gather(1, y) - lq_v.gather(1, y)).squeeze(1),
            "argmax_qP": q_p.argmax(-1), "argmax_pP": p_p.argmax(-1),
            "argmax_qV": q_v.argmax(-1), "argmax_pV": p_v.argmax(-1),
        }
        for name, value in values.items():
            if name.startswith("argmax"):
                out[name].extend(int(v) for v in value.tolist())
            else:
                out[name].extend(_compact(v) for v in value.tolist())
        del lq_p, lp_p, lq_v, lp_v, q_p, p_p, q_v, p_v, d_p, d_v
    return out


@torch.inference_mode()
def self_floor(
    conditional_a: torch.Tensor,
    marginal_a: torch.Tensor,
    conditional_b: torch.Tensor,
    marginal_b: torch.Tensor,
    excluded: set[int],
) -> dict[str, list]:
    """η of one model against itself, read under two row groupings.

    In exact arithmetic it is 0. In fp16 a different batch shape can take a
    different GEMM path, so this is the numeric floor under which η carries no
    signal. `noise_q` is TV between the two readings of the marginal.
    """
    out: dict[str, list] = {"eta_self": [], "noise_q": []}
    count = int(conditional_a.shape[0])
    for start in range(0, count, METRIC_CHUNK):
        stop = start + METRIC_CHUNK
        q_a = _log_probs64(marginal_a[start:stop], excluded).exp()
        p_a = _log_probs64(conditional_a[start:stop], excluded).exp()
        q_b = _log_probs64(marginal_b[start:stop], excluded).exp()
        p_b = _log_probs64(conditional_b[start:stop], excluded).exp()
        eta = 0.5 * ((p_a - q_a) - (p_b - q_b)).abs().sum(-1)
        noise = 0.5 * (q_a - q_b).abs().sum(-1)
        out["eta_self"].extend(_compact(v) for v in eta.tolist())
        out["noise_q"].extend(_compact(v) for v in noise.tolist())
    return out


def _compact(value: float) -> float:
    """Seven significant digits: tiny TVs keep their relative precision."""
    return float(f"{value:.7g}")


def _sync(devices: list[torch.device]) -> None:
    for device in devices:
        if device.type == "cuda":
            torch.cuda.synchronize(device)


@dataclass(frozen=True)
class MeasureConfig:
    continuation_length: int
    # Probe rows per model call. A B=16 step has at most 16 rows; one call
    # reads q and every p_j from a single batch.
    group_rows: int = 16
    # Steps measured per block, drawn uniformly from the block's steps that
    # still have at least two masked slots (rank 1 alone carries no η).
    measure_per_block: int = 1
    # Cap on measured steps per task. A 4096-token runaway would otherwise
    # contribute 256 blocks of a repeating loop.
    max_measured: int | None = 64
    # Also read the proposer's rows under this grouping and record its
    # self-η (`self_floor`). None skips it.
    floor_group_rows: int | None = None

    def __post_init__(self) -> None:
        if self.continuation_length < 1 or self.group_rows < 1:
            raise ValueError("continuation_length and group_rows must be positive")
        if self.measure_per_block < 0 or (self.max_measured is not None and self.max_measured < 0):
            raise ValueError("measure_per_block and max_measured must be non-negative")
        if self.floor_group_rows is not None and self.floor_group_rows == self.group_rows:
            raise ValueError("the floor needs a grouping different from group_rows")


@dataclass
class MeasuredResult:
    continuation_token_ids: list[int]
    full_continuation_token_ids: list[int]
    records: list[dict[str, Any]]
    denoising_steps: int
    commit_passes: int
    eos_position: int | None
    stop_token_id: int | None
    early_stopped: bool
    measure_seconds: float = 0.0
    probe_forwards: dict[str, int] = field(default_factory=dict)


def _distance_to_new(positions: list[int]) -> list[int | None]:
    """Per rank j: distance from slot i_j to the nearest of i_<j (None at j=1)."""
    out: list[int | None] = [None]
    for j in range(1, len(positions)):
        out.append(min(abs(positions[j] - earlier) for earlier in positions[:j]))
    return out


@torch.inference_mode()
def generate_measured(
    proposer: Any,
    reference: Any,
    prompt_ids: torch.Tensor,
    config: MeasureConfig,
    *,
    rng: random.Random,
) -> MeasuredResult:
    """fixed:1 greedy decode by `proposer`, with dependency probes at chosen steps."""
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("one prompt at a time")
    block = proposer.conventions.block_length
    if block is None:
        raise ValueError("the proposer must be a block diffusion checkpoint")
    if reference.conventions.block_length != block:
        raise ValueError("the reference must share the proposer's block length")
    if reference.mask_token_id != proposer.mask_token_id:
        raise ValueError("the reference must share the proposer's mask token")

    device, ref_device = proposer.device, reference.device
    devices = list(getattr(proposer, "devices", [device])) + list(getattr(reference, "devices", [ref_device]))
    mask = proposer.mask_token_id
    excluded = proposer.excluded_token_ids
    prompt_length = int(prompt_ids.shape[1])
    generated = config.continuation_length
    num_blocks = math.ceil((prompt_length + generated) / block)
    total_length = num_blocks * block

    canvas = torch.full((1, total_length), mask, dtype=torch.long, device=device)
    canvas[:, :prompt_length] = prompt_ids.to(device)
    attention = block_causal_mask(total_length, block, device=device)
    positions_all = torch.arange(total_length, device=device)
    ref_attention = attention.to(ref_device)
    ref_positions = positions_all.to(ref_device)

    from transformers.cache_utils import DynamicCache

    cache, ref_cache = DynamicCache(), DynamicCache()
    commit_passes = 0

    def commit(low: int, high: int) -> None:
        nonlocal commit_passes
        proposer.commit(canvas[:, low:high], attention[low:high, :high], positions_all[low:high], cache)
        reference.commit(
            canvas[:, low:high].to(ref_device), ref_attention[low:high, :high],
            ref_positions[low:high], ref_cache,
        )
        commit_passes += 1

    prefill_blocks = prompt_length // block
    if prefill_blocks:
        commit(0, prefill_blocks * block)

    stop_ids = stop_token_ids(proposer)
    stop_tensor = torch.tensor(sorted(stop_ids), device=device)
    canvas_low, canvas_high = prompt_length, prompt_length + generated
    records: list[dict[str, Any]] = []
    forwards = {"proposer": 0, "reference": 0, "floor": 0}
    measure_seconds = 0.0
    steps = 0
    early_stopped = False

    for index in range(prefill_blocks, num_blocks):
        low, high = index * block, (index + 1) * block
        state = canvas[:, low:high].clone()
        masked = int((state == mask).sum().item())
        if not masked:
            continue
        scoped = BlockCacheProvider(proposer, cache, attention[low:high, :high], positions_all[low:high])
        ref_scoped = BlockCacheProvider(
            reference, ref_cache, ref_attention[low:high, :high], ref_positions[low:high]
        )
        # Step s of this block has masked - s slots; measure only s with >= 2.
        budget = config.measure_per_block
        if config.max_measured is not None:
            budget = min(budget, config.max_measured - len(records))
        eligible = list(range(0, masked - 1))
        chosen = set(rng.sample(eligible, min(max(budget, 0), len(eligible))))
        block_step = 0
        while bool((state == mask).any()):
            remaining = torch.nonzero(state[0] == mask, as_tuple=False).flatten()
            candidates = _propose(
                scoped, state, remaining, block, temperature=0.0, top_p=None, confidence="logp"
            )
            if block_step in chosen:
                _sync(devices)
                began = time.perf_counter()
                record = _measure(
                    scoped, ref_scoped, state, candidates, config, excluded, forwards,
                    stop_tensor=stop_tensor, answer_from=max(prompt_length - low, 0),
                    device=device,
                )
                _sync(devices)
                record["measure_ms"] = round(1000 * (time.perf_counter() - began), 2)
                measure_seconds += time.perf_counter() - began
                record.update({
                    "block": index, "block_step": block_step, "step": steps,
                    "block_start": low, "answer_offset": low - prompt_length,
                    "filled_in_block": block - int(remaining.numel()),
                    "context_tokens": low + block - int(remaining.numel()),
                })
                records.append(record)
            # Exactly scheduler.generation_step at fixed:1: commit rank 1.
            state = state.clone()
            state[0, candidates.sequence_positions[:1]] = candidates.token_ids[:1]
            steps += 1
            block_step += 1
            canvas[:, low:high] = state
            finished = finished_at(canvas[0, canvas_low:canvas_high], mask_token_id=mask, stop_ids=stop_ids)
            if finished is not None:
                early_stopped = True
                break
        canvas[:, low:high] = state
        if early_stopped:
            break
        commit(low, high)

    continuation = canvas[0, prompt_length : prompt_length + generated].detach().cpu().tolist()
    scored, eos_position, stop_token = truncate_at_stop(continuation, stop_ids)
    return MeasuredResult(
        continuation_token_ids=[int(v) for v in scored],
        full_continuation_token_ids=[int(v) for v in continuation],
        records=records,
        denoising_steps=steps,
        commit_passes=commit_passes,
        eos_position=eos_position,
        stop_token_id=stop_token,
        early_stopped=early_stopped and mask in continuation,
        measure_seconds=measure_seconds,
        probe_forwards=forwards,
    )


def _measure(
    scoped: BlockCacheProvider,
    ref_scoped: BlockCacheProvider,
    state: torch.Tensor,
    candidates: Any,
    config: MeasureConfig,
    excluded: set[int],
    forwards: dict[str, int],
    *,
    stop_tensor: torch.Tensor,
    answer_from: int,
    device: torch.device,
) -> dict[str, Any]:
    positions = candidates.sequence_positions
    tokens = candidates.token_ids
    rows = probe_rows(state, positions, tokens)
    cond_p, marg_p, used = read_probe_logits(scoped, rows, positions, config.group_rows)
    forwards["proposer"] += used
    cond_v, marg_v, used = read_probe_logits(ref_scoped, rows, positions, config.group_rows)
    forwards["reference"] += used
    metrics = dependency_metrics(
        cond_p, marg_p, cond_v.to(device), marg_v.to(device), tokens, excluded
    )
    if config.floor_group_rows is not None:
        cond_b, marg_b, used = read_probe_logits(scoped, rows, positions, config.floor_group_rows)
        forwards["floor"] += used
        metrics.update(self_floor(cond_p, marg_p, cond_b, marg_b, excluded))
    slots = [int(v) for v in positions.tolist()]
    # The rendered prompt ends in <|im_end|>, a stop token; its tail can share
    # the first answer block, so only slots from `answer_from` on count.
    stops = torch.nonzero(torch.isin(state[0, answer_from:], stop_tensor), as_tuple=False).flatten()
    stops = stops + answer_from
    return {
        "k": len(slots),
        "slots": slots,
        "tokens": [int(v) for v in tokens.tolist()],
        "confidence_ranked": [_compact(v) for v in candidates.confidences.tolist()],
        "dist_new": _distance_to_new(slots),
        "first_stop_slot": int(stops.min().item()) if stops.numel() else -1,
        **metrics,
    }
