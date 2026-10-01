"""Cross-model adaptive delta: SDAR proposes, Dream verifies and remasks.

DESIGN.md F1-F4, in code:

* SDAR-1.7B-b64 runs the block loop exactly as `blockwise.generate_blockwise`
  does: a KV cache over committed blocks, one block at a time, and a proposal
  for every still-masked position of the current block ranked by confidence.
* Dream-7B scores that ranking. Row 0 is the unmodified canvas and gives the
  Dream marginal at every candidate position; row j has candidates 1..j filled
  and gives Dream's conditional for candidate j+1. Rows run in groups of 4 and
  stop at the first group containing a violation (F1). The acceptance ratio is
  Dream-only (F4): |sum log p_D(y_i | c, y_<i) - log p_D(y_i | c)| <= log(1+delta).
* Veto (F4): if SDAR's top candidate is not Dream's argmax at that position,
  Dream's argmax is committed there and nothing else this step.
* Remasking (F3): once per block -- when the block's canvas positions are all
  committed, or at the early-stop point, whichever comes first, and always
  before SDAR's commit forward -- Dream re-scores every committed token of the
  block with that one token masked. Of the 4 least probable, those Dream would
  change are kept; one is drawn with weight 1/p and replaced by Dream's argmax.

Coordinates. The SDAR canvas is the single source of truth: `canvas` holds the
SDAR prompt, the generation region and the block-rounding tail. Canvas index k
(0 <= k < continuation_length) is SDAR position `P_S + k` and Dream position
`P_D + k`. Dream sees only the generation region, with SDAR's mask swapped for
Dream's. Both providers read logits below the shared `vocab_limit`, so any
token either model can commit means the same thing to the other.

The block-rounding tail (positions past the canvas in the last block) is never
decoded: it stays masked, as it would at the start of a block, because the
tokens there are outside the answer and Dream has no position for them.

Block verifier (exp06, SDAR-8B verifying SDAR-1.7B). When the verifier is itself
a block-diffusion checkpoint it shares SDAR's tokenizer, prompt, mask token and
block grid. It then keeps its own KV cache, committed block by block alongside
SDAR's, and scores rows of the current block only (`BlockCacheProvider`), so
canvas index k is block-local position `P_S + k - low` for both models. The
rules above are unchanged; "Dream" in names and counters means "the verifier".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from adaptive_re.blockwise import BlockCacheProvider
from adaptive_re.model import block_causal_mask
from adaptive_re.scheduler import (
    GenerationConfig,
    GenerationResult,
    _filtered_log_probs,
    _forward_slices,
    _propose,
    finished_at,
    stop_token_ids,
    truncate_at_stop,
)
from adaptive_re.timing import NULL_TIMER, StageTimer


# The one group size for every Dream forward (DESIGN.md F1): per-row time is
# flat from 3-4 rows up on an A4000, so 4 is the smallest group that reaches
# it. Fixed, not a knob -- in bf16 the group size can move a borderline
# acceptance, so it must be identical across arms.
DREAM_GROUP_ROWS = 4


@dataclass(frozen=True)
class CrossConfig(GenerationConfig):
    """Dream-verified adaptive delta. `method` stays "adaptive"."""

    # Off for exp05's runs (decided 2026-09-19); kept as an ablation.
    veto: bool = False
    remask: bool = False
    remask_bottom: int = 4
    group_rows: int = DREAM_GROUP_ROWS

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.method != "adaptive":
            raise ValueError("CrossConfig is an adaptive method")
        if self.remask_bottom < 1 or self.group_rows < 1:
            raise ValueError("remask_bottom and group_rows must be positive")

    @property
    def label(self) -> str:
        value = f"{self.delta:.2f}".replace(".", "p")
        kind = "xadaptive_remask" if self.remask else "xadaptive"
        suffix = "" if self.veto else "_noveto"
        return f"{kind}{suffix}_delta_{value}"


@dataclass(frozen=True)
class VerifyResult:
    accepted: int
    tokens: torch.Tensor           # the committed tokens, Dream's argmax if vetoed
    marginal: torch.Tensor         # log p_D(y_i | c) for scored candidates
    conditional: torch.Tensor      # log p_D(y_i | c, y_<i) for scored candidates
    errors: torch.Tensor
    rows: int
    passes: int
    vetoed: bool
    head_argmax: int
    head_log_prob: float           # Dream's log-probability of its own argmax there


def dream_verify(
    dream: Any,
    base: torch.Tensor,
    positions: torch.Tensor,
    tokens: torch.Tensor,
    *,
    tau_nats: float,
    group_rows: int,
    veto: bool,
) -> VerifyResult:
    """Score SDAR's ranked candidates with Dream, in groups, with early exit.

    `base` is Dream's [1, S] state; `positions` are Dream positions of the
    ranked candidates and `tokens` their SDAR tokens, all on Dream's device.
    """
    count = int(tokens.numel())
    probes = base.repeat(count, 1)
    for row in range(1, count):
        probes[row, positions[:row]] = tokens[:row]
    excluded = dream.excluded_token_ids
    device = positions.device
    conditional_parts: list[torch.Tensor] = []
    marginal: torch.Tensor | None = None
    passes = scored = 0
    vetoed = False
    head_argmax, head_log_prob = -1, float("nan")
    for start in range(0, count, group_rows):
        chunk = probes[start : start + group_rows]
        width = int(chunk.shape[0])
        gathers = [(torch.arange(width, device=device), positions[start : start + width])]
        if start == 0:
            gathers.append((torch.zeros_like(positions), positions))
        outputs = _forward_slices(dream, chunk, gathers)
        passes += 1
        scored = start + width
        log_probs = _filtered_log_probs(outputs[0], excluded)
        conditional_parts.append(
            log_probs.gather(1, tokens[start:scored].reshape(-1, 1)).squeeze(1)
        )
        if start == 0:
            marginal_log_probs = _filtered_log_probs(outputs[1], excluded)
            marginal = marginal_log_probs.gather(1, tokens.reshape(-1, 1)).squeeze(1)
            del marginal_log_probs
            # Row 0 at candidate 1's position is Dream's marginal there.
            head_value, head_index = log_probs[0].max(-1)
            head_argmax, head_log_prob = int(head_index.item()), float(head_value.item())
            if veto and head_argmax != int(tokens[0].item()):
                vetoed = True
                break
        del log_probs
        conditional = torch.cat(conditional_parts)
        errors = (conditional.cumsum(0) - marginal[:scored].cumsum(0)).abs()
        passing = errors <= tau_nats
        passing[0] = True
        if not bool(passing.all()):
            break
    assert marginal is not None
    conditional = torch.cat(conditional_parts)
    errors = (conditional.cumsum(0) - marginal[:scored].cumsum(0)).abs()
    if vetoed:
        accepted = 1
        committed = torch.tensor([head_argmax], device=device, dtype=tokens.dtype)
    else:
        passing = errors <= tau_nats
        passing[0] = True
        accepted = int(torch.cumprod(passing.to(torch.int64), dim=0).sum().item())
        committed = tokens[:accepted]
    return VerifyResult(
        accepted=accepted, tokens=committed, marginal=marginal[:scored],
        conditional=conditional, errors=errors, rows=scored, passes=passes,
        vetoed=vetoed, head_argmax=head_argmax, head_log_prob=head_log_prob,
    )


@dataclass(frozen=True)
class RemaskResult:
    checked_positions: list[int]   # canvas indices
    log_probs: list[float]         # log p_D(z_i | z^{-i})
    argmax: list[int]
    bottom: list[int]              # canvas indices of the bottom-k, lowest first
    survivors: list[int]           # of those, where Dream's argmax != z_i
    chosen: int | None             # canvas index replaced, or None if skipped
    old_token: int | None
    new_token: int | None
    rows: int
    passes: int


def remask_pass(
    dream: Any,
    base: torch.Tensor,
    canvas_indices: torch.Tensor,
    dream_positions: torch.Tensor,
    tokens: torch.Tensor,
    *,
    bottom: int,
    group_rows: int,
    stop_ids: frozenset[int] = frozenset(),
) -> RemaskResult:
    """One Dream row per committed token, that token masked (DESIGN.md F3).

    The stop tokens count as one symbol, "the answer ends here" (F2). A
    committed stop token is scored by Dream's total probability on the whole
    stop set, and swapping one stop token for another is not a change.
    Without this, Dream -- whose only EOS is <|endoftext|> -- finds SDAR's
    <|im_end|> improbable at every stop point and spends the block's one
    remask on a cosmetic swap. Replacing a stop token with content, or
    content with a stop token, is still a change.
    """
    count = int(tokens.numel())
    device = dream_positions.device
    stops = torch.tensor(sorted(stop_ids), device=device, dtype=tokens.dtype)
    probes = base.repeat(count, 1)
    probes[torch.arange(count, device=device), dream_positions] = dream.mask_token_id
    parts_logp: list[torch.Tensor] = []
    parts_arg: list[torch.Tensor] = []
    passes = 0
    for start in range(0, count, group_rows):
        chunk = probes[start : start + group_rows]
        width = int(chunk.shape[0])
        (logits,) = _forward_slices(
            dream, chunk,
            [(torch.arange(width, device=device), dream_positions[start : start + width])],
        )
        passes += 1
        log_probs = _filtered_log_probs(logits, dream.excluded_token_ids)
        chunk_tokens = tokens[start : start + width]
        token_log_p = log_probs.gather(1, chunk_tokens.reshape(-1, 1)).squeeze(1)
        if stops.numel():
            stop_log_p = torch.logsumexp(log_probs[:, stops], dim=1)
            token_log_p = torch.where(torch.isin(chunk_tokens, stops), stop_log_p, token_log_p)
        parts_logp.append(token_log_p)
        parts_arg.append(log_probs.argmax(-1))
        del log_probs
    log_p = torch.cat(parts_logp)
    argmax = torch.cat(parts_arg)
    order = torch.argsort(log_p, stable=True)[: min(bottom, count)]
    same_stop = torch.isin(argmax[order], stops) & torch.isin(tokens[order], stops)
    keep = order[(argmax[order] != tokens[order]) & ~same_stop]
    chosen = old = new = None
    if keep.numel():
        # Weight 1/p, normalized over the survivors: softmax(-log p) is exactly
        # that, and stays finite where 1/p would overflow float32.
        weights = torch.softmax(-log_p[keep].to(torch.float64), dim=0)
        pick = int(keep[torch.multinomial(weights, 1)].item())
        chosen = int(canvas_indices[pick].item())
        old, new = int(tokens[pick].item()), int(argmax[pick].item())
    return RemaskResult(
        checked_positions=[int(v) for v in canvas_indices.tolist()],
        log_probs=[float(v) for v in log_p.tolist()],
        argmax=[int(v) for v in argmax.tolist()],
        bottom=[int(canvas_indices[i].item()) for i in order.tolist()],
        survivors=[int(canvas_indices[i].item()) for i in keep.tolist()],
        chosen=chosen, old_token=old, new_token=new,
        rows=count, passes=passes,
    )


def generate_cross(
    sdar: Any,
    dream: Any,
    sdar_prompt_ids: torch.Tensor,
    dream_prompt_ids: torch.Tensor,
    config: CrossConfig,
    *,
    capture_traces: bool = False,
    timer: StageTimer | None = None,
) -> GenerationResult:
    if sdar_prompt_ids.ndim != 2 or sdar_prompt_ids.shape[0] != 1:
        raise ValueError("one prompt at a time")
    timer = timer if timer is not None else NULL_TIMER
    block = sdar.conventions.block_length
    if block is None:
        raise ValueError("the proposer must be a block diffusion checkpoint")
    if sdar.vocab_limit is None or sdar.vocab_limit != dream.vocab_limit:
        raise ValueError("proposer and verifier must share one vocab_limit (DESIGN.md F4)")

    device = sdar.device
    prompt_length = int(sdar_prompt_ids.shape[1])
    generated = config.continuation_length
    num_blocks = math.ceil((prompt_length + generated) / block)
    total_length = num_blocks * block
    canvas = torch.full((1, total_length), sdar.mask_token_id, dtype=torch.long, device=device)
    canvas[:, :prompt_length] = sdar_prompt_ids.to(device)
    attention = block_causal_mask(total_length, block, device=device)
    all_positions = torch.arange(total_length, device=device)
    region = slice(prompt_length, prompt_length + generated)

    dream_prompt = dream_prompt_ids.to(dream.device)
    dream_offset = int(dream_prompt.shape[1])
    tau = config.tau_nats
    stop_ids = stop_token_ids(sdar)
    block_verifier = getattr(getattr(dream, "conventions", None), "block_length", None) is not None
    if block_verifier:
        if dream.conventions.block_length != block or dream.mask_token_id != sdar.mask_token_id:
            raise ValueError("a block verifier must share the proposer's block length and mask token")
        if not torch.equal(dream_prompt_ids.cpu(), sdar_prompt_ids.cpu()):
            raise ValueError("a block verifier must see the proposer's prompt token for token")

    def dream_base() -> torch.Tensor:
        body = canvas[0, region].to(dream.device)
        body = torch.where(body == sdar.mask_token_id, dream.mask_token_id, body)
        return torch.cat((dream_prompt, body.unsqueeze(0)), dim=1)

    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    verifier_cache = DynamicCache() if block_verifier else None
    prefill_blocks = prompt_length // block
    prefill_length = prefill_blocks * block
    commit_passes = 0

    def commit(low: int, high: int) -> None:
        nonlocal commit_passes
        with timer("commit"):
            sdar.commit(canvas[:, low:high], attention[low:high, :high], all_positions[low:high], cache)
            if block_verifier:
                dream.commit(
                    canvas[:, low:high].to(dream.device), attention[low:high, :high].to(dream.device),
                    all_positions[low:high].to(dream.device), verifier_cache,
                )
        commit_passes += 1

    if prefill_length:
        commit(0, prefill_length)

    traces: list[dict[str, Any]] = []
    histogram = [0] * config.max_candidates
    steps = 0
    counters = dict(
        dream_verify_rows=0, dream_verify_passes=0, veto_count=0,
        remask_passes=0, remask_rows=0, remask_forward_passes=0,
        remask_skipped=0, tokens_changed_by_remask=0,
    )
    # Every remask edit, so edits can be classified on all tasks, not only the
    # traced ones: canvas index, old token, new token, the old token's log p in
    # the remask view, and that view's bottom-k canvas indices.
    remask_edits: list[dict[str, Any]] = []
    early_stopped = False

    for index in range(prefill_blocks, num_blocks):
        low, high = index * block, (index + 1) * block
        # Block-local positions that are canvas positions (not prompt, not tail).
        local_low = max(prompt_length - low, 0)
        local_high = min(prompt_length + generated - low, block)
        if local_low >= local_high:
            continue
        state = canvas[:, low:high].clone()
        in_region = torch.zeros(block, dtype=torch.bool, device=device)
        in_region[local_low:local_high] = True
        if not bool(((state[0] == sdar.mask_token_id) & in_region).any()):
            continue
        scoped = BlockCacheProvider(sdar, cache, attention[low:high, :high], all_positions[low:high])
        if block_verifier:
            verifier = BlockCacheProvider(
                dream, verifier_cache,
                attention[low:high, :high].to(dream.device), all_positions[low:high].to(dream.device),
            )
            verifier_offset = prompt_length - low
        else:
            verifier, verifier_offset = dream, dream_offset

        def verifier_base() -> torch.Tensor:
            return state.to(dream.device) if block_verifier else dream_base()

        remask_used = not config.remask
        # Dream's commit-time log-probability of each token, for the traces:
        # the remask pass logs how much the rest of the block moved it.
        commit_logp: dict[int, float] = {}

        while True:
            remaining = torch.nonzero(
                (state[0] == sdar.mask_token_id) & in_region, as_tuple=False
            ).flatten()
            finished = None
            if remaining.numel():
                with timer("proposer"):
                    candidates = _propose(
                        scoped, state, remaining, config.max_candidates,
                        temperature=config.temperature, top_p=config.top_p,
                        confidence=config.confidence,
                    )
                canvas_index = candidates.sequence_positions + low - prompt_length
                with timer("verifier"):
                    verdict = dream_verify(
                        verifier, verifier_base(),
                        (canvas_index + verifier_offset).to(dream.device),
                        candidates.token_ids.to(dream.device),
                        tau_nats=tau, group_rows=config.group_rows, veto=config.veto,
                    )
                accepted_local = candidates.sequence_positions[: verdict.accepted]
                committed = verdict.tokens.to(device)
                state[0, accepted_local] = committed
                canvas[:, low:high] = state
                accepted_conditional = (
                    [verdict.head_log_prob] if verdict.vetoed
                    else [float(v) for v in verdict.conditional[: verdict.accepted].tolist()]
                )
                for k, lp in zip((canvas_index[: verdict.accepted]).tolist(), accepted_conditional):
                    commit_logp[int(k)] = lp
                steps += 1
                histogram[verdict.accepted - 1] += 1
                counters["dream_verify_rows"] += verdict.rows
                counters["dream_verify_passes"] += verdict.passes
                counters["veto_count"] += int(verdict.vetoed)
                if capture_traces:
                    traces.append({
                        "step_index": steps - 1, "block_index": index,
                        "remaining_masks_before": int(remaining.numel()),
                        "candidate_positions": [int(v) for v in canvas_index.tolist()],
                        "candidate_token_ids": [int(v) for v in candidates.token_ids.tolist()],
                        "candidate_confidences": [float(v) for v in candidates.confidences.tolist()],
                        "marginal_log_factors": [float(v) for v in verdict.marginal.tolist()],
                        "conditional_log_factors": [float(v) for v in verdict.conditional.tolist()],
                        "error_by_prefix_nats": [float(v) for v in verdict.errors.tolist()],
                        "threshold_delta": config.delta, "threshold_tau_nats": tau,
                        "vetoed": verdict.vetoed, "dream_head_argmax": verdict.head_argmax,
                        "accepted_prefix_length": verdict.accepted,
                        "committed_positions": [int(v) for v in canvas_index[: verdict.accepted].tolist()],
                        "committed_token_ids": [int(v) for v in committed.tolist()],
                        "dream_rows": verdict.rows,
                    })
            if config.early_stop:
                finished = finished_at(canvas[0, region], mask_token_id=sdar.mask_token_id, stop_ids=stop_ids)
            block_done = not bool(((state[0] == sdar.mask_token_id) & in_region).any())

            if (finished is not None or block_done) and not remask_used:
                remask_used = True
                committed_local = torch.nonzero(
                    (state[0] != sdar.mask_token_id) & in_region, as_tuple=False
                ).flatten()
                check_index = committed_local + low - prompt_length
                if finished is not None:
                    keep = check_index <= finished
                    committed_local, check_index = committed_local[keep], check_index[keep]
                if check_index.numel():
                    with timer("remask"):
                        outcome = remask_pass(
                            verifier, verifier_base(), check_index.to(dream.device),
                            (check_index + verifier_offset).to(dream.device),
                            state[0, committed_local].to(dream.device),
                            bottom=config.remask_bottom, group_rows=config.group_rows,
                            stop_ids=stop_ids,
                        )
                    counters["remask_passes"] += 1
                    counters["remask_rows"] += outcome.rows
                    counters["remask_forward_passes"] += outcome.passes
                    if outcome.chosen is None:
                        counters["remask_skipped"] += 1
                    else:
                        local = outcome.chosen + prompt_length - low
                        state[0, local] = outcome.new_token
                        canvas[:, low:high] = state
                        counters["tokens_changed_by_remask"] += 1
                        picked = outcome.checked_positions.index(outcome.chosen)
                        remask_edits.append({
                            "block_index": index, "canvas_index": outcome.chosen,
                            "old_token": outcome.old_token, "new_token": outcome.new_token,
                            "log_p_old": outcome.log_probs[picked], "bottom": outcome.bottom,
                        })
                    if capture_traces:
                        traces.append({
                            "step_index": steps - 1, "block_index": index, "remask": True,
                            "trigger": "stop" if finished is not None else "block_complete",
                            "checked_positions": outcome.checked_positions,
                            "log_probs": outcome.log_probs, "argmax": outcome.argmax,
                            "commit_time_log_probs": [
                                commit_logp.get(k, float("nan")) for k in outcome.checked_positions
                            ],
                            "bottom": outcome.bottom, "survivors": outcome.survivors,
                            "chosen": outcome.chosen, "old_token": outcome.old_token,
                            "new_token": outcome.new_token, "dream_rows": outcome.rows,
                        })
                    if config.early_stop:
                        finished = finished_at(
                            canvas[0, region], mask_token_id=sdar.mask_token_id, stop_ids=stop_ids
                        )
                    block_done = not bool(((state[0] == sdar.mask_token_id) & in_region).any())

            if finished is not None:
                early_stopped = True
                break
            if block_done:
                break

        canvas[:, low:high] = state
        if early_stopped:
            break
        commit(low, high)

    continuation = canvas[0, region].detach().cpu().tolist()
    scored, eos_position, stop_token = truncate_at_stop(continuation, stop_ids)
    early_stopped = early_stopped and sdar.mask_token_id in continuation
    counters["dream_rounds"] = counters["dream_verify_passes"] + counters["remask_forward_passes"]
    return GenerationResult(
        continuation_token_ids=[int(v) for v in scored],
        traces=traces,
        denoising_steps=steps,
        proposer_passes=steps,
        verifier_rows=counters["dream_verify_rows"],
        verifier_forward_passes=counters["dream_verify_passes"],
        commitment_histogram=histogram,
        eos_position=eos_position,
        maximum_length_termination=eos_position is None,
        full_continuation_token_ids=[int(v) for v in continuation],
        commit_passes=commit_passes,
        stage_seconds=timer.totals(),
        stop_token_id=stop_token,
        early_stopped=early_stopped,
        decoded_positions=sum(1 for v in continuation if v != sdar.mask_token_id),
        extras={**counters, "remask_edits": remask_edits},
    )
