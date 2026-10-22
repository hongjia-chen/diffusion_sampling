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
* Prefix plus one (opt-in, `subset_rule="prefix_plus_one"`, 2026-10-18): row m
  holds Dream's p(y_s | c, y_1..y_m) at every later candidate s, not only at
  m+1. When the prefix stops at a failing candidate m+1, the chain rule gives
  the joint of {1..m, s} exactly from rows 0..m. Of the later s whose set
  passes the same test, the one with the smallest set log-ratio is committed
  too (the user's choice, 2026-10-18; an exact tie goes to the more confident).
  No extra forward; at most one extra token per step, since no row conditions
  on a set holding s. `extra_min_confidence` (2026-10-18) lets only candidates
  the proposer gives at least that probability be the extra: unfloored, extras
  were low-confidence guesses (median 0.78 at δ=0.03) that pinned a slot before
  the masked slots between it and the prefix were decided, and the B=16 dev run
  lost 11-40 points to the prefix rule.
* One-sided test (opt-in, `pmi_test="one_sided"`, 2026-10-18): each candidate
  after the first passes when log p_D(y_i | c, y_<i) - log p_D(y_i | c) >= -log(1+delta),
  token by token. Only conflict (the higher-ranked tokens making y_i less
  likely) stops the prefix; support (making it more likely) no longer does. In
  the B=16 prefix-run traces, 78-79% of the failures that ended a step were
  support. Per token rather than cumulative, so a strongly supportive pair
  cannot bank credit for a later conflicting token. Not combined with plus one.
* Dynamic first (opt-in, `dynamic_threshold`, 2026-10-18, the user's design):
  the candidates the proposer gives more than the threshold (a prefix of the
  ranking) are committed without a test, as SDAR's dynamic decoder commits
  them. The unchanged MI rule then runs on the rest of the ranking from the
  canvas with those filled, with marginals measured on that canvas. Its first
  candidate is committed whatever the verifier says, as always, so a step with
  masks left past the confident ones commits at least one more than dynamic
  would. When every candidate is confident there is no verifier call; when
  none is, the step is the plain MI rule. Two-sided prefix rule only (no plus
  one, one-sided test, veto or remask).
* Anchored band (opt-in, `band_floor` with `dynamic_threshold`, 2026-10-20,
  the user's design): the confident candidates still all commit, but only
  those above the least confident one go onto the canvas untested. That last
  one is the MI rule's first candidate (the anchor), so its forced commit is
  one dynamic would make anyway. The rule then tests only the band, the
  candidates above `band_floor`; those at or below it are never offered. With
  no confident candidate the anchor is the top candidate, as in dynamic's
  one-token fallback; with an empty band there is no verifier call. Every step
  therefore commits what dynamic at the threshold commits, plus whatever of the
  band passes: δ = 0 is dynamic at the threshold (up to exact ties) and a δ
  large enough to pass everything is dynamic at the floor. In the dynamic-first
  B=16 runs the untested first candidate past the confident ones (median
  confidence 0.85, ~10% below 0.5) cost MATH ~9 points already at δ = 0.03.
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

from sdar_harness.blockwise import BlockCacheProvider
from sdar_harness.model import block_causal_mask
from sdar_harness.scheduler import (
    GenerationConfig,
    GenerationResult,
    _filtered_log_probs,
    _forward_slices,
    _propose,
    _uses_lean_path,
    delta_label,
    finished_at,
    stop_token_ids,
    truncate_at_stop,
)
from sdar_harness.timing import NULL_TIMER, SpanClock, StageTimer


# The default group size for every verifier forward (DESIGN.md F1): per-row
# time is flat from 3-4 rows up on an A4000, so 4 is the smallest group that
# reaches it. run_benchmark's --verifier-group-rows overrides it (2026-10-15,
# for B=32; MI_decoding/bench_verifier_rows.py). In fp16/bf16 the group size can
# move a borderline acceptance, so it must be identical across compared arms.
DREAM_GROUP_ROWS = 4

# Which candidate sets the MI rule may commit: the confidence-ordered prefixes
# (DESIGN.md F1), or those plus one later candidate (module docstring).
SUBSET_RULES = ("prefix", "prefix_plus_one")
# How a prefix is tested: |cumulative log-ratio| <= tau (DESIGN.md F1), or each
# token's log-ratio >= -tau (module docstring).
PMI_TESTS = ("two_sided", "one_sided")


@dataclass(frozen=True)
class CrossConfig(GenerationConfig):
    """Dream-verified adaptive delta. `method` stays "adaptive"."""

    # Off for exp05's runs (decided 2026-09-19); kept as an ablation.
    veto: bool = False
    remask: bool = False
    remask_bottom: int = 4
    group_rows: int = DREAM_GROUP_ROWS
    subset_rule: str = "prefix"
    extra_min_confidence: float = 0.0
    pmi_test: str = "two_sided"
    # Dynamic first (module docstring); 0 is off.
    dynamic_threshold: float = 0.0
    # Anchored band (module docstring): the MI rule sees the anchor and the
    # candidates above this floor only. 0 is off (dynamic first as above).
    band_floor: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.method != "adaptive":
            raise ValueError("CrossConfig is an adaptive method")
        if self.remask_bottom < 1 or self.group_rows < 1:
            raise ValueError("remask_bottom and group_rows must be positive")
        if self.subset_rule not in SUBSET_RULES:
            raise ValueError(f"subset_rule must be one of {SUBSET_RULES}")
        if not 0.0 <= self.extra_min_confidence <= 1.0:
            raise ValueError("extra_min_confidence must be in [0, 1]")
        if self.extra_min_confidence and self.subset_rule != "prefix_plus_one":
            raise ValueError("extra_min_confidence applies to subset_rule prefix_plus_one only")
        if self.pmi_test not in PMI_TESTS:
            raise ValueError(f"pmi_test must be one of {PMI_TESTS}")
        if self.pmi_test == "one_sided" and self.subset_rule != "prefix":
            raise ValueError("the one-sided test is for subset_rule prefix only")
        if self.dynamic_threshold:
            if not 0 < self.dynamic_threshold < 1:
                raise ValueError("dynamic_threshold must be in (0, 1), or 0 for off")
            if (self.subset_rule != "prefix" or self.pmi_test != "two_sided"
                    or self.veto or self.remask):
                raise ValueError("dynamic first takes the two-sided prefix rule without veto or remask")
        if self.band_floor:
            if not self.dynamic_threshold:
                raise ValueError("band_floor needs dynamic_threshold (the anchor comes from the confident set)")
            if not 0 < self.band_floor <= self.dynamic_threshold:
                raise ValueError("band_floor must be in (0, dynamic_threshold], or 0 for off")

    @property
    def label(self) -> str:
        value = delta_label(self.delta)
        kind = "xadaptive_remask" if self.remask else "xadaptive"
        if self.dynamic_threshold:
            kind += f"_dyn{round(100 * self.dynamic_threshold):02d}"
            if self.band_floor:
                kind += f"_band{round(100 * self.band_floor):02d}"
        if self.pmi_test == "one_sided":
            kind += "_os"
        if self.subset_rule == "prefix_plus_one":
            kind += "_plus1"
            if self.extra_min_confidence:
                kind += f"_c{round(100 * self.extra_min_confidence):02d}"
        suffix = "" if self.veto else "_noveto"
        return f"{kind}{suffix}_delta_{value}"


@dataclass(frozen=True)
class VerifyResult:
    accepted: int                  # tokens committed: the prefix, plus the extra if any
    tokens: torch.Tensor           # the committed tokens, Dream's argmax if vetoed
    marginal: torch.Tensor         # log p_D(y_i | c) for scored candidates
    conditional: torch.Tensor      # log p_D(y_i | c, y_<i) for scored candidates
    errors: torch.Tensor
    rows: int
    passes: int
    vetoed: bool
    head_argmax: int
    head_log_prob: float           # Dream's log-probability of its own argmax there
    # Prefix plus one. `committed` holds the candidate ranks committed, the
    # prefix 0..prefix-1 then the extra; `extra` is the extra's rank or -1,
    # `extra_passing` how many later ranks passed (the test and the confidence
    # floor) and `extra_first` the most confident of them (-1 if none), which the
    # confidence-order rule would take.
    # `later_conditional` is log p_D(y_s | c, y_1..y_prefix) for ranks after
    # the failing one, `later_errors` the test value of each {prefix, s} set;
    # both are empty unless the prefix stopped short of the last candidate.
    prefix: int = 0
    committed: torch.Tensor | None = None
    extra: int = -1
    extra_passing: int = 0
    extra_first: int = -1
    extra_log_prob: float = float("nan")
    later_conditional: torch.Tensor | None = None
    later_errors: torch.Tensor | None = None


def _prefix_test(conditional: torch.Tensor, marginal: torch.Tensor, tau_nats: float, one_sided: bool):
    """|cumulative log-ratio| of each prefix, and which candidates pass.

    Two-sided, a candidate passes when its prefix's |cumulative log-ratio| is
    within tau; one-sided, when its own log-ratio is at least -tau. The first
    candidate always passes. `errors` is the two-sided quantity either way, so
    traces stay comparable.
    """
    errors = (conditional.cumsum(0) - marginal.cumsum(0)).abs()
    passing = (conditional - marginal >= -tau_nats) if one_sided else (errors <= tau_nats)
    passing[0] = True
    return errors, passing


def _nothing_to_verify(device: torch.device, dtype: torch.dtype) -> VerifyResult:
    """The verdict on an empty candidate list (dynamic first took them all)."""
    empty = torch.empty(0, device=device)
    return VerifyResult(
        accepted=0, tokens=torch.empty(0, dtype=dtype, device=device), marginal=empty,
        conditional=empty, errors=empty, rows=0, passes=0, vetoed=False, head_argmax=-1,
        head_log_prob=float("nan"), prefix=0, committed=torch.zeros(0, dtype=torch.long, device=device),
    )


def _forward_kept(provider: Any, input_ids: torch.Tensor, gathers: list[tuple[torch.Tensor, torch.Tensor]]):
    """`_forward_slices`, plus a reader for more (row, position) logits of the same forward.

    The gathers are computed exactly as `_forward_slices` computes them, so the
    prefix test sees the same numbers under either subset rule.
    """
    if _uses_lean_path(provider):
        hidden = provider.hidden_states(input_ids)
        read = lambda rows, positions: provider.logits_from_hidden(hidden, rows, positions)  # noqa: E731
    else:
        full = provider.forward_logits(input_ids)
        read = lambda rows, positions: full[rows, positions, :]  # noqa: E731
    return [read(rows, positions) for rows, positions in gathers], read


def dream_verify(
    dream: Any,
    base: torch.Tensor,
    positions: torch.Tensor,
    tokens: torch.Tensor,
    *,
    tau_nats: float,
    group_rows: int,
    veto: bool,
    plus_one: bool = False,
    confidences: torch.Tensor | None = None,
    min_confidence: float = 0.0,
    one_sided: bool = False,
    timer: StageTimer = NULL_TIMER,
) -> VerifyResult:
    """Score SDAR's ranked candidates with Dream, in groups, with early exit.

    `base` is Dream's [1, S] state; `positions` are Dream positions of the
    ranked candidates and `tokens` their SDAR tokens, all on Dream's device.
    Profiled as "verify_forward" (each verifier model call) and "verify_test"
    (building the probe rows, the log-probabilities and the prefix test; one
    call per step). With `plus_one`, the extra candidate's output-head read and
    test also count as "verify_test". `confidences` are the proposer's
    probabilities of the ranked tokens; with `min_confidence`, only later
    candidates at or above it may be the extra.
    """
    count = int(tokens.numel())
    if count == 1 and not veto:
        # The prefix rule always commits the first candidate, so a lone
        # candidate is accepted whatever the verifier says: skip its forward.
        # At B=4 this is the last masked slot of most blocks.
        with timer("verify_test"):
            empty = torch.empty(0, device=positions.device)
            return VerifyResult(
                accepted=1, tokens=tokens, marginal=empty, conditional=empty, errors=empty,
                rows=0, passes=0, vetoed=False, head_argmax=-1, head_log_prob=float("nan"),
                prefix=1, committed=torch.zeros(1, dtype=torch.long, device=positions.device),
            )
    with timer("verify_test"):
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
    read, read_start = None, 0
    for start in range(0, count, group_rows):
        chunk = probes[start : start + group_rows]
        width = int(chunk.shape[0])
        gathers = [(torch.arange(width, device=device), positions[start : start + width])]
        if start == 0:
            gathers.append((torch.zeros_like(positions), positions))
        with timer("verify_forward"):
            if plus_one:
                outputs, read = _forward_kept(dream, chunk, gathers)
                read_start = start
            else:
                outputs = _forward_slices(dream, chunk, gathers)
        passes += 1
        scored = start + width
        with timer("verify_test", count=False):
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
                vetoed = veto and head_argmax != int(tokens[0].item())
            del log_probs
            if not vetoed:
                conditional = torch.cat(conditional_parts)
                errors, passing = _prefix_test(conditional, marginal[:scored], tau_nats, one_sided)
                all_pass = bool(passing.all())
        if vetoed or not all_pass:
            break
    assert marginal is not None
    extra, extra_log_prob, extra_passing, extra_first = -1, float("nan"), 0, -1
    later_conditional = later_errors = None
    with timer("verify_test", count=False):
        conditional = torch.cat(conditional_parts)
        errors, passing = _prefix_test(conditional, marginal[:scored], tau_nats, one_sided)
        if vetoed:
            accepted = prefix = 1
            chosen = torch.zeros(1, dtype=torch.long, device=device)
            committed = torch.tensor([head_argmax], device=device, dtype=tokens.dtype)
        else:
            accepted = prefix = int(torch.cumprod(passing.to(torch.int64), dim=0).sum().item())
            chosen = torch.arange(prefix, device=device)
            if plus_one and prefix + 1 < count:
                # Candidate `prefix` failed, and its conditional came from row
                # `prefix` (candidates 0..prefix-1 filled), the last group's. That
                # row's view of every later candidate s gives the set
                # {0..prefix-1, s} by the chain rule. Of the sets that pass, commit
                # the one with the smallest log-ratio; argmin takes the first of
                # equal values, so an exact tie goes to the more confident.
                assert read is not None and read_start <= prefix < read_start + group_rows
                later = positions[prefix + 1 :]
                logits = read(torch.full_like(later, prefix - read_start), later)
                later_conditional = _filtered_log_probs(logits, excluded).gather(
                    1, tokens[prefix + 1 :].reshape(-1, 1)
                ).squeeze(1)
                held = conditional[:prefix].sum() - marginal[:prefix].sum()
                later_errors = (held + later_conditional - marginal[prefix + 1 :]).abs()
                eligible = later_errors <= tau_nats
                if min_confidence > 0:
                    assert confidences is not None
                    eligible &= confidences[prefix + 1 :].to(device) >= min_confidence
                hits = torch.nonzero(eligible).flatten()
                extra_passing = int(hits.numel())
                if extra_passing:
                    extra_first = prefix + 1 + int(hits[0].item())
                    ranked = torch.where(eligible, later_errors, torch.full_like(later_errors, math.inf))
                    extra = prefix + 1 + int(torch.argmin(ranked).item())
                    extra_log_prob = float(later_conditional[extra - prefix - 1].item())
                    chosen = torch.cat((chosen, torch.tensor([extra], device=device)))
                    accepted += 1
            committed = tokens[chosen]
    return VerifyResult(
        accepted=accepted, tokens=committed, marginal=marginal[:scored],
        conditional=conditional, errors=errors, rows=scored, passes=passes,
        vetoed=vetoed, head_argmax=head_argmax, head_log_prob=head_log_prob,
        prefix=prefix, committed=chosen, extra=extra, extra_log_prob=extra_log_prob,
        extra_passing=extra_passing, extra_first=extra_first,
        later_conditional=later_conditional, later_errors=later_errors,
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

    def commit(low: int, high: int, stage: str = "commit") -> None:
        nonlocal commit_passes
        with timer(stage):
            sdar.commit(canvas[:, low:high], attention[low:high, :high], all_positions[low:high], cache)
            if block_verifier:
                dream.commit(
                    canvas[:, low:high].to(dream.device), attention[low:high, :high].to(dream.device),
                    all_positions[low:high].to(dream.device), verifier_cache,
                )
        commit_passes += 1

    # Both models' prompt commits are the prefill; Dream has none (it
    # re-encodes the whole canvas every forward).
    prefill_clock = SpanClock(
        list(getattr(sdar, "devices", [sdar.device])) + list(getattr(dream, "devices", [dream.device]))
    )
    prefill_clock.start()
    if prefill_length:
        commit(0, prefill_length, stage="prefill")
    prefill_clock.stop()

    traces: list[dict[str, Any]] = []
    histogram = [0] * config.max_candidates
    steps = 0
    counters = dict(
        dream_verify_rows=0, dream_verify_passes=0, veto_count=0,
        remask_passes=0, remask_rows=0, remask_forward_passes=0,
        remask_skipped=0, tokens_changed_by_remask=0,
        # What bounded each step: candidates on offer (every masked slot left
        # in the block, up to max_candidates), steps with only one, and steps
        # where the rule passed all of several, so only the block stopped it.
        candidates_offered=0, lone_candidate_steps=0, all_candidates_accepted_steps=0,
    )
    plus_one = config.subset_rule == "prefix_plus_one"
    if plus_one:
        # Steps whose prefix stopped at a failing candidate with later ones left
        # (where the extra is tried), steps that committed an extra, steps where
        # two or more later candidates passed (a real choice), and steps where
        # the smallest log-ratio was not the most confident passing candidate.
        counters.update(extra_tried_steps=0, extra_committed_steps=0,
                        extra_choice_steps=0, extra_not_most_confident_steps=0)
    dynamic_first = bool(config.dynamic_threshold)
    banded = bool(config.band_floor)
    if dynamic_first:
        # Tokens committed untested ahead of the MI rule, and steps with any.
        counters.update(dynamic_committed_tokens=0, dynamic_steps=0)
    if banded:
        # Steps whose anchor came from the confident set (with the untested
        # ones, that set's size), band candidates offered to the MI rule and
        # committed, and steps with a band (each a verifier call).
        counters.update(anchored_steps=0, band_offered_tokens=0, band_committed_tokens=0, band_steps=0)
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
                candidates = _propose(
                    scoped, state, remaining, config.max_candidates,
                    temperature=config.temperature, top_p=config.top_p,
                    confidence=config.confidence, timer=timer,
                )
                # Dynamic first: the leading candidates above the threshold go
                # onto the canvas before the verifier sees it, so the MI rule's
                # marginals and conditionals both condition on them. Anchored
                # band: all but the last of them do; the MI rule's candidates
                # end at `band_end`, the last one above the floor (at least its
                # first). Timed with the step's decision.
                dynamic_count = confident = 0
                band_end = int(candidates.token_ids.numel())
                if dynamic_first:
                    with timer("decide", count=False):
                        confident = int((candidates.confidences > config.dynamic_threshold).sum().item())
                        dynamic_count = confident
                        if banded:
                            dynamic_count = max(confident - 1, 0)
                            above_floor = int((candidates.confidences > config.band_floor).sum().item())
                            band_end = max(dynamic_count + 1, above_floor)
                        if dynamic_count:
                            state[0, candidates.sequence_positions[:dynamic_count]] = (
                                candidates.token_ids[:dynamic_count]
                            )
                            canvas[:, low:high] = state
                # Handing the candidates and the canvas to the verifier's card
                # is part of verifying; dream_verify counts the call.
                with timer("verify_test", count=False):
                    canvas_index = candidates.sequence_positions + low - prompt_length
                    verifier_state = verifier_base()
                    verifier_positions = (canvas_index[dynamic_count:band_end] + verifier_offset).to(dream.device)
                    verifier_tokens = candidates.token_ids[dynamic_count:band_end].to(dream.device)
                if dynamic_count == band_end:
                    # Every candidate was confident: nothing is left to test.
                    verdict = _nothing_to_verify(dream.device, verifier_tokens.dtype)
                else:
                    verdict = dream_verify(
                        verifier, verifier_state, verifier_positions, verifier_tokens,
                        tau_nats=tau, group_rows=config.group_rows, veto=config.veto,
                        plus_one=plus_one,
                        confidences=candidates.confidences[dynamic_count:band_end].to(dream.device),
                        min_confidence=config.extra_min_confidence,
                        one_sided=config.pmi_test == "one_sided", timer=timer,
                    )
                with timer("decide"):
                    # Ranks into the whole candidate list: the dynamic ones, then
                    # the MI rule's, which index the candidates after them.
                    picked = verdict.committed.to(device)
                    committed = verdict.tokens.to(device)
                    if dynamic_count:
                        picked = torch.cat((torch.arange(dynamic_count, device=device), picked + dynamic_count))
                        committed = torch.cat((candidates.token_ids[:dynamic_count], committed))
                    accepted_local = candidates.sequence_positions[picked]
                    state[0, accepted_local] = committed
                    canvas[:, low:high] = state
                    accepted_conditional = (
                        [verdict.head_log_prob] if verdict.vetoed
                        else [float(v) for v in verdict.conditional[: verdict.prefix].tolist()]
                        + ([verdict.extra_log_prob] if verdict.extra >= 0 else [])
                    )
                    committed_index = canvas_index[picked]
                    # The dynamic tokens have no verifier score at commit time.
                    for k, lp in zip(committed_index[dynamic_count:].tolist(), accepted_conditional):
                        commit_logp[int(k)] = lp
                accepted = dynamic_count + verdict.accepted
                steps += 1
                histogram[accepted - 1] += 1
                counters["dream_verify_rows"] += verdict.rows
                counters["dream_verify_passes"] += verdict.passes
                counters["veto_count"] += int(verdict.vetoed)
                offered = int(candidates.token_ids.numel())
                counters["candidates_offered"] += offered
                if offered == 1:
                    counters["lone_candidate_steps"] += 1
                elif accepted == offered:
                    counters["all_candidates_accepted_steps"] += 1
                if dynamic_first:
                    counters["dynamic_committed_tokens"] += dynamic_count
                    counters["dynamic_steps"] += int(dynamic_count > 0)
                if banded:
                    band = band_end - dynamic_count - 1
                    counters["anchored_steps"] += int(confident > 0)
                    counters["band_offered_tokens"] += band
                    counters["band_committed_tokens"] += verdict.accepted - 1
                    counters["band_steps"] += int(band > 0)
                if plus_one:
                    counters["extra_tried_steps"] += int(verdict.later_errors is not None)
                    counters["extra_committed_steps"] += int(verdict.extra >= 0)
                    counters["extra_choice_steps"] += int(verdict.extra_passing >= 2)
                    counters["extra_not_most_confident_steps"] += int(verdict.extra != verdict.extra_first)
                if capture_traces:
                    # Per-rank lists stay aligned with the candidate ranking; the
                    # dynamic ranks have no verifier scores (null), and the
                    # accepted prefix counts them, since they lead the ranking.
                    untested = [None] * dynamic_count
                    trace = {
                        "step_index": steps - 1, "block_index": index,
                        "remaining_masks_before": int(remaining.numel()),
                        "candidate_positions": [int(v) for v in canvas_index.tolist()],
                        "candidate_token_ids": [int(v) for v in candidates.token_ids.tolist()],
                        "candidate_confidences": [float(v) for v in candidates.confidences.tolist()],
                        "marginal_log_factors": untested + [float(v) for v in verdict.marginal.tolist()],
                        "conditional_log_factors": untested + [float(v) for v in verdict.conditional.tolist()],
                        "error_by_prefix_nats": untested + [float(v) for v in verdict.errors.tolist()],
                        "threshold_delta": config.delta, "threshold_tau_nats": tau,
                        "vetoed": verdict.vetoed, "dream_head_argmax": verdict.head_argmax,
                        "accepted_prefix_length": dynamic_count + verdict.prefix,
                        "committed_positions": [int(v) for v in committed_index.tolist()],
                        "committed_token_ids": [int(v) for v in committed.tolist()],
                        "dream_rows": verdict.rows,
                    }
                    if plus_one:
                        # Ranks after the failing one: Dream's conditional given
                        # the prefix, and the test value of prefix + that rank.
                        trace.update({
                            "extra_rank": verdict.extra,
                            "later_conditional_log_factors": (
                                [] if verdict.later_conditional is None
                                else [float(v) for v in verdict.later_conditional.tolist()]
                            ),
                            "later_set_error_nats": (
                                [] if verdict.later_errors is None
                                else [float(v) for v in verdict.later_errors.tolist()]
                            ),
                        })
                    if dynamic_first:
                        trace["dynamic_committed"] = dynamic_count
                    if banded:
                        # The MI rule saw ranks dynamic_committed..band_end-1;
                        # later ranks were at or below the floor.
                        trace["anchored"] = confident > 0
                        trace["band_end"] = band_end
                    traces.append(trace)
            with timer("stop_check"):
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
        stage_counts=timer.counts(),
        stop_token_id=stop_token,
        early_stopped=early_stopped,
        decoded_positions=sum(1 for v in continuation if v != sdar.mask_token_id),
        extras={**counters, "remask_edits": remask_edits},
        prefill_seconds=prefill_clock.seconds(),
    )
