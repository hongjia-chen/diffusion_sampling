## Scheduled sampling algorithm: Proposals, verifications, acceptance
## Fixed-k commitments, EOS handling, etc.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import torch


class LogitsProvider(Protocol):
    mask_token_id: int
    eos_token_id: int
    excluded_token_ids: set[int]
    device: torch.device

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor: ...


# Number of canvas positions projected through the vocabulary head at once.
# The proposer needs a filtered log-softmax over every masked position, which
# in float32 costs 4 * V bytes per row (~0.6 MB at V=152064). Chunking keeps
# that transient bounded instead of scaling with continuation_length.
PROPOSER_VOCAB_CHUNK = 64
# Nucleus filtering sorts the whole vocabulary, and the int64 index tensor
# `torch.sort` returns is twice the size of the float32 logits it sorts. At
# V=152064 that is ~117 MB of transient per 64-position chunk, which is the
# largest single allocation in the proposer. Chunk more finely when top_p is on.
PROPOSER_VOCAB_CHUNK_TOP_P = 16


def _proposer_chunk(top_p: float | None) -> int:
    return PROPOSER_VOCAB_CHUNK if top_p is None or top_p >= 1 else PROPOSER_VOCAB_CHUNK_TOP_P


@dataclass(frozen=True)
class GenerationConfig:
    method: Literal["fixed", "adaptive"]
    continuation_length: int
    fixed_k: int | None = None
    delta: float | None = None
    max_candidates: int = 8
    verifier_microbatch_size: int = 8
    # Proposal sampling. temperature=0.0 is argmax, which is the historical
    # behaviour; the Dream/ADJUST reference uses 1.0 with top_p 0.9. These
    # affect the shared proposer only, never the acceptance rule.
    temperature: float = 0.0
    top_p: float | None = None
    # Which quantity ranks candidate positions. "logp" is the probability of
    # the selected token (Dream's maskgit_plus); "entropy" is the negative
    # entropy of the whole filtered distribution (Dream's alg="entropy", the
    # only value ADJUST permits). Position ranking only -- the committed token
    # is drawn the same way either way.
    confidence: Literal["logp", "entropy"] = "logp"

    def __post_init__(self) -> None:
        if self.continuation_length < 1:
            raise ValueError("continuation_length must be positive")
        if self.max_candidates < 1 or self.verifier_microbatch_size < 1:
            raise ValueError("candidate and microbatch sizes must be positive")
        if self.method == "fixed":
            if self.fixed_k is None or self.fixed_k < 1:
                raise ValueError("fixed methods require fixed_k >= 1")
            if self.fixed_k > self.max_candidates:
                raise ValueError("fixed_k cannot exceed max_candidates")
            if self.delta is not None:
                raise ValueError("fixed methods cannot set delta")
        elif self.method == "adaptive":
            if self.fixed_k is not None:
                raise ValueError("adaptive methods cannot set fixed_k")
            if self.delta is None or not math.isfinite(self.delta) or self.delta < 0:
                raise ValueError("adaptive methods require finite delta >= 0")
        else:
            raise ValueError(f"unknown generation method: {self.method}")
        if self.confidence not in ("logp", "entropy"):
            raise ValueError(f"unknown confidence metric: {self.confidence}")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must lie in (0, 1]")

    @property
    def tau_nats(self) -> float | None:
        return None if self.delta is None else math.log1p(self.delta)

    @property
    def label(self) -> str:
        if self.method == "fixed":
            return f"fixed_k_{self.fixed_k}"
        value = f"{self.delta:.2f}".replace(".", "p")
        return f"adaptive_delta_{value}"


@dataclass(frozen=True)
class CandidateSet:
    sequence_positions: torch.Tensor
    token_ids: torch.Tensor
    confidences: torch.Tensor


@dataclass(frozen=True)
class StepResult:
    state: torch.Tensor
    trace: dict[str, Any]
    accepted_prefix_length: int
    verifier_rows: int
    verifier_forward_passes: int


@dataclass(frozen=True)
class GenerationResult:
    continuation_token_ids: list[int]
    traces: list[dict[str, Any]]
    denoising_steps: int
    proposer_passes: int
    verifier_rows: int
    verifier_forward_passes: int
    commitment_histogram: list[int]
    eos_position: int | None
    maximum_length_termination: bool
    # Full untruncated canvas, for diagnostics only. `continuation_token_ids`
    # is truncated at the first EOS, which discards the evidence needed to tell
    # a truncation artifact from a genuinely short generation.
    full_continuation_token_ids: list[int] = field(default_factory=list)


def _filtered_log_probs(logits: torch.Tensor, excluded: set[int]) -> torch.Tensor:
    # `.to(float32, copy=True)` replaces `.float().clone()`: the old form made a
    # second full-size copy whenever the input was already float32, and two
    # copies of [512, 152064] float32 is 622 MB.
    scoring = logits.to(torch.float32, copy=True)
    valid_excluded = sorted(index for index in excluded if 0 <= index < scoring.shape[-1])
    if len(valid_excluded) >= scoring.shape[-1]:
        raise ValueError("Token filtering removed the entire vocabulary")
    if valid_excluded:
        indices = torch.tensor(valid_excluded, device=scoring.device, dtype=torch.long)
        scoring.index_fill_(-1, indices, -torch.inf)
    return torch.log_softmax(scoring, dim=-1)


def _top_p_filter(scores: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filter, following Dream's `top_p_logits`."""
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    cumulative = torch.cumsum(torch.softmax(sorted_scores, dim=-1), dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, sorted_indices, remove)
    return scores.masked_fill(mask, torch.finfo(scores.dtype).min)


ENTROPY_EPSILON = 1e-10


def _select_tokens(
    log_probs: torch.Tensor,
    temperature: float,
    top_p: float | None,
    confidence: str = "logp",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick one token per row and score the row for position ranking.

    Follows Dream's `sample_tokens`: temperature scaling, then nucleus
    filtering, then a categorical draw. At temperature 0 the draw degenerates
    to the argmax, identical to the pre-sampling behaviour.

    The returned score is what ranks positions against each other. With
    "logp" it is the probability of the drawn token; with "entropy" it is the
    negative entropy of the filtered distribution, so a position the model is
    peaked about outranks one it is spread over. Both are computed on the same
    post-filter distribution the token is drawn from, as Dream does.
    """
    scores = log_probs if temperature <= 0 else log_probs / temperature
    if top_p is not None and top_p < 1:
        scores = _top_p_filter(scores, top_p)
    probs = torch.softmax(scores, dim=-1)
    if temperature <= 0:
        selected, token_ids = probs.max(dim=-1)
    else:
        token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        selected = probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    if confidence == "entropy":
        selected = (probs * torch.log(probs + ENTROPY_EPSILON)).sum(dim=-1)
    return selected, token_ids


def _uses_lean_path(provider: LogitsProvider) -> bool:
    """True when the provider can project single positions to the vocabulary."""
    return bool(getattr(provider, "supports_lean_logits", False))


def _forward_slices(
    provider: LogitsProvider,
    input_ids: torch.Tensor,
    gathers: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[torch.Tensor]:
    """One model forward; return shifted logits for each (rows, positions) pair.

    Counts as exactly one forward pass regardless of how many gathers are
    requested, so verifier accounting is unchanged.
    """
    if _uses_lean_path(provider):
        hidden = provider.hidden_states(input_ids)
        return [
            provider.logits_from_hidden(hidden, rows, positions)
            for rows, positions in gathers
        ]
    full = provider.forward_logits(input_ids)
    return [full[rows, positions, :] for rows, positions in gathers]


def _remaining_positions(
    state: torch.Tensor,
    *,
    canvas_start: int,
    canvas_length: int,
    mask_token_id: int,
) -> torch.Tensor:
    canvas = state[0, canvas_start : canvas_start + canvas_length]
    return torch.nonzero(canvas == mask_token_id, as_tuple=False).flatten() + canvas_start


def _propose(
    provider: LogitsProvider,
    state: torch.Tensor,
    remaining: torch.Tensor,
    max_candidates: int,
    temperature: float = 0.0,
    top_p: float | None = None,
    confidence: str = "logp",
) -> CandidateSet:
    excluded = provider.excluded_token_ids
    if _uses_lean_path(provider):
        # log_softmax and max are both row-wise, so chunking over masked
        # positions is numerically identical to scoring them in one shot.
        hidden = provider.hidden_states(state)
        rows = torch.zeros_like(remaining)
        logp_parts: list[torch.Tensor] = []
        id_parts: list[torch.Tensor] = []
        chunk_size = _proposer_chunk(top_p)
        for start in range(0, int(remaining.numel()), chunk_size):
            stop = start + chunk_size
            chunk_logits = provider.logits_from_hidden(
                hidden, rows[start:stop], remaining[start:stop]
            )
            chunk_log_probs = _filtered_log_probs(chunk_logits, excluded)
            values, indices = _select_tokens(chunk_log_probs, temperature, top_p, confidence)
            logp_parts.append(values)
            id_parts.append(indices)
            del chunk_logits, chunk_log_probs
        top_logps = torch.cat(logp_parts)
        top_ids = torch.cat(id_parts)
        del hidden
    else:
        logits = provider.forward_logits(state)
        masked_logits = logits[0].index_select(0, remaining)
        log_probs = _filtered_log_probs(masked_logits, excluded)
        top_logps, top_ids = _select_tokens(log_probs, temperature, top_p, confidence)
    order = torch.argsort(top_logps, descending=True, stable=True)
    order = order[: min(max_candidates, remaining.numel())]
    return CandidateSet(
        sequence_positions=remaining.index_select(0, order),
        token_ids=top_ids.index_select(0, order),
        confidences=top_logps.index_select(0, order),
    )


def _score_all_prefixes(
    provider: LogitsProvider,
    state: torch.Tensor,
    candidates: CandidateSet,
    microbatch_size: int,
    tau_nats: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Score candidate prefixes until the acceptance rule is decided.

    Acceptance is the leading run of candidates whose joint-vs-marginal error
    stays within ``tau_nats``. Once one candidate violates it, every later
    candidate is rejected no matter what it scores, so the remaining
    microbatches carry no information and are skipped. With microbatch 8 and a
    typical acceptance of ~4 this is one forward pass per step instead of
    ``ceil(max_candidates / microbatch_size)``.

    Passing ``tau_nats=None`` scores every candidate, which is what the
    equivalence tests compare against.

    Returns the marginal and conditional log-factors and the error curve over
    the scored prefix, plus the number of forward passes and rows actually
    spent. The caller's acceptance arithmetic is unchanged: a truncated error
    curve always contains the first violation, so its cumprod is the same.
    """
    count = int(candidates.token_ids.numel())
    probes = state.repeat(count, 1)
    # Row j contains y_1...y_j and scores y_{j+1}; row zero is c_t.
    for row in range(1, count):
        probes[row, candidates.sequence_positions[:row]] = candidates.token_ids[:row]

    conditional_parts: list[torch.Tensor] = []
    marginal: torch.Tensor | None = None
    forward_passes = 0
    scored = 0
    device = candidates.sequence_positions.device
    for start in range(0, count, microbatch_size):
        chunk = probes[start : start + microbatch_size]
        row_indices = torch.arange(chunk.shape[0], device=device)
        positions = candidates.sequence_positions[start : start + chunk.shape[0]]
        # Each probe row is only ever read at its own candidate position, so
        # there is no reason to materialize [microbatch, sequence, vocabulary].
        gathers = [(row_indices, positions)]
        if start == 0:
            # Row zero is the unmodified base state c_t; the marginal factors
            # read it at every candidate position, so they are complete after
            # the first microbatch and survive any early exit.
            gathers.append(
                (
                    torch.zeros_like(candidates.sequence_positions),
                    candidates.sequence_positions,
                )
            )
        outputs = _forward_slices(provider, chunk, gathers)
        forward_passes += 1
        scored = start + chunk.shape[0]

        # Reduce each chunk to one log-factor per row immediately. Keeping only
        # [chunk, V] alive instead of [count, V] also drops the largest verifier
        # allocation by a factor of count/microbatch_size.
        chunk_ids = candidates.token_ids[start:scored].reshape(-1, 1)
        chunk_log_probs = _filtered_log_probs(outputs[0], provider.excluded_token_ids)
        conditional_parts.append(chunk_log_probs.gather(1, chunk_ids).squeeze(1))
        del chunk_log_probs
        if start == 0:
            marginal_log_probs = _filtered_log_probs(
                outputs[1], provider.excluded_token_ids
            )
            marginal = marginal_log_probs.gather(
                1, candidates.token_ids.reshape(-1, 1)
            ).squeeze(1)
            del marginal_log_probs

        if tau_nats is None:
            continue
        assert marginal is not None
        conditional = torch.cat(conditional_parts)
        errors = (conditional.cumsum(0) - marginal[:scored].cumsum(0)).abs()
        passing = errors <= tau_nats
        passing[0] = True
        if not bool(passing.all()):
            break

    if marginal is None:
        raise RuntimeError("Verifier produced no base-state logits")
    conditional = torch.cat(conditional_parts)
    marginal = marginal[:scored]
    errors = (conditional.cumsum(0) - marginal.cumsum(0)).abs()
    return marginal, conditional, errors, forward_passes, scored


def _first_eos_position(
    state: torch.Tensor,
    *,
    canvas_start: int,
    canvas_length: int,
    eos_token_id: int,
) -> int | None:
    canvas = state[0, canvas_start : canvas_start + canvas_length]
    positions = torch.nonzero(canvas == eos_token_id, as_tuple=False).flatten()
    if positions.numel() == 0:
        return None
    return int(positions.min().item())


def generation_step(
    provider: LogitsProvider,
    state: torch.Tensor,
    *,
    canvas_start: int,
    config: GenerationConfig,
    step_index: int,
    capture_trace: bool = True,
) -> StepResult:
    remaining = _remaining_positions(
        state,
        canvas_start=canvas_start,
        canvas_length=config.continuation_length,
        mask_token_id=provider.mask_token_id,
    )
    if remaining.numel() == 0:
        raise ValueError("generation_step called on a complete state")
    candidates = _propose(
        provider,
        state,
        remaining,
        config.max_candidates,
        temperature=config.temperature,
        top_p=config.top_p,
        confidence=config.confidence,
    )
    marginal: torch.Tensor | None = None
    conditional: torch.Tensor | None = None
    errors: torch.Tensor | None = None
    verifier_rows = 0
    verifier_forward_passes = 0
    if config.method == "fixed":
        accepted = min(int(config.fixed_k), int(candidates.token_ids.numel()))
    else:
        assert config.tau_nats is not None
        (
            marginal,
            conditional,
            errors,
            verifier_forward_passes,
            verifier_rows,
        ) = _score_all_prefixes(
            provider,
            state,
            candidates,
            config.verifier_microbatch_size,
            tau_nats=config.tau_nats,
        )
        accepted = 1
        passing = errors <= config.tau_nats
        passing[0] = True
        accepted = int(torch.cumprod(passing.to(torch.int64), dim=0).sum().item())

    next_state = state.clone()
    accepted_positions = candidates.sequence_positions[:accepted]
    accepted_tokens = candidates.token_ids[:accepted]
    next_state[0, accepted_positions] = accepted_tokens
    # EOS is an ordinary committed token during denoising. In particular, do
    # not fill the suffix with EOS: both schedulers must finish the same fixed
    # canvas, and the caller truncates the decoded completion only afterwards.
    eos_position = _first_eos_position(
        next_state,
        canvas_start=canvas_start,
        canvas_length=config.continuation_length,
        eos_token_id=provider.eos_token_id,
    )
    remaining_after = _remaining_positions(
        next_state,
        canvas_start=canvas_start,
        canvas_length=config.continuation_length,
        mask_token_id=provider.mask_token_id,
    )
    trace: dict[str, Any] = {}
    if capture_trace:
        trace = {
            "step_index": step_index,
            "remaining_masks_before": int(remaining.numel()),
            "candidate_positions": [
                int(value - canvas_start)
                for value in candidates.sequence_positions.detach().cpu()
            ],
            "candidate_token_ids": [
                int(value) for value in candidates.token_ids.detach().cpu()
            ],
            "candidate_confidences": [
                float(value) for value in candidates.confidences.detach().cpu()
            ],
            "marginal_log_factors": (
                [] if marginal is None else [float(value) for value in marginal.detach().cpu()]
            ),
            "conditional_log_factors": (
                []
                if conditional is None
                else [float(value) for value in conditional.detach().cpu()]
            ),
            "error_by_prefix_nats": (
                [] if errors is None else [float(value) for value in errors.detach().cpu()]
            ),
            "threshold_delta": config.delta,
            "threshold_tau_nats": config.tau_nats,
            "accepted_prefix_length": accepted,
            "committed_positions": [
                int(value - canvas_start) for value in accepted_positions.detach().cpu()
            ],
            "remaining_masks_after": int(remaining_after.numel()),
            "eos_position_after": eos_position,
        }
    return StepResult(
        state=next_state,
        trace=trace,
        accepted_prefix_length=accepted,
        verifier_rows=verifier_rows,
        verifier_forward_passes=verifier_forward_passes,
    )


def generate(
    provider: LogitsProvider,
    prompt_ids: torch.Tensor,
    config: GenerationConfig,
    *,
    capture_traces: bool = False,
) -> GenerationResult:
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("Authoritative latency generation requires one prompt at a time")
    canvas_start = prompt_ids.shape[1]
    canvas = torch.full(
        (1, config.continuation_length),
        provider.mask_token_id,
        dtype=torch.long,
        device=provider.device,
    )
    state = torch.cat((prompt_ids.to(provider.device), canvas), dim=1)
    traces: list[dict[str, Any]] = []
    step_count = 0
    verifier_rows = 0
    verifier_forward_passes = 0
    histogram = [0] * config.max_candidates
    while True:
        remaining = _remaining_positions(
            state,
            canvas_start=canvas_start,
            canvas_length=config.continuation_length,
            mask_token_id=provider.mask_token_id,
        )
        if remaining.numel() == 0:
            break
        step = generation_step(
            provider,
            state,
            canvas_start=canvas_start,
            config=config,
            step_index=step_count,
            capture_trace=capture_traces,
        )
        state = step.state
        step_count += 1
        if capture_traces:
            traces.append(step.trace)
        verifier_rows += step.verifier_rows
        verifier_forward_passes += step.verifier_forward_passes
        histogram[step.accepted_prefix_length - 1] += 1

    continuation = state[0, canvas_start:].detach().cpu().tolist()
    try:
        eos_position = continuation.index(provider.eos_token_id)
    except ValueError:
        eos_position = None
    scored_tokens = continuation if eos_position is None else continuation[:eos_position]
    return GenerationResult(
        continuation_token_ids=[int(value) for value in scored_tokens],
        traces=traces,
        denoising_steps=step_count,
        proposer_passes=step_count,
        verifier_rows=verifier_rows,
        verifier_forward_passes=verifier_forward_passes,
        commitment_histogram=histogram,
        eos_position=eos_position,
        maximum_length_termination=eos_position is None,
        full_continuation_token_ids=[int(value) for value in continuation],
    )
