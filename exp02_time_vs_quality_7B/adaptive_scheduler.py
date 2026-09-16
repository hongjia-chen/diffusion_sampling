"""Short-circuit joint-vs-marginal scheduler for Dream generation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import torch


@dataclass(frozen=True)
class AdaptiveSchedulerConfig:
    tau: float
    max_candidates: int = 8
    conditional_microbatch_size: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.tau) or self.tau < 0:
            raise ValueError("tau must be a finite nonnegative value.")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least 1.")
        if self.conditional_microbatch_size != 1:
            raise NotImplementedError(
                "Version one requires conditional_microbatch_size=1."
            )


@dataclass(frozen=True)
class AdaptiveStepResult:
    state: Any
    remaining_masks_before: int
    remaining_masks_after: int
    candidate_canvas_positions: torch.Tensor
    candidate_sequence_positions: torch.Tensor
    candidate_token_ids: torch.Tensor
    candidate_confidences: torch.Tensor
    marginal_log_factors: tuple[float, ...]
    conditional_log_factors: tuple[float, ...]
    cumulative_log_marginals: tuple[float, ...]
    cumulative_log_joints: tuple[float, ...]
    cumulative_errors: tuple[float, ...]
    accepted_prefix_length: int
    conditional_probes: int


@dataclass(frozen=True)
class AdaptiveGenerationResult:
    state: Any
    continuation_token_ids: torch.Tensor
    scheduler_steps: int
    conditional_probes: int
    accepted_prefix_histogram: tuple[int, ...]


def _canvas_slice(state: Any) -> slice:
    return slice(
        state.canvas_start,
        state.canvas_start + state.canvas_length,
    )


def _filtered_top1(
    adapter: Any,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return filtered winner IDs, probabilities, and log probabilities."""

    if logits.ndim != 2:
        raise ValueError("Expected logits [events, vocabulary].")
    scoring_logits = logits.float()
    excluded = (
        adapter.collect_excluded_token_ids(scoring_logits.shape[-1])
        if adapter.filter_special_tokens
        else set()
    )
    if excluded:
        if len(excluded) >= scoring_logits.shape[-1]:
            raise ValueError("Special-token filtering removed the vocabulary.")
        excluded_tensor = torch.tensor(
            sorted(excluded),
            device=scoring_logits.device,
            dtype=torch.long,
        )
        scoring_logits.index_fill_(-1, excluded_tensor, -torch.inf)
    best_logits, best_ids = scoring_logits.max(dim=-1)
    best_log_probs = best_logits - torch.logsumexp(scoring_logits, dim=-1)
    return best_ids, torch.exp(best_log_probs), best_log_probs


def _filtered_fixed_token_log_probability(
    adapter: Any,
    logits: torch.Tensor,
    token_id: torch.Tensor,
) -> torch.Tensor:
    """Score one frozen token without materializing all log probabilities."""

    if logits.ndim != 1:
        raise ValueError("Expected one vocabulary-logit vector.")
    scoring_logits = logits.float()
    excluded = (
        adapter.collect_excluded_token_ids(scoring_logits.shape[-1])
        if adapter.filter_special_tokens
        else set()
    )
    if excluded:
        excluded_tensor = torch.tensor(
            sorted(excluded),
            device=scoring_logits.device,
            dtype=torch.long,
        )
        scoring_logits.index_fill_(0, excluded_tensor, -torch.inf)
    target_logit = scoring_logits.gather(0, token_id.reshape(1)).squeeze(0)
    return target_logit - torch.logsumexp(scoring_logits, dim=0)


def _atomic_commit_prefix(
    state: Any,
    sequence_positions: torch.Tensor,
    token_ids: torch.Tensor,
) -> Any:
    """Commit a frozen prefix with one scatter and no intermediate model pass."""

    new_input_ids = state.input_ids.clone()
    new_input_ids.scatter_(
        1,
        sequence_positions.reshape(1, -1),
        token_ids.reshape(1, -1),
    )
    return replace(state, input_ids=new_input_ids)


def adaptive_step(
    adapter: Any,
    state: Any,
    *,
    config: AdaptiveSchedulerConfig,
    remaining_count: int,
) -> AdaptiveStepResult:
    """Run one base proposal, short-circuit probes, and atomic commitment."""

    if state.input_ids.shape[0] != 1:
        raise ValueError("Adaptive scheduler version one requires batch size 1.")
    if remaining_count < 1 or remaining_count > state.canvas_length:
        raise ValueError("remaining_count is outside the canvas range.")

    candidate_count = min(config.max_candidates, remaining_count)
    canvas_slice = _canvas_slice(state)
    remaining_canvas = adapter.remaining_canvas_mask(state)[:, canvas_slice]
    masked_canvas_positions = torch.argsort(
        remaining_canvas.to(dtype=torch.int8),
        dim=-1,
        descending=True,
        stable=True,
    )[0, :remaining_count]

    base_logits = adapter.forward_aligned_logits(state.input_ids, None)
    canvas_logits = base_logits[0, canvas_slice, :]
    masked_logits = canvas_logits.index_select(0, masked_canvas_positions)
    best_ids, confidences, marginal_logps = _filtered_top1(
        adapter,
        masked_logits,
    )
    confidence_order = torch.argsort(
        confidences,
        descending=True,
        stable=True,
    )[:candidate_count]
    candidate_canvas_positions = masked_canvas_positions.index_select(
        0,
        confidence_order,
    )
    candidate_sequence_positions = (
        candidate_canvas_positions + state.canvas_start
    )
    candidate_token_ids = best_ids.index_select(0, confidence_order)
    candidate_confidences = confidences.index_select(0, confidence_order)
    candidate_marginal_logps = marginal_logps.index_select(
        0,
        confidence_order,
    )
    marginal_values = tuple(
        float(value)
        for value in candidate_marginal_logps.detach().cpu().tolist()
    )
    del base_logits, canvas_logits, masked_logits

    marginal_cumulative: list[float] = []
    running_marginal = 0.0
    for value in marginal_values:
        running_marginal = math.fsum((running_marginal, value))
        marginal_cumulative.append(running_marginal)

    conditional_values: list[float] = []
    joint_cumulative: list[float] = [marginal_values[0]]
    errors: list[float] = [0.0]
    accepted = 1
    probe_input_ids = state.input_ids.clone()

    for target_index in range(1, candidate_count):
        previous_index = target_index - 1
        probe_input_ids[
            0,
            candidate_sequence_positions[previous_index],
        ] = candidate_token_ids[previous_index]
        conditional_logits = adapter.forward_aligned_logits(
            probe_input_ids,
            None,
        )
        conditional_logp = _filtered_fixed_token_log_probability(
            adapter,
            conditional_logits[
                0,
                candidate_sequence_positions[target_index],
            ],
            candidate_token_ids[target_index],
        )
        conditional_value = float(conditional_logp.item())
        conditional_values.append(conditional_value)
        next_joint = math.fsum(
            (joint_cumulative[-1], conditional_value)
        )
        joint_cumulative.append(next_joint)
        error = abs(next_joint - marginal_cumulative[target_index])
        errors.append(error)
        del conditional_logits

        if error <= config.tau:
            accepted = target_index + 1
        else:
            break

    next_state = _atomic_commit_prefix(
        state,
        candidate_sequence_positions[:accepted],
        candidate_token_ids[:accepted],
    )
    return AdaptiveStepResult(
        state=next_state,
        remaining_masks_before=remaining_count,
        remaining_masks_after=remaining_count - accepted,
        candidate_canvas_positions=candidate_canvas_positions,
        candidate_sequence_positions=candidate_sequence_positions,
        candidate_token_ids=candidate_token_ids,
        candidate_confidences=candidate_confidences,
        marginal_log_factors=marginal_values,
        conditional_log_factors=tuple(conditional_values),
        cumulative_log_marginals=tuple(
            marginal_cumulative[: len(errors)]
        ),
        cumulative_log_joints=tuple(joint_cumulative),
        cumulative_errors=tuple(errors),
        accepted_prefix_length=accepted,
        conditional_probes=len(conditional_values),
    )


def generate_adaptive(
    adapter: Any,
    initial_state: Any,
    *,
    config: AdaptiveSchedulerConfig,
) -> AdaptiveGenerationResult:
    """Fill one fresh canvas using the adaptive short-circuit scheduler."""

    if initial_state.input_ids.shape[0] != 1:
        raise ValueError("Adaptive generation requires batch size 1.")

    state = initial_state
    remaining = initial_state.canvas_length
    steps = 0
    conditional_probes = 0
    histogram = [0] * config.max_candidates
    while remaining:
        step = adaptive_step(
            adapter,
            state,
            config=config,
            remaining_count=remaining,
        )
        state = step.state
        remaining = step.remaining_masks_after
        steps += 1
        conditional_probes += step.conditional_probes
        histogram[step.accepted_prefix_length - 1] += 1

    continuation = state.input_ids[:, _canvas_slice(state)]
    return AdaptiveGenerationResult(
        state=state,
        continuation_token_ids=continuation,
        scheduler_steps=steps,
        conditional_probes=conditional_probes,
        accepted_prefix_histogram=tuple(histogram),
    )
