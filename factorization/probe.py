"""Probe exhaustive subsets of Dream's confidence-ranked proposals.

At one fixed denoising state c_t, freeze the top-K proposed events

    x_1, ..., x_K.

For every requested subset S = (i_1, ..., i_m), where
i_1 < ... < i_m, compute

    Q_S = product_r p(x_{i_r} | c_t)

and

    P_S = p(x_{i_1} | c_t)
          product_{r=2}^m
          p(x_{i_r} | x_{i_1}, ..., x_{i_{r-1}}, c_t).

The subset order is always the original confidence order.
Temporary conditional states do not advance the canonical rollout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import torch

from factorization.dream_adapter import (
    DreamAdapter,
    DreamState,
    TokenPrediction,
)


@dataclass(frozen=True)
class FactorizationEstimate:

    subset_ranks: tuple[int, ...]
    subset_mask: int
    subset_size: int

    # Individual factors retained for auditing.
    marginal_log_factors: tuple[float, ...]
    ordered_joint_log_factors: tuple[float, ...]

    # P_S: confidence-order conditional-chain product.
    ordered_joint_probability: float

    # Q_S: product of original-state marginals.
    marginal_product_probability: float

    # Raw signed gap: P_S - Q_S.
    difference_joint_minus_product: float

    # log(P_S).
    ordered_joint_log_probability: float

    # log(Q_S).
    marginal_product_log_probability: float

    # log(P_S) - log(Q_S) = log(P_S / Q_S).
    log_gap_joint_minus_product: float

    @property
    def k(self) -> int:
        return self.subset_size

    @property
    def absolute_difference(self) -> float:
        return abs(self.difference_joint_minus_product)

    @property
    def absolute_log_gap(self) -> float:
        return abs(self.log_gap_joint_minus_product)


@dataclass(frozen=True)
class ProbeResult:
    """All exhaustive-subset measurements from one frozen denoising state."""

    remaining_mask_count: int

    # Number of confidence-ranked events in the frozen candidate pool.
    max_k: int

    # Subset cardinalities materialized in ``estimates``.
    subset_sizes: tuple[int, ...]

    # Frozen x_1, ..., x_K selected by the original forward pass.
    selected: tuple[TokenPrediction, ...]

    # Entry j-1 is log p(x_j | c_t).
    marginal_log_factors: tuple[float, ...]

    # One result for every requested subset, ordered first by size and then
    # lexicographically by one-based confidence rank.
    estimates: tuple[FactorizationEstimate, ...]

    # Value-retriving functions 
    def estimates_for_size(self, subset_size: int) -> tuple[FactorizationEstimate, ...]:
        """Return every estimate whose subset has the requested cardinality."""

        matches = tuple(
            estimate
            for estimate in self.estimates
            if estimate.subset_size == subset_size
        )
        if not matches:
            raise KeyError(
                f"No estimates for subset_size={subset_size}; "
                f"available sizes are {list(self.subset_sizes)}."
            )
        return matches

    def estimate_for_subset(
        self,
        subset_ranks: Sequence[int],
    ) -> FactorizationEstimate:
        """Return the estimate for one exact tuple of one-based ranks."""

        key = tuple(int(rank) for rank in subset_ranks)
        for estimate in self.estimates:
            if estimate.subset_ranks == key:
                return estimate

        raise KeyError(f"No estimate for subset_ranks={key}.")

# Cleans and preps in put subset_sizes and max_k values
def _normalize_subset_sizes(
    subset_sizes: Sequence[int],
    *,
    max_k: int,
) -> tuple[tuple[int, ...], int]:
    """Validate subset sizes and resolve the frozen candidate-pool size."""

    normalized = tuple(int(size) for size in subset_sizes)

    if not normalized:
        raise ValueError("subset_sizes must contain at least one value.")
    if any(size < 1 for size in normalized):
        raise ValueError("Every subset size must be at least 1.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("subset_sizes must not contain duplicates.")

    normalized = tuple(sorted(normalized))
    resolved_max_k = int(max_k)

    if resolved_max_k < 1:
        raise ValueError("max_k must be at least 1.")
    if normalized[-1] > resolved_max_k:
        raise ValueError(
            f"Largest subset size {normalized[-1]} exceeds max_k={resolved_max_k}."
        )

    return normalized, resolved_max_k

## Generates every candidates subset for a given max_k; ordered by confidence and indexed at 0
def _enumerate_subsets(
    max_k: int,
    subset_sizes: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Enumerate zero-based rank subsets by size, then lexicographically."""

    return tuple(
        subset
        for size in subset_sizes
        for subset in combinations(range(max_k), size)
    )

def _subset_mask(subset_indices: Sequence[int]) -> int:
    """Encode a zero-based rank subset as an integer bit mask."""

    mask = 0
    for index in subset_indices:
        mask |= 1 << int(index)
    return mask

def _difference_from_logs(log_p: float, log_q: float) -> float:
    """Compute ``exp(log_p) - exp(log_q)`` with improved local precision."""

    if log_p >= log_q:
        # exp(log_p) * (1 - exp(log_q - log_p))
        return math.exp(log_p) * (-math.expm1(log_q - log_p))

    # exp(log_q) * (exp(log_p - log_q) - 1)
    return math.exp(log_q) * math.expm1(log_p - log_q)

def _build_factor_requests(
    subsets: Sequence[tuple[int, ...]],
) -> dict[tuple[int, ...], tuple[int, ...]]:
    """Map each conditioning subset to all target ranks scored from its state.

    For subset ``(0, 2, 4)``, the required non-marginal factors are
    ``x_3 | x_1`` and ``x_5 | x_1, x_3``. These correspond to requests
    ``(0,) -> 2`` and ``(0, 2) -> 4``.
    """

    mutable: dict[tuple[int, ...], set[int]] = {}

    for subset in subsets:
        for target_offset in range(1, len(subset)):
            conditioning = subset[:target_offset]
            target_index = subset[target_offset]
            mutable.setdefault(conditioning, set()).add(target_index)

    return {
        conditioning: tuple(sorted(targets))
        for conditioning, targets in sorted(
            mutable.items(),
            key=lambda item: (len(item[0]), item[0]),
        )
    }

## Appends tokens to our current context to create the correct conditioning sequence states
def _build_conditional_states(
    adapter: DreamAdapter,
    state: DreamState,
    selected: Sequence[TokenPrediction],
    conditioning_subsets: Sequence[tuple[int, ...]],
) -> dict[tuple[int, ...], DreamState]:
    """Build one temporary state for every unique conditioning subset."""

    cache: dict[tuple[int, ...], DreamState] = {(): state}

    def build(conditioning: tuple[int, ...]) -> DreamState:
        cached = cache.get(conditioning)
        if cached is not None:
            return cached

        parent = conditioning[:-1]
        last_index = conditioning[-1]
        parent_state = build(parent)
        prediction = selected[last_index]
        conditional_state = adapter.commit_token(
            parent_state,
            sequence_position=prediction.sequence_position,
            token_id=prediction.token_id,
        )
        cache[conditioning] = conditional_state
        return conditional_state

    return {
        conditioning: build(conditioning)
        for conditioning in conditioning_subsets
    }



def _score_conditional_factors(
    adapter: DreamAdapter,
    conditional_states: Mapping[tuple[int, ...], DreamState],
    targets_by_conditioning: Mapping[tuple[int, ...], tuple[int, ...]],
    selected: Sequence[TokenPrediction],
    *,
    microbatch_size: int | None,
) -> dict[tuple[tuple[int, ...], int], float]:
    """Score all unique ``target | conditioning-subset`` factors.

    A single conditional-state forward pass may score several later frozen
    targets. Explicit ``batch_indices`` align each event with the correct row
    when several states are evaluated in one microbatch.
    """

    conditioning_keys = tuple(targets_by_conditioning)
    if not conditioning_keys:
        return {}

    if set(conditioning_keys) != set(conditional_states):
        raise ValueError(
            "conditional_states and targets_by_conditioning must have "
            "identical conditioning keys."
        )

    if microbatch_size is None:
        microbatch_size = len(conditioning_keys)
    if microbatch_size < 1:
        raise ValueError("microbatch_size must be at least 1 or None.")

    scores: dict[tuple[tuple[int, ...], int], float] = {}

    for start in range(0, len(conditioning_keys), microbatch_size):
        chunk_keys = conditioning_keys[start : start + microbatch_size]
        state_chunk = [conditional_states[key] for key in chunk_keys]

        input_ids = torch.cat(
            [item.input_ids for item in state_chunk],
            dim=0,
        )
        attention_mask = torch.cat(
            [item.attention_mask for item in state_chunk],
            dim=0,
        )

        aligned_logits = adapter.forward_aligned_logits(
            input_ids,
            attention_mask,
        )

        event_keys: list[tuple[tuple[int, ...], int]] = []
        batch_indices: list[int] = []
        sequence_positions: list[int] = []
        token_ids: list[int] = []

        for batch_index, conditioning in enumerate(chunk_keys):
            for target_index in targets_by_conditioning[conditioning]:
                prediction = selected[target_index]
                event_keys.append((conditioning, target_index))
                batch_indices.append(batch_index)
                sequence_positions.append(prediction.sequence_position)
                token_ids.append(prediction.token_id)

        chunk_logps = adapter.event_log_probabilities(
            aligned_logits,
            sequence_positions=sequence_positions,
            token_ids=token_ids,
            batch_indices=batch_indices,
        )

        chunk_values = chunk_logps.detach().cpu().tolist()
        if len(chunk_values) != len(event_keys):
            raise RuntimeError("Conditional scorer returned the wrong event count.")

        for event_key, value in zip(event_keys, chunk_values):
            scores[event_key] = float(value)

        del aligned_logits

    expected_count = sum(len(targets) for targets in targets_by_conditioning.values())
    if len(scores) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} unique conditional factors, "
            f"received {len(scores)}."
        )

    return scores


def probe_state(
    adapter: DreamAdapter,
    state: DreamState,
    *,
    subset_sizes: Sequence[int],
    max_k: int,
    conditional_microbatch_size: int | None = 1,
) -> ProbeResult:
    """Measure exhaustive subset factorizations at one denoising state.

    The original forward pass freezes ``x_1, ..., x_K`` in descending original
    confidence order. For every subset whose cardinality appears in
    ``subset_sizes``, factors are evaluated in that same frozen order.

    ``max_k`` controls the candidate-pool size. When omitted, it defaults to
    ``max(subset_sizes)``. Singleton subsets are supported for invariant tests;
    the empty subset is intentionally not materialized.

    When ``subset_sizes`` is omitted, it defaults to every size from 2 up to
    ``max_k`` (or up to 8 if ``max_k`` is also omitted).
    """

    requested_sizes, resolved_max_k = _normalize_subset_sizes(
        subset_sizes,
        max_k=max_k,
    )

    remaining_count = int(adapter.remaining_positions(state).numel())
    if remaining_count < resolved_max_k:
        raise ValueError(
            f"Cannot probe max_k={resolved_max_k}: only {remaining_count} "
            "masked canvas positions remain."
        )

    canonical_input_before = state.input_ids.clone()

    # ------------------------------------------------------------------
    # 1. One forward pass: select positions and freeze tokens  x_1, ..., x_K. 
    # ------------------------------------------------------------------
    base_logits = adapter.forward_aligned_logits(
        state.input_ids,
        state.attention_mask,
    )

    selected = tuple(
        adapter.rank_remaining_positions(
            state,
            base_logits,
            top_n=resolved_max_k,
        )
    )

    if len(selected) != resolved_max_k:
        raise RuntimeError(
            f"Expected {resolved_max_k} selected events, received {len(selected)}."
        )

    selected_positions = [item.sequence_position for item in selected]
    selected_token_ids = [item.token_id for item in selected]

    if len(set(selected_positions)) != resolved_max_k:
        raise RuntimeError("The selected tuple contains duplicate positions.")

    # ------------------------------------------------------------------
    # 2. Score every frozen token under the original state c_t. These are the marginals
    # ------------------------------------------------------------------
    marginal_logps = adapter.event_log_probabilities(
        base_logits,
        sequence_positions=selected_positions,
        token_ids=selected_token_ids,
    )
    marginal_values = tuple(
        float(value)
        for value in marginal_logps.detach().cpu().tolist()
    )
    del base_logits

    if len(marginal_values) != resolved_max_k:
        raise RuntimeError("The original pass returned the wrong factor count.")

    # ------------------------------------------------------------------
    # 3. Enumerate subsets and score conditional factors
    # ------------------------------------------------------------------
    subsets = _enumerate_subsets(resolved_max_k, requested_sizes)
    targets_by_conditioning = _build_factor_requests(subsets)
    conditional_states = _build_conditional_states(
        adapter,
        state,
        selected,
        tuple(targets_by_conditioning),
    )
    conditional_scores = _score_conditional_factors(
        adapter,
        conditional_states,
        targets_by_conditioning,
        selected,
        microbatch_size=conditional_microbatch_size,
    )

    if not torch.equal(state.input_ids, canonical_input_before):
        raise RuntimeError("The exhaustive subset probe mutated the canonical state.")

    # ------------------------------------------------------------------
    # 4. Calculate one Estimate (bundles of metrics) per subset.
    # ------------------------------------------------------------------
    estimates: list[FactorizationEstimate] = []

    for subset in subsets:
        marginal_factors = tuple(marginal_values[index] for index in subset)

        ordered_factors = [marginal_values[subset[0]]]
        for target_offset in range(1, len(subset)):
            conditioning = subset[:target_offset]
            target_index = subset[target_offset]
            ordered_factors.append(
                conditional_scores[(conditioning, target_index)]
            )
        ordered_factor_tuple = tuple(ordered_factors)

        log_q = math.fsum(marginal_factors)
        log_p = math.fsum(ordered_factor_tuple)
        subset_ranks = tuple(index + 1 for index in subset)

        estimates.append(
            FactorizationEstimate(
                subset_ranks=subset_ranks,
                subset_mask=_subset_mask(subset),
                subset_size=len(subset),
                marginal_log_factors=marginal_factors,
                ordered_joint_log_factors=ordered_factor_tuple,
                ordered_joint_probability=math.exp(log_p),
                marginal_product_probability=math.exp(log_q),
                difference_joint_minus_product=_difference_from_logs(
                    log_p,
                    log_q,
                ),
                ordered_joint_log_probability=log_p,
                marginal_product_log_probability=log_q,
                log_gap_joint_minus_product=log_p - log_q,
            )
        )

    expected_estimate_count = sum(
        math.comb(resolved_max_k, size)
        for size in requested_sizes
    )
    if len(estimates) != expected_estimate_count:
        raise RuntimeError(
            f"Expected {expected_estimate_count} subset estimates, "
            f"received {len(estimates)}."
        )

    return ProbeResult(
        remaining_mask_count=remaining_count,
        max_k=resolved_max_k,
        subset_sizes=requested_sizes,
        selected=selected,
        marginal_log_factors=marginal_values,
        estimates=tuple(estimates),
    )