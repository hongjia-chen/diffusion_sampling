"""Canonical greedy rollout for exhaustive Dream factorization probes.

At each denoising state ``c_t`` this module:

1. probes every requested subset of the frozen top-``K`` proposals;
2. permanently commits only the highest-confidence proposal ``x_1``; and
3. verifies that exactly one masked canvas position changed.

The temporary states created inside :func:`factorization.probe.probe_state`
never become part of the canonical trajectory.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from factorization.dream_adapter import (
    DreamAdapter,
    DreamState,
    TokenPrediction,
)
from factorization.probe import ProbeResult, probe_state


# -----------------------------------------------------------------------------
# Experiment configuration
#
# This dataclass defines exactly what every rollout measures. It validates and
# normalizes the requested experiment before any model computation begins.
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RolloutConfig:
    """Configuration for one deterministic one-token-per-step trajectory.

    ``max_k`` is the size of the frozen confidence-ranked candidate pool.
    ``subset_sizes`` controls which subset cardinalities are materialized by
    each probe. For the informative exhaustive ``K=6`` experiment, use
    ``max_k=6`` and ``subset_sizes=(2, 3, 4, 5, 6)``.
    """

    # Required experiment definition. Keeping these fields without defaults
    # prevents a rollout from silently selecting a different subset experiment.
    max_k: int
    subset_sizes: tuple[int, ...]

    num_steps: int = 64
    conditional_microbatch_size: int | None = 1
    rollout_id: str = "greedy-000"
    seed: int = 0

    # Normalize user/CLI values, reject inconsistent experiment settings, and
    # write the cleaned values back into this frozen dataclass.
    def __post_init__(self) -> None:
        num_steps = int(self.num_steps)
        max_k = int(self.max_k)
        subset_sizes = tuple(int(size) for size in self.subset_sizes)

        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")
        if not subset_sizes:
            raise ValueError("subset_sizes must contain at least one value.")
        if any(size < 1 for size in subset_sizes):
            raise ValueError("Every subset size must be at least 1.")
        if len(set(subset_sizes)) != len(subset_sizes):
            raise ValueError("subset_sizes must not contain duplicates.")

        subset_sizes = tuple(sorted(subset_sizes))

        if max_k < 1:
            raise ValueError("max_k must be at least 1.")
        if subset_sizes[-1] > max_k:
            raise ValueError(
                f"Largest subset size {subset_sizes[-1]} exceeds max_k={max_k}."
            )

        microbatch_size = self.conditional_microbatch_size
        if microbatch_size is not None:
            microbatch_size = int(microbatch_size)
            if microbatch_size < 1:
                raise ValueError(
                    "conditional_microbatch_size must be at least 1 or None."
                )

        rollout_id = str(self.rollout_id)
        if not rollout_id.strip():
            raise ValueError("rollout_id must not be empty.")

        object.__setattr__(self, "num_steps", num_steps)
        object.__setattr__(self, "max_k", max_k)
        object.__setattr__(self, "subset_sizes", subset_sizes)
        object.__setattr__(
            self,
            "conditional_microbatch_size",
            microbatch_size,
        )
        object.__setattr__(self, "rollout_id", rollout_id)
        object.__setattr__(self, "seed", int(self.seed))

    # Compute how many subset estimates one successful probe must emit. For
    # K=6 and subset sizes 2..6, the expected count is 57.
    @property
    def estimates_per_step(self) -> int:
        """Number of subset estimates emitted by every successful probe."""

        return sum(math.comb(self.max_k, size) for size in self.subset_sizes)


# -----------------------------------------------------------------------------
# Per-step output record
#
# One instance stores the measurements at c_t and the single permanent token
# commitment that advances the canonical trajectory to c_{t+1}.
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RolloutStep:
    """One measured state followed by one permanent greedy commitment."""

    step_index: int
    committed_count_before: int
    committed_count_after: int
    remaining_mask_count_before: int
    remaining_mask_count_after: int
    progress_before: float
    progress_after: float
    probe_elapsed_seconds: float
    commit_elapsed_seconds: float
    total_elapsed_seconds: float
    committed: TokenPrediction
    probe: ProbeResult


# -----------------------------------------------------------------------------
# Whole-rollout output record
#
# This bundles the start, finish, timings, and every validated RolloutStep for
# the outer runner to serialize without repeating model work.
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RolloutResult:
    """Completed canonical trajectory and its per-step measurements."""

    config: RolloutConfig
    initial_state: DreamState
    final_state: DreamState
    steps: tuple[RolloutStep, ...]
    initial_mask_count: int
    final_mask_count: int
    total_elapsed_seconds: float

    # Report how many canonical one-token commitments completed successfully.
    @property
    def completed_steps(self) -> int:
        return len(self.steps)

    # Extract committed sequence positions in rollout order.
    @property
    def committed_sequence_positions(self) -> tuple[int, ...]:
        return tuple(step.committed.sequence_position for step in self.steps)

    # Extract committed token IDs in rollout order.
    @property
    def committed_token_ids(self) -> tuple[int, ...]:
        return tuple(step.committed.token_id for step in self.steps)


# Pull the fixed generation canvas out of the full model input and convert it
# to ordinary Python integers for JSON output, auditing, and decoding.
def canvas_token_ids(state: DreamState) -> tuple[int, ...]:
    """Return the exact token IDs occupying the fixed generation canvas."""

    start = int(state.canvas_start)
    end = start + int(state.canvas_length)
    values = state.input_ids[0, start:end].detach().cpu().tolist()
    if len(values) != state.canvas_length:
        raise RuntimeError(
            f"Expected {state.canvas_length} canvas tokens, received {len(values)}."
        )
    return tuple(int(value) for value in values)


# Decode only the generated canvas, excluding the prompt/prefix around it.
# This is used for human-readable progress and final-result reporting.
def decode_canvas(
    adapter: DreamAdapter,
    state: DreamState,
    *,
    skip_special_tokens: bool,
) -> str:
    """Decode only the generation canvas while preserving token spacing."""

    decoded = adapter.tokenizer.decode(
        list(canvas_token_ids(state)),
        skip_special_tokens=bool(skip_special_tokens),
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(decoded, str):
        raise TypeError("Tokenizer decode did not return a string.")
    return decoded


# Count remaining MASK tokens on the generation canvas. Keeping this operation
# in one helper makes all setup, per-step, and final checks consistent.
def _count_remaining(adapter: DreamAdapter, state: DreamState) -> int:
    return int(adapter.remaining_positions(state).numel())


# Audit one permanent c_t -> c_{t+1} transition. Exactly the selected MASK must
# change to the selected token; tensor shapes, masks, and metadata must not.
# The validated remaining-mask count is returned to the rollout loop.
def _validate_commit_transition(
    adapter: DreamAdapter,
    before: DreamState,
    after: DreamState,
    committed: TokenPrediction,
    *,
    remaining_before: int,
) -> int:
    """Verify that the canonical transition changed exactly the chosen event."""

    if before.input_ids.shape != after.input_ids.shape:
        raise RuntimeError("A commitment changed the input tensor shape.")
    if before.attention_mask.shape != after.attention_mask.shape:
        raise RuntimeError("A commitment changed the attention-mask shape.")
    if before.canvas_mask.shape != after.canvas_mask.shape:
        raise RuntimeError("A commitment changed the canvas-mask shape.")

    differences = torch.nonzero(
        before.input_ids != after.input_ids,
        as_tuple=False,
    )
    if differences.shape != (1, 2):
        raise RuntimeError(
            "A canonical commitment must change exactly one tensor element; "
            f"found {int(differences.shape[0])}."
        )

    changed_batch = int(differences[0, 0].item())
    changed_position = int(differences[0, 1].item())
    if changed_batch != 0:
        raise RuntimeError("The changed token was not in the single sequence row.")
    if changed_position != committed.sequence_position:
        raise RuntimeError(
            "The committed tensor position does not match probe.selected[0]: "
            f"changed={changed_position}, selected={committed.sequence_position}."
        )

    old_token = int(before.input_ids[0, changed_position].item())
    new_token = int(after.input_ids[0, changed_position].item())
    if old_token != int(adapter.mask_token_id):
        raise RuntimeError("The canonical commitment did not replace a MASK token.")
    if new_token != committed.token_id:
        raise RuntimeError(
            "The canonical commitment wrote the wrong token ID: "
            f"wrote={new_token}, selected={committed.token_id}."
        )

    if not torch.equal(before.attention_mask, after.attention_mask):
        raise RuntimeError("A token commitment mutated the attention mask.")
    if not torch.equal(before.canvas_mask, after.canvas_mask):
        raise RuntimeError("A token commitment mutated the canvas mask.")

    metadata_fields = (
        "prefix_length",
        "canvas_start",
        "canvas_length",
        "mode",
        "visible_text",
    )
    for field in metadata_fields:
        if getattr(before, field) != getattr(after, field):
            raise RuntimeError(f"A token commitment changed state metadata: {field}.")

    remaining_after = _count_remaining(adapter, after)
    if remaining_after != remaining_before - 1:
        raise RuntimeError(
            "Exactly one mask must disappear per canonical step: "
            f"before={remaining_before}, after={remaining_after}."
        )

    return remaining_after


# Prove that run_rollout did not mutate the DreamState supplied by its caller.
# The trajectory advances through newly returned states, not in-place edits.
def _assert_initial_state_unchanged(
    initial_state: DreamState,
    *,
    input_ids_before: torch.Tensor,
    attention_mask_before: torch.Tensor,
    canvas_mask_before: torch.Tensor,
) -> None:
    if not torch.equal(initial_state.input_ids, input_ids_before):
        raise RuntimeError("run_rollout mutated the caller's initial input_ids.")
    if not torch.equal(initial_state.attention_mask, attention_mask_before):
        raise RuntimeError("run_rollout mutated the caller's attention_mask.")
    if not torch.equal(initial_state.canvas_mask, canvas_mask_before):
        raise RuntimeError("run_rollout mutated the caller's canvas_mask.")


# Run the canonical experiment loop. At each c_t: measure all configured
# subsets, validate the probe, permanently commit only x_1, validate the state
# transition, and record the complete step. Probe-only states are discarded.
def run_rollout(
    adapter: DreamAdapter,
    initial_state: DreamState,
    *,
    config: RolloutConfig,
    on_step: Callable[[RolloutStep], None] | None = None,
) -> RolloutResult:
    """Run a deterministic trajectory while exhaustively probing each state.

    Every loop iteration probes the current canonical state, then commits only
    ``probe.selected[0]``. The remaining selected events and all temporary
    conditional states are discarded.
    """

    # Reject an invalid configuration object before touching model state.
    if not isinstance(config, RolloutConfig):
        raise TypeError("config must be a RolloutConfig instance.")


    # Snapshot caller-owned tensors so the final invariant check can verify that
    # the original DreamState was not mutated in place.
    initial_input_ids = initial_state.input_ids.clone()
    initial_attention_mask = initial_state.attention_mask.clone()
    initial_canvas_mask = initial_state.canvas_mask.clone()

    # Verify that the starting state can supply K candidates at every requested
    # step and can therefore support the requested trajectory length.
    initial_mask_count = _count_remaining(adapter, initial_state)
    if initial_mask_count < config.max_k:
        raise ValueError(
            f"Cannot begin a max_k={config.max_k} rollout with only "
            f"{initial_mask_count} masked canvas positions."
        )

    # A probe runs before each commitment. Therefore the final permitted step
    # may start with exactly max_k masks and leave max_k - 1 afterward.
    maximum_steps = initial_mask_count - config.max_k + 1
    if config.num_steps > maximum_steps:
        raise ValueError(
            f"num_steps={config.num_steps} is too large for "
            f"initial_masks={initial_mask_count} and max_k={config.max_k}; "
            f"the maximum is {maximum_steps}."
        )

    # current_state is the sole canonical trajectory state. Temporary states
    # created by probe_state never replace or advance this variable.
    current_state = initial_state
    steps: list[RolloutStep] = []
    rollout_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Canonical loop: probe c_t, commit x_1, record c_t -> c_{t+1}.
    # ------------------------------------------------------------------
    for step_index in range(config.num_steps):
        step_start = time.perf_counter()
        remaining_before = _count_remaining(adapter, current_state)
        committed_before = initial_mask_count - remaining_before

        # Measure every requested subset of the frozen top-K proposals at the
        # current state. probe_state must leave current_state unchanged.
        probe_start = time.perf_counter()
        probe = probe_state(
            adapter,
            current_state,
            max_k=config.max_k,
            subset_sizes=config.subset_sizes,
            conditional_microbatch_size=config.conditional_microbatch_size,
        )
        probe_elapsed = time.perf_counter() - probe_start

        # Validate the probe contract before trusting or serializing its output.
        # These checks catch stale APIs, wrong K values, and partial estimates.
        if probe.remaining_mask_count != remaining_before:
            raise RuntimeError(
                "Probe mask count disagrees with the canonical state: "
                f"probe={probe.remaining_mask_count}, state={remaining_before}."
            )
        if probe.max_k != config.max_k:
            raise RuntimeError(
                f"Probe returned max_k={probe.max_k}; expected {config.max_k}."
            )
        if probe.subset_sizes != config.subset_sizes:
            raise RuntimeError(
                "Probe returned unexpected subset sizes: "
                f"{probe.subset_sizes} != {config.subset_sizes}."
            )
        if len(probe.selected) != config.max_k:
            raise RuntimeError(
                f"Probe selected {len(probe.selected)} events; "
                f"expected {config.max_k}."
            )
        if len(probe.estimates) != config.estimates_per_step:
            raise RuntimeError(
                f"Probe emitted {len(probe.estimates)} estimates; "
                f"expected {config.estimates_per_step}."
            )

        # Advance the real trajectory with only the highest-confidence frozen
        # event x_1. All other selected events were measurement-only.
        committed = probe.selected[0]
        commit_start = time.perf_counter()
        next_state = adapter.commit_token(
            current_state,
            sequence_position=committed.sequence_position,
            token_id=committed.token_id,
        )
        remaining_after = _validate_commit_transition(
            adapter,
            current_state,
            next_state,
            committed,
            remaining_before=remaining_before,
        )
        commit_elapsed = time.perf_counter() - commit_start

        # Bundle counters, timings, the committed event, and the full probe into
        # one immutable record for the persistence layer.
        committed_after = initial_mask_count - remaining_after
        step = RolloutStep(
            step_index=step_index,
            committed_count_before=committed_before,
            committed_count_after=committed_after,
            remaining_mask_count_before=remaining_before,
            remaining_mask_count_after=remaining_after,
            progress_before=committed_before / initial_mask_count,
            progress_after=committed_after / initial_mask_count,
            probe_elapsed_seconds=probe_elapsed,
            commit_elapsed_seconds=commit_elapsed,
            total_elapsed_seconds=time.perf_counter() - step_start,
            committed=committed,
            probe=probe,
        )
        steps.append(step)
        current_state = next_state

        # Allow the outer runner to stream JSONL/progress after each completely
        # validated step without coupling persistence to rollout logic.
        if on_step is not None:
            on_step(step)

    # Check whole-run invariants: one mask disappeared per completed step and
    # the caller's original state stayed unchanged.
    total_elapsed = time.perf_counter() - rollout_start
    final_mask_count = _count_remaining(adapter, current_state)
    expected_final = initial_mask_count - config.num_steps
    if final_mask_count != expected_final:
        raise RuntimeError(
            f"Final mask count is {final_mask_count}; expected {expected_final}."
        )

    _assert_initial_state_unchanged(
        initial_state,
        input_ids_before=initial_input_ids,
        attention_mask_before=initial_attention_mask,
        canvas_mask_before=initial_canvas_mask,
    )

    # Return the full in-memory trajectory. The runner decides how to write
    # JSONL, summaries, decoded text, and metadata.
    return RolloutResult(
        config=config,
        initial_state=initial_state,
        final_state=current_state,
        steps=tuple(steps),
        initial_mask_count=initial_mask_count,
        final_mask_count=final_mask_count,
        total_elapsed_seconds=total_elapsed,
    )
