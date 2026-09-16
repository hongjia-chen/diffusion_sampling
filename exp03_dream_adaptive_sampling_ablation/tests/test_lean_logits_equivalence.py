"""The memory-lean read path must reproduce the full-logits path exactly.

These tests guard the optimization that keeps 7B bfloat16 weights and the
sampler's activations inside a 16 GB card: the scheduler now projects only the
positions it reads through the vocabulary head instead of materializing
[batch, sequence, vocabulary]. That is a pure memory change and must not move a
single committed token.
"""

from __future__ import annotations

import torch

from dream_ablation.scheduler import (
    GenerationConfig,
    _propose,
    _remaining_positions,
    generate,
    generation_step,
)


HIDDEN = 16
VOCAB = 32


class FullLogitsProvider:
    """Reference provider: exposes only the original [B, S, V] read path."""

    mask_token_id = 9
    eos_token_id = 8
    excluded_token_ids = {9, 5}
    device = torch.device("cpu")
    supports_lean_logits = False

    def __init__(self, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.embedding = torch.randn(64, HIDDEN, generator=generator)
        self.weight = torch.randn(VOCAB, HIDDEN, generator=generator)
        self.positional = torch.randn(256, HIDDEN, generator=generator)

    def _hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, length = input_ids.shape
        states = self.embedding[input_ids] + self.positional[:length].unsqueeze(0)
        # A cheap stand-in for bidirectional attention: every position mixes in
        # the whole-sequence mean, so edits to one row change every position.
        return torch.tanh(states + states.mean(dim=1, keepdim=True))

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self._hidden(input_ids) @ self.weight.T
        return torch.cat((logits[:, :1, :], logits[:, :-1, :]), dim=1)


class LeanLogitsProvider(FullLogitsProvider):
    """Same model, exposing the position-sliced read path the scheduler prefers."""

    supports_lean_logits = True

    def hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._hidden(input_ids)

    def logits_from_hidden(
        self,
        hidden: torch.Tensor,
        rows: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        source = (positions - 1).clamp(min=0)
        return hidden[rows, source, :] @ self.weight.T


def _configs() -> list[GenerationConfig]:
    return [
        GenerationConfig(
            method="adaptive",
            continuation_length=12,
            delta=delta,
            max_candidates=6,
            verifier_microbatch_size=microbatch,
        )
        for delta in (0.01, 0.20, 0.75)
        for microbatch in (1, 4, 6)
    ] + [
        GenerationConfig(
            method="fixed",
            continuation_length=12,
            fixed_k=k,
            max_candidates=6,
            verifier_microbatch_size=4,
        )
        for k in (1, 2, 4, 6)
    ]


def _prompt() -> torch.Tensor:
    return torch.tensor([[0, 1, 2, 3]], dtype=torch.long)


def _assert_traces_match(reference: list[dict], sliced: list[dict], label: str) -> None:
    """Decisions must match exactly; log-factors only to float tolerance.

    The vocabulary projection runs at a different GEMM shape on the lean path,
    so BLAS may accumulate in a different order. That moves log-factors by about
    one ULP and must never move a committed token.
    """
    exact_keys = (
        "step_index",
        "remaining_masks_before",
        "candidate_positions",
        "candidate_token_ids",
        "accepted_prefix_length",
        "committed_positions",
        "remaining_masks_after",
        "eos_position_after",
        "threshold_delta",
        "threshold_tau_nats",
    )
    float_keys = (
        "candidate_confidences",
        "marginal_log_factors",
        "conditional_log_factors",
        "error_by_prefix_nats",
    )
    assert len(reference) == len(sliced), label
    for step, (want, got) in enumerate(zip(reference, sliced)):
        assert want.keys() == got.keys(), f"{label} step {step}"
        for key in exact_keys:
            assert want[key] == got[key], f"{label} step {step} key {key}"
        for key in float_keys:
            assert len(want[key]) == len(got[key]), f"{label} step {step} key {key}"
            for a, b in zip(want[key], got[key]):
                assert abs(a - b) <= 1e-5, f"{label} step {step} key {key}: {a} vs {b}"


def _assert_close(reference: torch.Tensor, sliced: torch.Tensor, label: str) -> None:
    """Agreement to float tolerance, plus exact agreement of the argmax.

    Bitwise equality is deliberately not asserted. The lean path multiplies
    [N, H] @ [H, V] where the reference multiplies [B, S, H] @ [H, V], and BLAS
    is free to accumulate those in different orders. Whether the last bit
    happens to match varies by BLAS build and thread count, so requiring it
    produces failures that track the machine rather than the code. An actual
    defect -- an off-by-one in the shift, say -- moves the result by order 1.
    """
    gap = (reference - sliced).abs().max().item()
    scale = reference.abs().max().item() or 1.0
    assert gap / scale <= 1e-5, f"{label}: relative gap {gap / scale}"
    assert torch.equal(reference.argmax(-1), sliced.argmax(-1)), f"{label}: argmax moved"


def test_lm_head_slicing_matches_full_logits() -> None:
    lean = LeanLogitsProvider()
    probe = torch.randint(0, 64, (3, 10))
    rows = torch.tensor([0, 2, 1, 0])
    positions = torch.tensor([0, 9, 4, 1])
    reference = lean.forward_logits(probe)[rows, positions, :]
    sliced = lean.logits_from_hidden(lean.hidden_states(probe), rows, positions)
    _assert_close(reference, sliced, "lm_head slicing")


def test_lm_head_slicing_catches_a_shift_error() -> None:
    """The tolerance above must still be tight enough to catch a real defect."""
    lean = LeanLogitsProvider()
    probe = torch.randint(0, 64, (3, 10))
    rows = torch.tensor([0, 2, 1, 0])
    positions = torch.tensor([0, 9, 4, 1])
    reference = lean.forward_logits(probe)[rows, positions, :]
    # Drop the shift-by-one that forward_logits applies.
    unshifted = lean.hidden_states(probe)[rows, positions, :] @ lean.weight.T
    try:
        _assert_close(reference, unshifted, "deliberately wrong")
    except AssertionError:
        return
    raise AssertionError("tolerance is too loose: an unshifted read path passed")


def test_proposer_is_identical_under_chunking() -> None:
    full, lean = FullLogitsProvider(), LeanLogitsProvider()
    config = GenerationConfig(
        method="adaptive", continuation_length=12, delta=0.1, max_candidates=6
    )
    state = torch.cat(
        (_prompt(), torch.full((1, 12), full.mask_token_id, dtype=torch.long)), dim=1
    )
    remaining = _remaining_positions(
        state, canvas_start=4, canvas_length=12, mask_token_id=full.mask_token_id
    )
    reference = _propose(full, state, remaining, config.max_candidates)
    sliced = _propose(lean, state, remaining, config.max_candidates)
    # Positions and token ids are decisions and must match exactly; the
    # confidences behind them are floats and only need to agree to tolerance.
    assert torch.equal(reference.sequence_positions, sliced.sequence_positions)
    assert torch.equal(reference.token_ids, sliced.token_ids)
    _assert_close(reference.confidences, sliced.confidences, "proposer confidences")


def test_generation_is_identical_across_read_paths() -> None:
    full, lean = FullLogitsProvider(), LeanLogitsProvider()
    for config in _configs():
        reference = generate(full, _prompt(), config, capture_traces=True)
        sliced = generate(lean, _prompt(), config, capture_traces=True)
        assert sliced.continuation_token_ids == reference.continuation_token_ids, config.label
        assert sliced.denoising_steps == reference.denoising_steps, config.label
        assert sliced.commitment_histogram == reference.commitment_histogram, config.label
        assert sliced.eos_position == reference.eos_position, config.label
        # Verifier accounting must not drift: one forward per microbatch chunk.
        assert sliced.verifier_rows == reference.verifier_rows, config.label
        assert (
            sliced.verifier_forward_passes == reference.verifier_forward_passes
        ), config.label
        _assert_traces_match(reference.traces, sliced.traces, config.label)


def test_step_traces_match_including_error_curve() -> None:
    full, lean = FullLogitsProvider(), LeanLogitsProvider()
    config = GenerationConfig(
        method="adaptive",
        continuation_length=12,
        delta=0.05,
        max_candidates=6,
        verifier_microbatch_size=4,
    )
    state = torch.cat(
        (_prompt(), torch.full((1, 12), full.mask_token_id, dtype=torch.long)), dim=1
    )
    reference = generation_step(full, state, canvas_start=4, config=config, step_index=0)
    sliced = generation_step(lean, state, canvas_start=4, config=config, step_index=0)
    _assert_traces_match([reference.trace], [sliced.trace], "single step")
    assert reference.accepted_prefix_length == sliced.accepted_prefix_length
    assert torch.equal(reference.state, sliced.state)
