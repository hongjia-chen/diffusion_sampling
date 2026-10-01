from __future__ import annotations

import torch

import pytest

from sdar_harness.scheduler import (
    GenerationConfig,
    finished_at,
    generate,
    generation_step,
    truncate_at_stop,
)


class FakeProvider:
    mask_token_id = 9
    eos_token_id = 8
    excluded_token_ids = {9}
    device = torch.device("cpu")

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 10), -6.0)
        for row in range(batch):
            for position in range(1, length):
                token = position
                logits[row, position, token] = 7.0 - 0.1 * position
                logits[row, position, 0] = 0.0
                # Once token 1 is committed, frozen y_2 becomes unlikely.
                if input_ids[row, 1].item() == 1 and position == 2:
                    logits[row, position, 2] = -7.0
                    logits[row, position, 7] = 7.0
        return logits


class EosProvider(FakeProvider):
    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = super().forward_logits(input_ids)
        # Prefer EOS at continuation position two before positions zero/one.
        logits[:, 3, :] = -6.0
        logits[:, 3, self.eos_token_id] = 12.0
        return logits


def prompt() -> torch.Tensor:
    return torch.tensor([[0]], dtype=torch.long)


def _adaptive_config() -> GenerationConfig:
    return GenerationConfig(
        method="adaptive",
        continuation_length=3,
        delta=0.01,
        max_candidates=3,
        verifier_microbatch_size=2,
    )


def test_adaptive_stops_at_the_microbatch_holding_the_first_failure() -> None:
    provider = FakeProvider()
    config = _adaptive_config()
    initial = torch.tensor([[0, 9, 9, 9]], dtype=torch.long)
    step = generation_step(provider, initial, canvas_start=1, config=config, step_index=0)
    # Candidate 1 violates the threshold, so candidate 2 cannot be accepted no
    # matter what it scores. The second microbatch is never run.
    assert step.trace["error_by_prefix_nats"][0] == 0.0
    assert step.trace["error_by_prefix_nats"][1] > config.tau_nats
    assert len(step.trace["error_by_prefix_nats"]) == 2
    assert step.trace["accepted_prefix_length"] == 1
    assert step.verifier_rows == 2
    assert step.verifier_forward_passes == 1
    assert initial.tolist() == [[0, 9, 9, 9]]


def test_early_exit_does_not_change_the_acceptance_decision() -> None:
    """Early exit is a cost optimisation; the committed prefix must be identical."""
    from sdar_harness.scheduler import _propose, _score_all_prefixes

    provider = FakeProvider()
    config = _adaptive_config()
    state = torch.tensor([[0, 9, 9, 9]], dtype=torch.long)
    remaining = torch.tensor([1, 2, 3])
    candidates = _propose(provider, state, remaining, config.max_candidates)

    def accepted_for(tau: float | None) -> int:
        _, _, errors, passes, rows = _score_all_prefixes(
            provider, state, candidates, config.verifier_microbatch_size, tau_nats=tau
        )
        passing = errors <= config.tau_nats
        passing[0] = True
        return int(torch.cumprod(passing.to(torch.int64), dim=0).sum().item()), passes, rows

    early, early_passes, early_rows = accepted_for(config.tau_nats)
    full, full_passes, full_rows = accepted_for(None)
    assert early == full
    assert early_passes < full_passes and early_rows < full_rows


def test_fixed_generation_fills_canvas() -> None:
    result = generate(
        FakeProvider(),
        prompt(),
        GenerationConfig(method="fixed", continuation_length=5, fixed_k=2),
    )
    assert result.continuation_token_ids == [1, 2, 3, 4, 5]
    assert result.denoising_steps == 3
    assert result.verifier_rows == 0
    assert result.commitment_histogram[:2] == [1, 2]


def test_eos_does_not_finish_while_earlier_positions_are_masked() -> None:
    result = generate(
        EosProvider(),
        prompt(),
        GenerationConfig(method="fixed", continuation_length=5, fixed_k=1, early_stop=False),
        capture_traces=True,
    )
    assert result.eos_position == 2
    # FakeProvider makes token 7 preferable at the second position after token
    # 1 is committed. The invariant under test is that this position is still
    # resolved before the already-committed EOS, not that it keeps its old
    # marginal argmax (token 2).
    assert result.continuation_token_ids == [1, 7]
    assert result.denoising_steps == 5
    assert not result.maximum_length_termination
    assert result.traces[0]["eos_position_after"] == 2
    assert result.traces[0]["remaining_masks_after"] == 4
    assert result.traces[-1]["remaining_masks_after"] == 0


def test_delta_is_converted_with_log1p() -> None:
    config = GenerationConfig(method="adaptive", continuation_length=3, delta=0.10)
    assert abs(config.tau_nats - 0.09531017980432487) < 1e-12


def test_eos_commit_does_not_mutate_suffix_for_either_scheduler() -> None:
    initial = torch.tensor([[0, 9, 9, 9, 9, 9]], dtype=torch.long)
    configs = [
        GenerationConfig(method="fixed", continuation_length=5, fixed_k=1),
        GenerationConfig(
            method="adaptive",
            continuation_length=5,
            delta=0.0,
            max_candidates=1,
            verifier_microbatch_size=1,
        ),
    ]
    for config in configs:
        step = generation_step(
            EosProvider(),
            initial,
            canvas_start=1,
            config=config,
            step_index=0,
        )
        assert initial.tolist() == [[0, 9, 9, 9, 9, 9]]
        assert step.state.tolist() == [[0, 9, 9, 8, 9, 9]]


def test_early_stop_returns_the_same_answer_in_fewer_steps() -> None:
    """EOS lands at position 2 first; positions 0 and 1 are filled next."""
    full = generate(
        EosProvider(), prompt(),
        GenerationConfig(method="fixed", continuation_length=5, fixed_k=1, early_stop=False),
    )
    early = generate(
        EosProvider(), prompt(),
        GenerationConfig(method="fixed", continuation_length=5, fixed_k=1),
    )
    assert early.continuation_token_ids == full.continuation_token_ids == [1, 7]
    assert early.eos_position == full.eos_position == 2
    assert early.denoising_steps == 3 and full.denoising_steps == 5
    assert early.early_stopped and not full.early_stopped
    assert early.decoded_positions == 3 and full.decoded_positions == 5
    assert early.stop_token_id == 8
    # Never-decoded positions are left masked, not filled.
    assert early.full_continuation_token_ids[3:] == [9, 9]


MASK = 9


@pytest.mark.parametrize(
    "canvas, expected",
    [
        ([1, 2, 8, MASK], 2),          # stop committed, everything before it too
        ([1, MASK, 8, MASK], None),    # a hole before the stop: not settled
        ([1, 2, 3, MASK], None),       # no stop token yet
        ([8, MASK, MASK, MASK], 0),    # stop at the very first position
        ([1, 7, 2, 8], 1),             # the FIRST stop token counts (7 and 8 both stop)
    ],
)
def test_finished_at(canvas, expected) -> None:
    result = finished_at(
        torch.tensor(canvas), mask_token_id=MASK, stop_ids=frozenset({7, 8})
    )
    assert result == expected


def test_truncate_at_the_first_of_several_stop_tokens() -> None:
    # SDAR's usual ending: <|im_end|> \n <|endoftext|>
    tokens = [10, 11, 151645, 198, 151643, 151643]
    scored, index, token = truncate_at_stop(tokens, frozenset({151645, 151643}))
    assert scored == [10, 11] and index == 2 and token == 151645
    scored, index, token = truncate_at_stop([10, 11], frozenset({151645}))
    assert scored == [10, 11] and index is None and token is None


class CalibratedProvider(FakeProvider):
    """Position p proposes token p with probability PROBS[p], else token 0."""

    PROBS = {1: 0.95, 2: 0.92, 3: 0.70, 4: 0.60}

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 10), -30.0)
        logits[:, :, 0] = 0.0
        for position, probability in self.PROBS.items():
            if position < length:
                logits[:, position, position] = float(torch.logit(torch.tensor(probability)))
        return logits


def _dynamic_step(threshold: float):
    config = GenerationConfig(
        method="dynamic", continuation_length=4, threshold=threshold, max_candidates=4
    )
    initial = torch.tensor([[0, 9, 9, 9, 9]], dtype=torch.long)
    return generation_step(CalibratedProvider(), initial, canvas_start=1, config=config, step_index=0)


def test_dynamic_commits_every_candidate_above_the_threshold() -> None:
    step = _dynamic_step(0.90)
    assert step.accepted_prefix_length == 2
    assert step.trace["committed_positions"] == [0, 1]
    assert step.state.tolist() == [[0, 1, 2, 9, 9]]
    assert step.verifier_rows == 0 and step.verifier_forward_passes == 0
    assert step.trace["dynamic_threshold"] == 0.90


def test_dynamic_falls_back_to_one_token_when_nothing_is_confident() -> None:
    # SDAR's floor: num_transfer_tokens = block_length / denoising_steps = 1.
    step = _dynamic_step(0.99)
    assert step.accepted_prefix_length == 1
    assert step.state.tolist() == [[0, 1, 9, 9, 9]]


def test_dynamic_generation_finishes_the_canvas() -> None:
    config = GenerationConfig(
        method="dynamic", continuation_length=4, threshold=0.90, max_candidates=4,
        early_stop=False,
    )
    result = generate(CalibratedProvider(), prompt(), config)
    assert result.continuation_token_ids == [1, 2, 3, 4]
    assert sum(result.commitment_histogram) == result.denoising_steps == 3
    assert result.commitment_histogram[:2] == [2, 1]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(method="dynamic", threshold=1.0),
        dict(method="dynamic", threshold=0.0),
        dict(method="dynamic"),
        dict(method="dynamic", threshold=0.9, confidence="entropy"),
        dict(method="dynamic", threshold=0.9, delta=0.1),
        dict(method="fixed", fixed_k=1, threshold=0.9),
    ],
)
def test_dynamic_config_is_validated(kwargs) -> None:
    with pytest.raises(ValueError):
        GenerationConfig(continuation_length=4, **kwargs)


def test_dynamic_label() -> None:
    config = GenerationConfig(method="dynamic", continuation_length=4, threshold=0.9)
    assert config.label == "dynamic_theta_0p90"
