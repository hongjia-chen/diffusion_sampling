from __future__ import annotations

import torch

from dream_ablation.scheduler import GenerationConfig, generate, generation_step


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
    from dream_ablation.scheduler import _propose, _score_all_prefixes

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
        GenerationConfig(method="fixed", continuation_length=5, fixed_k=1),
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
