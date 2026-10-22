"""The η measurement, against block-diffusion fakes with a known in-block dependence.

The fakes are test_blockwise's SDAR fake (position % 5, the real cache
protocol) plus one coupling: a filled left neighbour t raises the logit of
token (t + 2) % 5 by `strength`. Strength 0 is a model with no in-block
dependence; two different strengths are two models that depend on the same
neighbours by different amounts.
"""
from __future__ import annotations

import math
import random

import pytest
import torch

from sdar_harness.blockwise import generate_blockwise
from sdar_harness.dependency import (
    MeasureConfig,
    _distance_to_new,
    dependency_metrics,
    generate_measured,
    probe_rows,
)
from sdar_harness.model import FAMILIES, MaskedDiffusionProvider
from sdar_harness.scheduler import GenerationConfig
from tests.test_blockwise import EOS, MASK, FakeBackbone, FakeHead, FakeSDAR, Tok

BLOCK = 4


class CoupledBackbone(FakeBackbone):
    """FakeBackbone, plus the in-block left neighbour's token in channel 2 (-1 if masked)."""

    def __call__(self, *, input_ids, **kwargs):
        out = super().__call__(input_ids=input_ids, **kwargs)
        left = torch.full(input_ids.shape, -1.0)
        left[:, 1:] = input_ids[:, :-1].float()
        left[:, 1:][input_ids[:, :-1] == MASK] = -1.0
        out.last_hidden_state[..., 2] = left
        return out


class CoupledHead(FakeHead):
    def __init__(self, strength: float):
        super().__init__()
        self.strength = strength

    def forward(self, hidden):
        logits = super().forward(hidden)
        left = hidden[..., 2].long()
        bumped = ((left.clamp(min=0) + 2) % 5).unsqueeze(-1)
        current = logits.gather(-1, bumped)
        raised = torch.where((left >= 0).unsqueeze(-1), current + self.strength, current)
        return logits.scatter(-1, bumped, raised)


class CoupledSDAR(FakeSDAR):
    def __init__(self, strength: float):
        super().__init__()
        self.model = CoupledBackbone(self.log)
        self.lm_head = CoupledHead(strength)


def provider(strength: float) -> MaskedDiffusionProvider:
    conventions = FAMILIES["sdar"].__class__(
        logits_alignment="none", attention_mask="block_causal",
        loader="auto_model_for_causal_lm", block_length=BLOCK,
    )
    return MaskedDiffusionProvider(
        CoupledSDAR(strength), Tok(), torch.device("cpu"), conventions, stop_token_ids=(EOS,),
    )


def prompt(length: int = 8) -> torch.Tensor:
    return (torch.arange(length, dtype=torch.long) % 5).unsqueeze(0)


def measure(p_strength=10.0, v_strength=4.0, *, prompt_length=8, generated=12, seed=0, **kwargs):
    config = MeasureConfig(continuation_length=generated, **kwargs)
    return generate_measured(
        provider(p_strength), provider(v_strength), prompt(prompt_length), config,
        rng=random.Random(seed),
    )


def ranks(result, name):
    return [value for record in result.records for value in record[name]]


@pytest.mark.parametrize("prompt_length", [8, 6])
@pytest.mark.parametrize("per_block", [0, 1, 4])
def test_the_decode_is_fixed1_blockwise_whatever_is_measured(prompt_length, per_block) -> None:
    baseline = generate_blockwise(
        provider(10.0), prompt(prompt_length),
        GenerationConfig(method="fixed", continuation_length=12, fixed_k=1, max_candidates=BLOCK),
    )
    result = measure(prompt_length=prompt_length, measure_per_block=per_block)
    assert result.continuation_token_ids == baseline.continuation_token_ids
    assert result.full_continuation_token_ids == baseline.full_continuation_token_ids
    assert result.denoising_steps == baseline.denoising_steps
    assert result.commit_passes == baseline.commit_passes


def test_the_coupling_moves_the_proposer_but_not_its_argmax() -> None:
    result = measure(measure_per_block=4)
    assert max(ranks(result, "D_P")) > 0.01
    assert ranks(result, "argmax_pP") == ranks(result, "argmax_qP")
    assert ranks(result, "argmax_qP") == ranks(result, "tokens")


def test_rank_one_carries_no_dependence() -> None:
    result = measure(measure_per_block=4)
    for record in result.records:
        assert record["D_P"][0] == record["D_V"][0] == record["eta"][0] == 0.0
        assert record["lr_P"][0] == record["lr_V"][0] == 0.0
        assert record["dist_new"][0] is None


def test_a_reference_identical_to_the_proposer_has_zero_eta() -> None:
    result = measure(10.0, 10.0, measure_per_block=4)
    assert ranks(result, "eta") == [0.0] * len(ranks(result, "eta"))
    assert ranks(result, "D_P") == ranks(result, "D_V")
    assert ranks(result, "tv_PV") == [0.0] * len(ranks(result, "tv_PV"))


def test_a_reference_with_no_dependence_leaves_eta_equal_to_D_P() -> None:
    result = measure(10.0, 0.0, measure_per_block=4)
    assert ranks(result, "D_V") == [0.0] * len(ranks(result, "D_V"))
    assert ranks(result, "eta") == ranks(result, "D_P")


def test_eta_bounds_the_transfer_both_ways() -> None:
    result = measure(10.0, 4.0, measure_per_block=4)
    d_p, d_v, eta = ranks(result, "D_P"), ranks(result, "D_V"), ranks(result, "eta")
    assert any(value > 0 for value in eta)
    for p, v, e in zip(d_p, d_v, eta):
        assert p <= v + e + 1e-6
        assert abs(p - v) <= e + 1e-6
        assert v < p or p == v == 0.0  # the reference's coupling is the weaker one


def test_one_measured_step_per_block_with_at_least_two_slots() -> None:
    result = measure(measure_per_block=1, prompt_length=6, generated=12)
    blocks = [record["block"] for record in result.records]
    assert blocks == sorted(set(blocks))
    # Prompt 6 leaves 2 masked slots in block 1, then blocks 2, 3 and 4.
    assert blocks == [1, 2, 3, 4]
    for record in result.records:
        assert record["k"] >= 2
        assert record["k"] == len(record["slots"]) == len(record["eta"])
        assert record["filled_in_block"] == BLOCK - record["k"]
        assert record["context_tokens"] == record["block_start"] + record["filled_in_block"]


def test_the_choice_of_steps_is_seeded() -> None:
    first = [r["step"] for r in measure(seed=7).records]
    again = [r["step"] for r in measure(seed=7).records]
    other = [[r["step"] for r in measure(seed=seed).records] for seed in range(8)]
    assert first == again
    assert any(steps != first for steps in other)


def test_the_cap_on_measured_steps_per_task() -> None:
    result = measure(measure_per_block=4, max_measured=3, generated=16)
    assert len(result.records) == 3


def test_row_grouping_does_not_change_the_numbers_in_exact_arithmetic() -> None:
    whole = measure(measure_per_block=4, group_rows=4)
    single = measure(measure_per_block=4, group_rows=1)
    for name in ("D_P", "D_V", "eta", "lr_P", "conf_V"):
        assert ranks(whole, name) == ranks(single, name)
    assert whole.probe_forwards["proposer"] < single.probe_forwards["proposer"]


def test_the_self_floor_is_zero_on_the_cpu_fake() -> None:
    result = measure(measure_per_block=4, group_rows=4, floor_group_rows=1)
    assert ranks(result, "eta_self") == [0.0] * len(ranks(result, "eta_self"))
    assert ranks(result, "noise_q") == [0.0] * len(ranks(result, "noise_q"))
    assert result.probe_forwards["floor"] > 0


def test_probe_rows_fill_the_higher_ranks() -> None:
    state = torch.full((1, 6), MASK)
    positions = torch.tensor([4, 1, 3])
    tokens = torch.tensor([0, 1, 2])
    rows = probe_rows(state, positions, tokens)
    assert rows[0].tolist() == [MASK] * 6
    assert rows[1].tolist() == [MASK, MASK, MASK, MASK, 0, MASK]
    assert rows[2].tolist() == [MASK, 1, MASK, MASK, 0, MASK]


def test_metrics_on_hand_made_distributions() -> None:
    def logits(*probs):
        return torch.tensor([[math.log(p) for p in probs]])

    marginal_p = logits(0.5, 0.3, 0.2)
    conditional_p = logits(0.2, 0.3, 0.5)
    marginal_v = logits(0.6, 0.2, 0.2)
    conditional_v = logits(0.4, 0.2, 0.4)
    out = dependency_metrics(conditional_p, marginal_p, conditional_v, marginal_v,
                             torch.tensor([0]), excluded=set())
    assert out["D_P"][0] == pytest.approx(0.3)
    assert out["D_V"][0] == pytest.approx(0.2)
    # (p - q)^P = (-.3, 0, .3), (p - q)^V = (-.2, 0, .2): half the L1 of the difference.
    assert out["eta"][0] == pytest.approx(0.1)
    assert out["tv_PV"][0] == pytest.approx(0.1)
    assert out["gap_P"][0] == pytest.approx(0.2)
    assert out["conf_V"][0] == pytest.approx(0.6)
    assert out["lr_P"][0] == pytest.approx(math.log(0.2 / 0.5))
    assert out["argmax_pP"][0] == 2 and out["argmax_qP"][0] == 0


def test_excluded_tokens_are_outside_the_support() -> None:
    marginal = torch.tensor([[0.0, 0.0, 5.0]])
    conditional = torch.tensor([[0.0, 1.0, -5.0]])
    out = dependency_metrics(conditional, marginal, marginal, marginal,
                             torch.tensor([0]), excluded={2})
    expected = 0.5 * (abs(1 / (1 + math.e) - 0.5) + abs(math.e / (1 + math.e) - 0.5))
    assert out["D_P"][0] == pytest.approx(expected)
    assert out["D_V"][0] == 0.0


def test_distance_to_the_newly_filled_slots() -> None:
    assert _distance_to_new([3, 1, 4, 2]) == [None, 2, 1, 1]


def test_a_reference_on_another_block_grid_is_refused() -> None:
    other = provider(1.0)
    other.conventions = FAMILIES["sdar"].__class__(
        logits_alignment="none", attention_mask="block_causal",
        loader="auto_model_for_causal_lm", block_length=8,
    )
    with pytest.raises(ValueError, match="block length"):
        generate_measured(provider(1.0), other, prompt(), MeasureConfig(continuation_length=8),
                          rng=random.Random(0))
