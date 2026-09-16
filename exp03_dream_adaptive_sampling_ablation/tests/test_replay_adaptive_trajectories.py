from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from dream_ablation.scheduler import GenerationConfig, generate


PATH = Path(__file__).resolve().parent.parent / "scripts" / "replay_adaptive_trajectories.py"
SPEC = importlib.util.spec_from_file_location("replay_adaptive_trajectories", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeProvider:
    mask_token_id = 9
    eos_token_id = 8
    excluded_token_ids = {9}
    device = torch.device("cpu")

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 10), -6.0)
        for position in range(1, length):
            logits[:, position, position] = 7.0 - 0.1 * position
        return logits


def test_compact_replay_matches_authoritative_generation() -> None:
    provider = FakeProvider()
    prompt = torch.tensor([[0]], dtype=torch.long)
    config = GenerationConfig(
        method="adaptive",
        continuation_length=5,
        delta=0.1,
        max_candidates=3,
        verifier_microbatch_size=2,
    )
    expected = generate(provider, prompt, config)
    actual = MODULE.generate_trajectory(provider, prompt, config)
    assert actual.token_ids == expected.continuation_token_ids
    assert actual.histogram == expected.commitment_histogram
    assert len(actual.commitments) == expected.denoising_steps
    assert sum(actual.commitments) == config.continuation_length
