"""The permutation Algorithm 3 samples in (DESIGN.md D2).

Nothing else in the experiment consumes it: fixed-k and adaptive order their
positions by confidence and never see a permutation.
"""
from __future__ import annotations

import torch

from scripts.run_benchmark import sigma_rank_for_task, sigma_seed


def test_sigma_rank_is_a_permutation() -> None:
    rank = sigma_rank_for_task(1, "x", 128, torch.device("cpu"))
    assert torch.equal(rank.sort().values, torch.arange(128))


def test_sigma_is_reproducible_and_task_specific() -> None:
    a = sigma_rank_for_task(7, "gsm8k-test-3", 64, torch.device("cpu"))
    assert torch.equal(a, sigma_rank_for_task(7, "gsm8k-test-3", 64, torch.device("cpu")))
    assert not torch.equal(a, sigma_rank_for_task(7, "gsm8k-test-4", 64, torch.device("cpu")))
    assert sigma_seed(7, "x") != sigma_seed(8, "x")


def test_sigma_seed_omits_the_method() -> None:
    """Two configurations of Algorithm 3 must walk the same permutation.

    `task_seed` mixes in the method label so token draws differ per arm;
    `sigma_seed` deliberately does not, or a rho sweep would compare different
    random orders rather than different rho.
    """
    from scripts.run_benchmark import task_seed

    assert task_seed(1, "autospec_rho_a", "x") != task_seed(1, "autospec_rho_b", "x")
    assert sigma_seed(1, "x") == sigma_seed(1, "x")
    assert "autospec" not in f"sigma:{1}:{'x'}"


def test_sigma_is_uniform_enough_to_be_a_random_order() -> None:
    """Guards against an accidental identity or near-identity permutation."""
    total = 0
    for index in range(40):
        rank = sigma_rank_for_task(0, f"task-{index}", 256, torch.device("cpu"))
        total += int((rank == torch.arange(256)).sum())
    # Expected fixed points under a uniform permutation is 1 per draw.
    assert total < 40 * 8
