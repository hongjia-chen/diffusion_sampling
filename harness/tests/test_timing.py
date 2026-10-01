"""Stage timing: off by default, additive, and never inside what it measures."""
from __future__ import annotations

import torch

from sdar_harness.blockwise import generate_blockwise
from sdar_harness.scheduler import GenerationConfig
from sdar_harness.timing import NULL_TIMER, StageTimer
from tests.test_blockwise import build


def test_a_disabled_timer_records_nothing_and_costs_no_branches() -> None:
    timer = StageTimer(torch.device("cpu"), enabled=False)
    with timer("proposer"):
        pass
    assert timer.totals() == {}
    assert timer.counts() == {}
    assert NULL_TIMER.totals() == {}


def test_an_enabled_cpu_timer_accumulates_per_stage() -> None:
    timer = StageTimer(torch.device("cpu"), enabled=True)
    for _ in range(3):
        with timer("proposer"):
            pass
    with timer("verifier"):
        pass
    totals = timer.totals()
    assert set(totals) == {"proposer", "verifier"}
    assert timer.counts() == {"proposer": 3, "verifier": 1}
    assert all(value >= 0.0 for value in totals.values())


def test_merge_folds_one_timer_into_another() -> None:
    left = StageTimer(torch.device("cpu"), enabled=True)
    right = StageTimer(torch.device("cpu"), enabled=True)
    with left("commit"):
        pass
    with right("commit"):
        pass
    with right("oracle"):
        pass
    left.merge(right)
    assert left.counts() == {"commit": 2, "oracle": 1}


def test_a_timed_stage_that_raises_is_still_recorded() -> None:
    """The context manager must close its span on the way out, not swallow it."""
    timer = StageTimer(torch.device("cpu"), enabled=True)
    try:
        with timer("proposer"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert timer.counts() == {"proposer": 1}


def test_profiling_does_not_change_what_the_sampler_produces() -> None:
    """The whole point: the breakdown must explain the run, not alter it."""
    _, provider = build(block_length=4)
    prompt = torch.arange(8, dtype=torch.long).unsqueeze(0) % 5
    config = GenerationConfig(
        method="fixed", continuation_length=8, fixed_k=2,
        max_candidates=4, verifier_microbatch_size=2,
    )
    torch.manual_seed(0)
    plain = generate_blockwise(provider, prompt, config)
    torch.manual_seed(0)
    timed = generate_blockwise(
        provider, prompt, config, timer=StageTimer(torch.device("cpu"), enabled=True)
    )
    assert timed.continuation_token_ids == plain.continuation_token_ids
    assert timed.denoising_steps == plain.denoising_steps
    assert plain.stage_seconds == {}
    assert set(timed.stage_seconds) == {"proposer", "commit"}


def test_adaptive_profiling_separates_proposer_from_verifier() -> None:
    _, provider = build(block_length=4)
    prompt = torch.arange(8, dtype=torch.long).unsqueeze(0) % 5
    config = GenerationConfig(
        method="adaptive", continuation_length=8, delta=0.5,
        max_candidates=4, verifier_microbatch_size=4,
    )
    torch.manual_seed(0)
    result = generate_blockwise(
        provider, prompt, config, timer=StageTimer(torch.device("cpu"), enabled=True)
    )
    assert set(result.stage_seconds) == {"proposer", "verifier", "commit"}
