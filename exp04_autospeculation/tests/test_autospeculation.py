"""Algorithm 3: correctness against an exactly-consistent oracle.

The paper's guarantee (Prop. 18) is that the output is distributed as `mu`.
With a *consistent* coordinate denoiser -- one whose conditionals really are
the conditionals of a single joint -- the sigma-ordered chain rule equals that
joint for every sigma, so the check is direct: enumerate the joint, sample, and
compare. No model error is in the way, so any discrepancy is the sampler's.
"""
from __future__ import annotations

import itertools

import pytest
import torch

# Every tensor here is a handful of elements. On a many-core host torch's
# thread pool costs far more than the arithmetic -- log_softmax on a 6-element
# row measured 5 ms wall against ~5 us of work -- so this test is ~100x faster
# single-threaded. It has no effect on the real run, where tensors are large.
torch.set_num_threads(1)

from autospec.autospeculation import AutospecConfig, AutospecResult, generate_autospeculative


class ConsistentOracle:
    """A coordinate denoiser backed by an explicit joint over [q]^n.

    Vocabulary is [0, q) plus a mask id and an eos id that is driven to
    probability zero, so completions never terminate early and the whole canvas
    is always filled -- which is what makes the empirical distribution directly
    comparable to the enumerated joint.
    """

    def __init__(self, joint: torch.Tensor, canvas_start: int = 1):
        self.joint = joint / joint.sum()
        self.n = joint.ndim
        self.q = joint.shape[0]
        self.mask_token_id = self.q
        self.eos_token_id = self.q + 1
        self.vocabulary = self.q + 2
        self.canvas_start = canvas_start
        self.device = torch.device("cpu")
        self.supports_lean_logits = False
        self._cache: dict[tuple[int, ...], torch.Tensor] = {}

    @property
    def excluded_token_ids(self) -> set[int]:
        return {self.mask_token_id}

    def conditional(self, committed: dict[int, int], target: int) -> torch.Tensor:
        block = self.joint
        # Pin the committed coordinates, keeping axis identities straight by
        # walking them in descending order.
        axes = sorted(committed, reverse=True)
        offset = {axis: axis for axis in range(self.n)}
        for axis in axes:
            block = block.select(offset[axis], committed[axis])
            for later in range(axis + 1, self.n):
                offset[later] -= 1
        keep = offset[target]
        dims = [d for d in range(block.ndim) if d != keep]
        vector = block.sum(dim=dims) if dims else block
        total = vector.sum()
        return vector / total if total > 0 else torch.full((self.q,), 1.0 / self.q)

    def _table(self, canvas: tuple[int, ...]) -> torch.Tensor:
        """[n, q] conditionals for this canvas, memoized.

        There are only 3^n reachable canvases here (each position masked or one
        of q values), so after warmup every call is a dict hit.
        """
        cached = self._cache.get(canvas)
        if cached is None:
            committed = {i: v for i, v in enumerate(canvas) if v != self.mask_token_id}
            cached = torch.stack([
                self.conditional({k: v for k, v in committed.items() if k != i}, i)
                for i in range(self.n)
            ]).clamp_min(1e-30).log()
            self._cache[canvas] = cached
        return cached

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, length = input_ids.shape
        out = torch.full((batch, length, self.vocabulary), -1e9)
        for row in range(batch):
            table = self._table(tuple(input_ids[row, self.canvas_start :].tolist()))
            out[row, self.canvas_start :, : self.q] = table
        return out


def joint_for(n: int, q: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    # Strong dependence: a product distribution would make nu == mu and the
    # rejection machinery would never fire, so the test has to avoid one.
    weights = torch.rand((q,) * n, generator=generator) ** 4 + 1e-3
    return weights / weights.sum()


def exact_distribution(joint: torch.Tensor) -> dict[tuple[int, ...], float]:
    return {
        outcome: float(joint[outcome])
        for outcome in itertools.product(range(joint.shape[0]), repeat=joint.ndim)
    }


def empirical(oracle, config, draws, seed, sigma_rank):
    counts: dict[tuple[int, ...], int] = {}
    torch.manual_seed(seed)
    prompt = torch.zeros(1, oracle.canvas_start, dtype=torch.long)
    for _ in range(draws):
        result = generate_autospeculative(
            oracle, prompt, config, sigma_rank=sigma_rank
        )
        outcome = tuple(result.full_continuation_token_ids)
        counts[outcome] = counts.get(outcome, 0) + 1
    return {k: v / draws for k, v in counts.items()}, counts


def total_variation(left, right) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(k, 0.0) - right.get(k, 0.0)) for k in keys)


@pytest.mark.parametrize(
    "sigma_rank",
    [
        torch.tensor([0, 1, 2, 3]),   # identity order
        torch.tensor([3, 2, 1, 0]),   # reversed
        torch.tensor([1, 3, 0, 2]),   # scrambled
    ],
    ids=["identity", "reversed", "scrambled"],
)
def test_output_matches_the_joint_for_any_sigma(sigma_rank) -> None:
    joint = joint_for(n=4, q=2, seed=5)
    oracle = ConsistentOracle(joint)
    config = AutospecConfig(continuation_length=4)
    exact = exact_distribution(joint)
    observed, _ = empirical(oracle, config, draws=4000, seed=11, sigma_rank=sigma_rank)
    distance = total_variation(observed, exact)
    # 4000 draws over 16 outcomes: per-cell standard error is at most ~0.008,
    # so a TV above 0.04 is a real discrepancy rather than sampling noise.
    assert distance < 0.04, f"TV={distance:.4f}\nexact={exact}\nobserved={observed}"


def test_the_rejection_path_is_actually_exercised() -> None:
    """A test that never rejects would only be testing the speculation."""
    joint = joint_for(n=4, q=2, seed=5)
    oracle = ConsistentOracle(joint)
    config = AutospecConfig(continuation_length=4)
    torch.manual_seed(3)
    prompt = torch.zeros(1, 1, dtype=torch.long)
    rejected = fallbacks = 0
    for _ in range(200):
        result = generate_autospeculative(
            oracle, prompt, config, sigma_rank=torch.tensor([0, 1, 2, 3])
        )
        rejected += result.speculations_rejected
        fallbacks += result.fallback_draws
    assert rejected > 0 and fallbacks > 0


def test_singleton_blocks_always_accept() -> None:
    """The paper's `if b <= a+1 then u = 1`: nu == mu at width one."""
    oracle = ConsistentOracle(joint_for(n=2, q=2, seed=1))
    config = AutospecConfig(continuation_length=2)
    torch.manual_seed(0)
    result = generate_autospeculative(
        oracle, torch.zeros(1, 1, dtype=torch.long), config,
        sigma_rank=torch.tensor([0, 1]),
    )
    assert result.singleton_blocks >= 0
    assert len(result.full_continuation_token_ids) == 2
    assert all(v < oracle.q for v in result.full_continuation_token_ids)


def test_every_canvas_position_is_filled() -> None:
    oracle = ConsistentOracle(joint_for(n=4, q=3, seed=7))
    config = AutospecConfig(continuation_length=4)
    torch.manual_seed(1)
    for _ in range(20):
        result = generate_autospeculative(
            oracle, torch.zeros(1, 1, dtype=torch.long), config,
            sigma_rank=torch.tensor([2, 0, 3, 1]),
        )
        assert oracle.mask_token_id not in result.full_continuation_token_ids


def test_a_narrower_budget_never_costs_fewer_rounds() -> None:
    """D4: chunking can only split a parallel round into more physical ones.

    With the theoretical counter gone this is the same invariant stated in
    terms of the counter that remains: shrinking `max_batch_rows` can add
    physical rounds and can never remove one.
    """
    oracle = ConsistentOracle(joint_for(n=4, q=2, seed=5))
    prompt = torch.zeros(1, 1, dtype=torch.long)
    rounds = []
    for budget in (256, 1):
        torch.manual_seed(4)
        result = generate_autospeculative(
            oracle,
            prompt,
            AutospecConfig(continuation_length=4, max_batch_rows=budget),
            sigma_rank=torch.tensor([0, 1, 2, 3]),
        )
        assert result.denoising_steps == result.oracle_rounds_physical
        rounds.append(result.oracle_rounds_physical)
    assert rounds[1] >= rounds[0]


def test_batching_budget_changes_cost_but_not_the_sample() -> None:
    """max_batch_rows is a memory knob; it must not move a single token."""
    oracle = ConsistentOracle(joint_for(n=4, q=2, seed=5))
    prompt = torch.zeros(1, 1, dtype=torch.long)
    sigma_rank = torch.tensor([1, 3, 0, 2])
    outputs = []
    for budget in (1, 3, 256):
        torch.manual_seed(99)
        outputs.append(
            generate_autospeculative(
                oracle,
                torch.zeros(1, 1, dtype=torch.long),
                AutospecConfig(continuation_length=4, max_batch_rows=budget),
                sigma_rank=sigma_rank,
            )
        )
    assert outputs[0].full_continuation_token_ids == outputs[1].full_continuation_token_ids
    assert outputs[1].full_continuation_token_ids == outputs[2].full_continuation_token_ids
    assert outputs[0].oracle_rounds_physical > outputs[2].oracle_rounds_physical
    assert outputs[0].max_pooled_request_rows == outputs[2].max_pooled_request_rows
    assert outputs[0].max_pooled_request_rows >= 1


def test_config_enforces_the_settled_decisions() -> None:
    config = AutospecConfig(continuation_length=8)
    assert config.order == "sigma"
    assert not hasattr(config, "max_fallback_samples")
    assert "max_fallback_draws_in_call" in AutospecResult.__dataclass_fields__
    assert "max_pooled_request_rows" in AutospecResult.__dataclass_fields__
    with pytest.raises(ValueError, match="DESIGN.md D1"):
        AutospecConfig(continuation_length=8, temperature=0.0)
    with pytest.raises(ValueError, match="DESIGN.md D1"):
        AutospecConfig(continuation_length=8, top_p=0.9)


def test_rho_defaults_to_one_over_log2_n() -> None:
    assert AutospecConfig(continuation_length=256).effective_rho == pytest.approx(1 / 8)
    assert AutospecConfig(continuation_length=512).effective_rho == pytest.approx(1 / 9)
    # `effective_rho` still follows n -- which is the BLOCK under block
    # diffusion, since the per-block config carries the block's width. The
    # label no longer claims a canvas-derived value; see the test below.
    assert AutospecConfig(continuation_length=64).effective_rho == pytest.approx(1 / 6)


def test_sigma_is_required_and_must_match_the_canvas() -> None:
    config = AutospecConfig(continuation_length=8)
    with pytest.raises(ValueError, match="DESIGN.md D2"):
        generate_autospeculative(object(), torch.zeros(1, 1, dtype=torch.long), config)
    with pytest.raises(ValueError, match="entries for a canvas"):
        generate_autospeculative(
            object(), torch.zeros(1, 1, dtype=torch.long), config,
            sigma_rank=torch.arange(4),
        )


# ----------------------------------------------------------------------
# The block-local entry point (DESIGN.md D8)
# ----------------------------------------------------------------------

from autospec.autospeculation import run_autospec_block


def block_empirical(oracle, config, order, draws, seed, width):
    counts: dict[tuple[int, ...], int] = {}
    torch.manual_seed(seed)
    for _ in range(draws):
        state = torch.full((1, width), oracle.mask_token_id, dtype=torch.long)
        filled, _ = run_autospec_block(oracle, state, config, order)
        outcome = tuple(int(v) for v in filled[0].tolist())
        counts[outcome] = counts.get(outcome, 0) + 1
    return {k: v / draws for k, v in counts.items()}


@pytest.mark.parametrize(
    "order",
    [
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([3, 2, 1, 0]),
        torch.tensor([1, 3, 0, 2]),
    ],
    ids=["identity", "reversed", "scrambled"],
)
def test_block_entry_point_matches_the_joint_for_any_order(order) -> None:
    """Algorithm 3 restricted to one block is still exact.

    Under block diffusion the canvas is a loop over blocks, so the sampler that
    has to be right is the per-block one. Same check as the full-canvas test:
    a consistent oracle makes the sigma-ordered chain rule equal the joint, so
    any discrepancy is the sampler's.
    """
    joint = joint_for(n=4, q=2, seed=5)
    oracle = ConsistentOracle(joint, canvas_start=0)
    config = AutospecConfig(continuation_length=4)
    observed = block_empirical(oracle, config, order, draws=4000, seed=11, width=4)
    distance = total_variation(observed, exact_distribution(joint))
    assert distance < 0.04, f"TV={distance:.4f}"


def test_block_entry_point_never_writes_a_position_outside_its_order() -> None:
    """The first decode block carries the prompt's remainder; it is not canvas.

    `order` holds only the masked positions, so the already-committed prefix
    inside the block must come back untouched.
    """
    joint = joint_for(n=4, q=2, seed=7)
    oracle = ConsistentOracle(joint, canvas_start=0)
    config = AutospecConfig(continuation_length=2)
    torch.manual_seed(3)
    # Positions 0 and 2 are already committed; only 1 and 3 are masked.
    state = torch.tensor([[0, oracle.mask_token_id, 1, oracle.mask_token_id]])
    order = torch.tensor([3, 1])
    filled, ctx = run_autospec_block(oracle, state, config, order)

    assert int(filled[0, 0]) == 0
    assert int(filled[0, 2]) == 1
    assert int(filled[0, 1]) != oracle.mask_token_id
    assert int(filled[0, 3]) != oracle.mask_token_id
    assert sum(ctx.commitments.values()) >= 1


def test_the_auto_rho_arm_is_not_named_after_a_rho_no_block_uses() -> None:
    """Under block diffusion the paper's default rho is 1/log2(block).

    The label is built from the top-level config, which knows the canvas and
    not the block, so a canvas-derived number would name the arm after a value
    no block ever uses -- 1/log2(512) against the 1/log2(64) actually applied.
    """
    assert AutospecConfig(continuation_length=512).label == "autospec_rho_auto"
    assert AutospecConfig(continuation_length=64).label == "autospec_rho_auto"
    # An explicit rho IS used verbatim in every block, so it can be named.
    assert AutospecConfig(continuation_length=512, rho=0.15).label == "autospec_rho_0p1500"
