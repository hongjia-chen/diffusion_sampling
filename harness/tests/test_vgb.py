"""Density-ratio VGB against exact toy distributions.

Two strictly positive joints over {0,1}^n play pi_b and pi_s. Every marginal
is enumerable, so V(z) = pi_b(z)/pi_s(z) is exact and the reduced rule in
`sdar_harness.vgb` can be checked against the raw Definition 4.2 weights and
against the exact stationary law of the chain it defines.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

import pytest
import torch

from sdar_harness.vgb import (
    DOWN,
    UP,
    VGBConfig,
    choose_direction,
    local_moves,
    walk_block,
)

MASK = 2
VOCAB = 2


class Joint:
    def __init__(self, n: int, seed: int):
        gen = torch.Generator().manual_seed(seed)
        weights = torch.rand(VOCAB**n, generator=gen, dtype=torch.float64) + 0.05
        self.n = n
        self.table = {
            y: float(w) for y, w in zip(itertools.product(range(VOCAB), repeat=n), weights / weights.sum())
        }

    def marginal(self, z: tuple[int, ...]) -> float:
        return sum(p for y, p in self.table.items()
                   if all(zi == MASK or zi == yi for zi, yi in zip(z, y)))

    def conditional(self, z: tuple[int, ...], i: int) -> list[float]:
        base = self.marginal(z)
        return [self.marginal(z[:i] + (a,) + z[i + 1:]) / base for a in range(VOCAB)]

    def scorer(self, states, rows, positions):
        out = torch.full((len(rows), VOCAB + 1), -math.inf, dtype=torch.float32)
        for r, (row, pos) in enumerate(zip(rows.tolist(), positions.tolist())):
            z = tuple(states[row].tolist())
            out[r, :VOCAB] = torch.tensor(self.conditional(z, pos)).log()
        return out


def all_states(n):
    return list(itertools.product(range(VOCAB + 1), repeat=n))


def unmask(z, i, a):
    return z[:i] + (a,) + z[i + 1:]


def remask(z, i):
    return z[:i] + (MASK,) + z[i + 1:]


def value(big, small, z):
    return big.marginal(z) / small.marginal(z)


def log_rho(big, small, z):
    revealed = [i for i, t in enumerate(z) if t != MASK]
    values = [
        math.log(small.conditional(remask(z, i), i)[z[i]]) - math.log(big.conditional(remask(z, i), i)[z[i]])
        for i in revealed
    ]
    return revealed, torch.tensor(values, dtype=torch.float64)


def s(n, k):
    return 1.0 / math.comb(n - 1, k)


def transitions(big, small, z, lam, leaf_rule):
    """{next_state: probability} from the reduced rule."""
    n = len(z)
    k = sum(t != MASK for t in z)
    revealed, rho = log_rho(big, small, z)
    moves = local_moves(k, n, rho, lam=lam, leaf_rule=leaf_rule,
                        log_value=math.log(value(big, small, z)))
    out: dict[tuple, float] = {}
    p_forward = moves.forward_probability
    masked = [i for i, t in enumerate(z) if t == MASK]
    for i in masked:
        for a, p in enumerate(big.conditional(z, i)):
            out[unmask(z, i, a)] = out.get(unmask(z, i, a), 0.0) + p_forward * p / len(masked)
    if revealed:
        back = torch.softmax(moves.backward_logits, 0).tolist()
        for i, p in zip(revealed, back):
            out[remask(z, i)] = out.get(remask(z, i), 0.0) + (1 - p_forward) * p
    return out


def stationary(matrix):
    values, vectors = torch.linalg.eig(matrix.T)
    index = int(torch.argmin((values - 1).abs()))
    vector = vectors[:, index].real
    return vector / vector.sum()


@pytest.fixture
def pair():
    return Joint(3, seed=1), Joint(3, seed=2)


# ----------------------------------------------------------------------


def test_balancing_identity():
    for n in range(2, 9):
        for k in range(1, n):
            assert (n - k) * s(n, k) == pytest.approx(k * s(n, k - 1))


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0, 2.5])
def test_reduced_rule_matches_raw_definition_4_2(lam):
    """Direction and backward split equal the unreduced weights, exact V."""
    n = 4
    big, small = Joint(n, 3), Joint(n, 4)
    for z in all_states(n):
        k = sum(t != MASK for t in z)
        if k in (0, n):
            continue
        v = value(big, small, z)
        forward = 0.0
        for i in (i for i, t in enumerate(z) if t == MASK):
            for a in range(VOCAB):
                c = unmask(z, i, a)
                p_ref = small.conditional(z, i)[a]
                if k + 1 == n:  # leaf child: no V(z)^lambda, tau(c) = V(c)
                    forward += s(n, k) * p_ref * value(big, small, c)
                else:
                    forward += s(n, k) * v**lam * value(big, small, c) * p_ref
        revealed, rho = log_rho(big, small, z)
        raw_back = [s(n, k - 1) * v * value(big, small, remask(z, i)) ** lam for i in revealed]
        moves = local_moves(k, n, rho, lam=lam, leaf_rule="exact", log_value=math.log(v))
        assert moves.forward_probability == pytest.approx(forward / (forward + sum(raw_back)), rel=1e-9)
        split = torch.softmax(moves.backward_logits, 0)
        assert split.tolist() == pytest.approx([w / sum(raw_back) for w in raw_back], rel=1e-9)


def test_interior_forward_probability_is_one_over_one_plus_mean():
    rho = torch.tensor([0.3, -1.2, 0.8], dtype=torch.float64)
    for lam in (0.5, 1.0, 2.0):
        moves = local_moves(3, 10, rho, lam=lam)
        mean = float((lam * rho).exp().mean())
        assert moves.forward_probability == pytest.approx(1 / (1 + mean))
    assert local_moves(3, 10, None, lam=0.0).forward_probability == pytest.approx(0.5)
    assert local_moves(0, 10, None, lam=0.0).forward_probability == 1.0


def test_shortlist_scale_recovers_full_sum():
    rho = torch.zeros(2, dtype=torch.float64)
    full = local_moves(6, 10, torch.zeros(6, dtype=torch.float64), lam=1.0)
    short = local_moves(6, 10, rho, lam=1.0, shortlist_scale=3.0)
    assert short.forward_probability == pytest.approx(full.forward_probability)


@pytest.mark.parametrize("lam", [0.0, 0.7, 2.0])
def test_exact_leaf_rule_targets_big_model(pair, lam):
    """Theorem 4.1 with tau = pi_b/pi_s: leaf-conditioned stationary law = pi_b."""
    big, small = pair
    states = all_states(3)
    index = {z: j for j, z in enumerate(states)}
    matrix = torch.zeros(len(states), len(states), dtype=torch.float64)
    for z in states:
        for nxt, p in transitions(big, small, z, lam, "exact").items():
            matrix[index[z], index[nxt]] += p
    assert torch.allclose(matrix.sum(1), torch.ones(len(states), dtype=torch.float64))
    mu = stationary(matrix)
    leaves = [z for z in states if MASK not in z]
    law = torch.tensor([float(mu[index[y]]) for y in leaves], dtype=torch.float64)
    law /= law.sum()
    assert law.tolist() == pytest.approx([big.table[y] for y in leaves], abs=1e-9)


@pytest.mark.parametrize("lam", [0.7, 2.0])
def test_interior_leaf_rule_targets_contrastive_tilt(pair, lam):
    """Official-code leaf handling: leaf law ∝ pi_b(y) sum_i V(y^{-i})^lambda."""
    big, small = pair
    states = all_states(3)
    index = {z: j for j, z in enumerate(states)}
    matrix = torch.zeros(len(states), len(states), dtype=torch.float64)
    for z in states:
        for nxt, p in transitions(big, small, z, lam, "interior").items():
            matrix[index[z], index[nxt]] += p
    mu = stationary(matrix)
    leaves = [z for z in states if MASK not in z]
    law = torch.tensor([float(mu[index[y]]) for y in leaves], dtype=torch.float64)
    law /= law.sum()
    tilt = torch.tensor([
        big.table[y] * sum(value(big, small, remask(y, i)) ** lam for i in range(3)) for y in leaves
    ], dtype=torch.float64)
    tilt /= tilt.sum()
    assert law.tolist() == pytest.approx(tilt.tolist(), abs=1e-9)
    assert not torch.allclose(law, torch.tensor([big.table[y] for y in leaves], dtype=law.dtype), atol=1e-3)


@pytest.mark.parametrize("chi", [0.0, 0.5, 1.0])
def test_momentum_lift_preserves_stationary_law(pair, chi):
    """Theorem 4.4: the lifted chain projects onto the plain chain's law."""
    big, small = pair
    lam = 1.0
    states = all_states(3)
    lifted = [(z, d) for z in states for d in (DOWN, UP)]
    index = {s_: j for j, s_ in enumerate(lifted)}
    matrix = torch.zeros(len(lifted), len(lifted), dtype=torch.float64)
    plain = torch.zeros(len(states), len(states), dtype=torch.float64)
    for z in states:
        k = sum(t != MASK for t in z)
        _, rho = log_rho(big, small, z)
        moves = local_moves(k, 3, rho, lam=lam, leaf_rule="exact",
                            log_value=math.log(value(big, small, z)))
        p_forward = moves.forward_probability
        full = transitions(big, small, z, lam, "exact")
        for nxt, p in full.items():
            plain[states.index(z), states.index(nxt)] += p
        for d in (DOWN, UP):
            # Integrate choose_direction over its uniform draw exactly.
            cuts = sorted({0.0, 1.0, p_forward, 1 - p_forward,
                           p_forward + chi * min(p_forward, 1 - p_forward),
                           1 - p_forward + chi * min(p_forward, 1 - p_forward)})
            for low, high in zip(cuts, cuts[1:]):
                if high - low <= 0 or high > 1:
                    continue
                choice = choose_direction(moves, d, chi, (low + high) / 2)
                mass = high - low
                if choice.move in ("stay", "switch"):
                    matrix[index[(z, d)], index[(z, choice.direction)]] += mass
                    continue
                forward = choice.move == "forward"
                share = p_forward if forward else 1 - p_forward
                for nxt, p in full.items():
                    if (sum(t != MASK for t in nxt) > k) == forward:
                        matrix[index[(z, d)], index[(nxt, d)]] += mass * p / share
    assert torch.allclose(matrix.sum(1), torch.ones(len(lifted), dtype=torch.float64))
    mu = stationary(matrix)
    projected = torch.tensor([float(mu[index[(z, DOWN)]] + mu[index[(z, UP)]]) for z in states], dtype=torch.float64)
    assert projected.tolist() == pytest.approx(stationary(plain).tolist(), abs=1e-9)


# ----------------------------------------------------------------------
# The sampler itself, on the toy scorers.
# ----------------------------------------------------------------------


def run(config, big, small, start, editable, seed):
    gen = torch.Generator().manual_seed(seed)
    return walk_block(
        torch.tensor(start), torch.tensor(editable), big.scorer, small.scorer, config,
        mask_token_id=MASK, generator=gen,
    )


def test_forward_only_is_ancestral_sampling_of_big(pair):
    big, small = pair
    config = VGBConfig(continuation_length=3, backward=False)
    counts: dict[tuple, int] = {}
    trials = 6000
    for seed in range(trials):
        walk = run(config, big, small, (MASK,) * 3, (True,) * 3, seed)
        y = tuple(walk.state.tolist())
        counts[y] = counts.get(y, 0) + 1
        assert walk.backward_moves == 0 and walk.reference_rows == 0
    for y, p in big.table.items():
        assert counts.get(y, 0) / trials == pytest.approx(p, abs=4 * math.sqrt(p * (1 - p) / trials))


@pytest.mark.parametrize("leaf_rule", ["exact", "interior"])
def test_first_step_matches_local_moves(pair, leaf_rule):
    big, small = pair
    start = (1, MASK, 0)  # k = 2 = n - 1: the leaf boundary
    config = VGBConfig(continuation_length=3, lam=1.5, leaf_rule=leaf_rule,
                       max_steps_per_block=1, trace=True)
    forward = 0
    trials = 4000
    for seed in range(trials):
        walk = run(config, big, small, start, (True,) * 3, seed)
        forward += walk.trace[0]["move"] == "forward"
    revealed, rho = log_rho(big, small, start)
    # The walker starts its log V at 0: the block root. From a mid-block start
    # the exact rule therefore sees V = 1, not the true value.
    expected = local_moves(2, 3, rho, lam=1.5, leaf_rule=leaf_rule, log_value=0.0).forward_probability
    p = expected
    assert forward / trials == pytest.approx(p, abs=4 * math.sqrt(p * (1 - p) / trials))


def test_walk_tracks_value_and_counts_rows(pair):
    big, small = pair
    config = VGBConfig(continuation_length=3, lam=1.0, leaf_rule="exact", trace=True)
    walk = run(config, big, small, (MASK,) * 3, (True,) * 3, seed=7)
    assert MASK not in walk.state.tolist()
    assert walk.steps == walk.forward_moves + walk.backward_moves
    assert walk.forward_moves - walk.backward_moves == 3
    # Tracked log V equals the exact marginal ratio at every visited state.
    state = [MASK] * 3
    for record in walk.trace:
        assert record["log_value"] == pytest.approx(math.log(value(big, small, tuple(state))), abs=1e-5)
        if record["move"] == "forward":
            state[record["position"]] = record["token"]
        else:
            state[record["position"]] = MASK


def test_prompt_tail_positions_are_never_touched(pair):
    big, small = pair
    config = VGBConfig(continuation_length=2, lam=1.0, momentum_chi=1.0, trace=True)
    for seed in range(50):
        walk = run(config, big, small, (1, MASK, MASK), (False, True, True), seed)
        assert walk.state[0] == 1
        assert all(record.get("position", 1) != 0 for record in walk.trace)


def test_budget_forces_forward_completion(pair):
    big, small = pair
    config = VGBConfig(continuation_length=3, lam=1.0, max_steps_per_block=1)
    walk = run(config, big, small, (MASK,) * 3, (True,) * 3, seed=0)
    assert walk.forced_completion and MASK not in walk.state.tolist()
    assert walk.forced_positions == 2


def test_config_validation():
    with pytest.raises(ValueError):
        VGBConfig(continuation_length=8, lam=-1.0)
    VGBConfig(continuation_length=8, lam=1.0, temperature=0.0)
    VGBConfig(continuation_length=8, lam=0.0, temperature=0.0)
    VGBConfig(continuation_length=8, backward=False, temperature=0.0)
    assert VGBConfig(continuation_length=8, lam=0.5, momentum_chi=1.0,
                     backward_shortlist=8).label == "vgb_lam0p5_interior_mom1_lb8_T1"


def test_row_chunking_does_not_change_the_walk():
    big, small = Joint(4, 5), Joint(4, 6)
    walks = []
    for chunk in (1, 64):
        config = VGBConfig(continuation_length=4, lam=1.0, momentum_chi=1.0, row_chunk=chunk, trace=True)
        walks.append(run(config, big, small, (MASK,) * 4, (True,) * 4, seed=3))
    assert walks[0].state.tolist() == walks[1].state.tolist()
    assert walks[0].trace == walks[1].trace
    assert walks[0].target_calls > walks[1].target_calls


# ----------------------------------------------------------------------
# Leaf start (editing a draft)
# ----------------------------------------------------------------------


def test_draft_block_is_greedy_confidence_order():
    from sdar_harness.vgb import draft_block

    small = Joint(3, 9)
    state, calls = draft_block(torch.tensor((MASK,) * 3), torch.tensor((True,) * 3),
                               small.scorer, mask_token_id=MASK)
    assert calls == 3 and MASK not in state.tolist()
    # Replay: each commit was the argmax at the most confident position.
    z = [MASK] * 3
    for _ in range(3):
        options = [(max(small.conditional(tuple(z), i)), i) for i in range(3) if z[i] == MASK]
        _, i = max(options)
        z[i] = int(torch.tensor(small.conditional(tuple(z), i)).argmax())
    assert state.tolist() == z


def test_leaf_start_zero_excursions_returns_draft(pair):
    big, small = pair
    config = VGBConfig(continuation_length=3, lam=1.0, start="draft", max_excursions=0)
    walk = run(config, big, small, (1, 0, 1), (True,) * 3, seed=0)
    assert walk.state.tolist() == [1, 0, 1] and walk.steps == 0


@pytest.mark.parametrize("chi", [None, 1.0])
def test_leaf_start_excursions(pair, chi):
    big, small = pair
    config = VGBConfig(continuation_length=3, lam=1.0, start="draft", max_excursions=4,
                       momentum_chi=chi, temperature=0.0, forward_selection="confidence",
                       trace=True)
    for seed in range(40):
        walk = run(config, big, small, (1, 0, 1), (True,) * 3, seed=seed)
        assert MASK not in walk.state.tolist()
        assert 1 <= len(walk.excursion_changes) <= 4
        # Stops early only on an excursion that changed nothing.
        if len(walk.excursion_changes) < 4:
            assert walk.excursion_changes[-1] == 0
        assert all(c > 0 for c in walk.excursion_changes[:-1])
        assert walk.trace[0]["move"] in ("backward", "stay", "switch")
        assert walk.forward_moves == walk.backward_moves


def test_draft_start_rejects_exact_rule():
    with pytest.raises(ValueError):
        VGBConfig(continuation_length=8, lam=1.0, leaf_rule="exact", start="draft")


# ----------------------------------------------------------------------
# Shortlist forward rule (the paper's high_conf, importance-resampled)
# ----------------------------------------------------------------------


def test_shortlist_with_full_coverage_is_ancestral_sampling_of_big(pair):
    """Every position kept and K large: (count/K) pi_b/pi_s -> pi_b, uniform over positions."""
    big, small = pair
    config = VGBConfig(continuation_length=3, backward=False, forward_selection="shortlist",
                       shortlist_positions=3, shortlist_tokens=2000)
    counts: dict[tuple, int] = {}
    trials = 3000
    for seed in range(trials):
        walk = run(config, big, small, (MASK,) * 3, (True,) * 3, seed)
        y = tuple(walk.state.tolist())
        counts[y] = counts.get(y, 0) + 1
    for y, p in big.table.items():
        assert counts.get(y, 0) / trials == pytest.approx(p, abs=4.5 * math.sqrt(p * (1 - p) / trials))


def test_shortlist_single_draw_proposes_from_small(pair):
    """K = 1 over one position: the committed token is a pi_s draw (weight is irrelevant)."""
    big, small = pair
    config = VGBConfig(continuation_length=3, backward=False, forward_selection="shortlist",
                       shortlist_positions=1, shortlist_tokens=1, max_steps_per_block=1)
    start = (1, MASK, 0)
    ones = 0
    trials = 4000
    for seed in range(trials):
        walk = run(config, big, small, start, (True,) * 3, seed)
        ones += walk.state[1].item() == 1
    p = small.conditional(start, 1)[1]
    assert ones / trials == pytest.approx(p, abs=4 * math.sqrt(p * (1 - p) / trials))


def test_shortlist_counts_one_row_per_model(pair):
    big, small = pair
    config = VGBConfig(continuation_length=3, backward=False, forward_selection="shortlist")
    walk = run(config, big, small, (MASK,) * 3, (True,) * 3, seed=1)
    assert walk.target_rows == walk.reference_rows == 3
    assert VGBConfig(continuation_length=3, forward_selection="shortlist").label.endswith("_sl8x8_T1")


@pytest.mark.parametrize("budget", [None, 3])
def test_tracing_does_not_change_the_walk(budget):
    """The trace reads what each move computed and draws nothing, so the moves match."""
    big, small = Joint(4, 5), Joint(4, 6)
    for seed in range(20):
        base = VGBConfig(continuation_length=4, lam=1.0, momentum_chi=1.0, forward_selection="shortlist",
                         shortlist_positions=2, shortlist_tokens=3, max_steps_per_block=budget)
        plain = run(base, big, small, (MASK,) * 4, (True,) * 4, seed)
        traced = run(replace(base, trace=True), big, small, (MASK,) * 4, (True,) * 4, seed)
        assert plain.state.tolist() == traced.state.tolist()
        assert plain.remasked == traced.remasked
        moves = [r for r in traced.trace if r["move"] in ("forward", "backward")]
        assert len(moves) == traced.forward_moves + traced.backward_moves == plain.forward_moves + plain.backward_moves
        assert len(traced.trace) == traced.steps
        for r in moves:
            assert r["top"], r
            if r["move"] == "backward":
                assert r["log_rho"] == pytest.approx(r["top"][0][2]) or r["pick_share"] <= r["top"][0][3]
        if traced.forced_completion:
            assert len(traced.trace[-1]["forced"]) == traced.forced_positions
