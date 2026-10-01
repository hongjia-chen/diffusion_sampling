"""Density-ratio VGB, singleton moves (r = 1), on a block-diffusion model.

Definition 4.2 of Jeon, Vuong & Tao, "VGB for Masked Diffusion Model"
(arXiv:2606.28301), specialised as in `vgb_density_ratio_decoding.tex`:
reference model = SDAR-1.7B (pi_s), terminal reward tau = pi_b / pi_s with
pi_b = SDAR-8B, so the reward-tilted target is pi_b itself and the ideal value
of a partial state is the marginal ratio V(z) = pi_b(z) / pi_s(z).

Reduced transition rule
-----------------------
Divide every outgoing weight at a depth-k state (1 <= k <= n-1) by the common
factor s_{k-1,1} V(z)^(1+lambda). Because (n-k) s_{k,1} = k s_{k-1,1}, the
forward and backward masses become

    F~ = k,        B~ = sum_{i in R(z)} rho_i^lambda,
    rho_i = pi_s(z_i | z^{-i}) / pi_b(z_i | z^{-i}),

so P(forward | z) = 1 / (1 + mean_i rho_i^lambda). Forward moves pick a masked
coordinate uniformly and draw its token from pi_b; backward moves remask i with
probability proportional to rho_i^lambda. No absolute value is needed ...

... except at the leaf boundary. Definition 4.2 drops V(z)^lambda from the
edge into a leaf (that is what makes the leaf law exactly pi_b). At depth n-1
the forward child is a leaf, so F~ = k V(z)^(-lambda) and the absolute value
reappears. `leaf_rule` selects how to handle it:

  "exact"     Definition 4.2 as written. log V(z) is tracked along the walk
              (V = 1 at the block root, exactly, since both models are
              normalised), which needs pi_s at the current state on forward
              moves. Leaf law = pi_b under model consistency.
  "interior"  Use the interior formula at depth n-1 too, which is what the
              official MDM-VGB code does (it gates every forward child by
              V(z)^lambda, leaf or not). Scale-free, no tracking. Leaf law
              becomes pi_b(y) * sum_i V(y^{-i})^lambda, roughly
              pi_b^(1+lambda) / pi_s^lambda: a sequence-level contrastive tilt.

With lambda = 0 the two rules coincide and the reference model drops out
entirely (every rho^0 = 1). See DESIGN.md for the full discussion.

Block structure
---------------
SDAR is causal across blocks, so its conditionals are only in-distribution for
states whose earlier blocks are complete. The walk therefore runs inside one
block at a time (n = the block's editable positions, at most 64), with the
prompt and committed blocks as the fixed context x. Each block's chain starts
at its own fully masked root and stops at its first leaf; the block is then
committed to both models' caches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal

import torch

from sdar_harness.blockwise import BlockCacheProvider
from sdar_harness.model import block_causal_mask
from sdar_harness.scheduler import (
    GenerationResult,
    finished_at,
    stop_token_ids,
    truncate_at_stop,
)
from sdar_harness.timing import NULL_TIMER, StageTimer


# (states [S, L], rows [R], positions [R]) -> log-probs [R, V], float32.
# One forward over S distinct block states; only the R (row, position) pairs
# asked for are projected through the vocabulary head.
Scorer = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

DOWN, UP = "down", "up"


@dataclass(frozen=True)
class VGBConfig:
    continuation_length: int
    # Remasking exponent. 0 = uniform remasking; the reference model is unused.
    lam: float = 1.0
    # Temperature of the forward token draw only (0 = argmax). rho is always
    # computed from both models' untempered (T = 1) conditionals: those are
    # the distributions the density ratio is defined on. T = 1 is Definition
    # 4.2; T = 0 with confidence order reproduces the fixed_k_1 decoder
    # whenever no backward move happens.
    temperature: float = 1.0
    leaf_rule: Literal["exact", "interior"] = "interior"
    # None = plain Definition 4.2. A value in [0, 1] = the momentum lift of
    # Definition 4.3 with that cancellation strength chi.
    momentum_chi: float | None = None
    # None = every leave-one-out parent (exact). An int L_b = the official
    # code's shortlist: L_b uniformly chosen revealed positions, backward mass
    # scaled by k / L_b.
    backward_shortlist: int | None = None
    # Which forward child an unmask move takes. None of these changes the
    # forward/backward split, which always uses the exact forward mass.
    #   "uniform"     Definition 4.2 exactly: uniform masked position, token
    #                 drawn from pi_b (the closed form of pi_s * pi_b/pi_s).
    #   "confidence"  SDAR's decoder: the most confident pi_b draw.
    #   "shortlist"   The paper's `high_conf` shortlist (official code,
    #                 build_mdm_forward_candidates): the `shortlist_positions`
    #                 masked positions where pi_s is most confident, K =
    #                 `shortlist_tokens` draws from pi_s at each, and one
    #                 (position, token) picked with weight
    #                 (count/K) * pi_b(a)/pi_s(a): importance resampling of
    #                 1.7B proposals toward 8B. See DESIGN.md §8.
    forward_selection: Literal["uniform", "confidence", "shortlist"] = "uniform"
    shortlist_positions: int = 8
    shortlist_tokens: int = 8
    # False disables backward moves: forward-only random-order ancestral
    # sampling from pi_b, the no-remasking control.
    backward: bool = True
    # Moves (forward + backward) allowed per block before the rest of the
    # block is completed forward-only from pi_b (the paper's fallback).
    # Momentum stay/switch steps reuse the cached rho and cost no model call,
    # so they do not count. None = no limit.
    max_steps_per_block: int | None = 1024
    # Parent rows per model forward. Bounds activation memory, not the math.
    row_chunk: int = 32
    early_stop: bool = True
    # Keep a per-step trace (k, direction, move, log masses) for analysis.
    trace: bool = False
    # "root": every block's chain starts fully masked (generation).
    # "draft": the reference model first drafts the block with its greedy
    # fixed_k_1 decoder, and the chain starts from that complete block (the
    # paper's leaf-start editing). It then runs excursions -- leave the leaf,
    # return to a leaf -- until one returns the block unchanged or
    # `max_excursions` is reached. max_excursions = 0 is the plain draft.
    start: Literal["root", "draft"] = "root"
    max_excursions: int = 4

    def __post_init__(self) -> None:
        if self.continuation_length < 1:
            raise ValueError("continuation_length must be positive")
        if not math.isfinite(self.lam) or self.lam < 0:
            raise ValueError("lam must be finite and >= 0")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.leaf_rule not in ("exact", "interior"):
            raise ValueError(f"unknown leaf_rule {self.leaf_rule!r}")
        if self.momentum_chi is not None and not 0 <= self.momentum_chi <= 1:
            raise ValueError("momentum_chi must lie in [0, 1]")
        if self.backward_shortlist is not None and self.backward_shortlist < 1:
            raise ValueError("backward_shortlist must be positive")
        if self.forward_selection not in ("uniform", "confidence", "shortlist"):
            raise ValueError(f"unknown forward_selection {self.forward_selection!r}")
        if self.shortlist_positions < 1 or self.shortlist_tokens < 1:
            raise ValueError("shortlist sizes must be positive")
        if self.max_steps_per_block is not None and self.max_steps_per_block < 1:
            raise ValueError("max_steps_per_block must be positive")
        if self.row_chunk < 1:
            raise ValueError("row_chunk must be positive")
        if self.start not in ("root", "draft"):
            raise ValueError(f"unknown start {self.start!r}")
        if self.start == "draft" and self.tracks_value:
            # log V at the draft leaf would need 8B scores along the draft path.
            raise ValueError("draft start supports only the interior leaf rule")
        if self.max_excursions < 0:
            raise ValueError("max_excursions must be >= 0")

    @property
    def method(self) -> str:
        return "vgb"

    @property
    def needs_reference(self) -> bool:
        return self.needs_rho or self.start == "draft" or self.forward_selection == "shortlist"

    @property
    def needs_rho(self) -> bool:
        return self.backward and self.lam > 0

    @property
    def tracks_value(self) -> bool:
        return self.needs_rho and self.leaf_rule == "exact"

    @property
    def label(self) -> str:
        if not self.backward:
            name = "vgb_fwdonly"
        else:
            lam = f"{self.lam:g}".replace(".", "p")
            name = f"vgb_lam{lam}_{self.leaf_rule}"
            if self.momentum_chi is not None:
                name += f"_mom{self.momentum_chi:g}".replace(".", "p")
            if self.backward_shortlist is not None:
                name += f"_lb{self.backward_shortlist}"
        if self.forward_selection == "confidence":
            name += "_conf"
        elif self.forward_selection == "shortlist":
            name += f"_sl{self.shortlist_positions}x{self.shortlist_tokens}"
        if self.start == "draft":
            name = name.replace("vgb_", "vgbedit_", 1) + f"_exc{self.max_excursions}"
        return name + f"_T{self.temperature:g}".replace(".", "p")


def parse_vgb_spec(spec: str, *, continuation_length: int, budget: int | None = 512,
                   **overrides: Any) -> VGBConfig:
    """Build a config from a colon-separated spec (the probe and run_benchmark syntax).

    First field: `fwdonly` (no backward moves) or `lam<λ>`. Flags: `mom`
    (momentum, χ = 1), `exact` (exact leaf rule), `T<temp>`, `conf`
    (confidence order), `sl<L_f>x<K>` (shortlist), `draft` (1.7B draft,
    leaf-start editing), `exc<E>` (max excursions), `lb<L_b>` (backward
    shortlist). Examples: `lam0.5:mom:sl8x8`, `lam1:mom:T0:conf:draft:exc4`.
    """
    parts = spec.split(":")
    config = VGBConfig(continuation_length=continuation_length, max_steps_per_block=budget,
                       **overrides)
    if parts[0] == "fwdonly":
        config = replace(config, backward=False)
    elif parts[0].startswith("lam"):
        config = replace(config, lam=float(parts[0][3:]))
    else:
        raise ValueError(f"VGB spec must start with fwdonly or lam<λ>: {spec!r}")
    for flag in parts[1:]:
        if flag == "mom":
            config = replace(config, momentum_chi=1.0)
        elif flag == "exact":
            config = replace(config, leaf_rule="exact")
        elif flag == "conf":
            config = replace(config, forward_selection="confidence")
        elif flag == "draft":
            config = replace(config, start="draft")
        elif flag.startswith("T"):
            config = replace(config, temperature=float(flag[1:]))
        elif flag.startswith("sl"):
            positions, tokens = flag[2:].split("x")
            config = replace(config, forward_selection="shortlist",
                             shortlist_positions=int(positions), shortlist_tokens=int(tokens))
        elif flag.startswith("exc"):
            config = replace(config, max_excursions=int(flag[3:]))
        elif flag.startswith("lb"):
            config = replace(config, backward_shortlist=int(flag[2:]))
        else:
            raise ValueError(f"unknown VGB flag {flag!r} in {spec!r}")
    return config


# ----------------------------------------------------------------------
# The transition rule. Pure functions of scalars and small vectors, so they
# can be checked against exact toy distributions (tests/test_vgb.py).
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LocalMoves:
    """Log forward/backward masses at one state, in a common (arbitrary) unit.

    `backward_logits` is over the candidate parents, in the order given; its
    softmax is P(parent | backward move).
    """

    log_forward: float
    log_backward: float
    backward_logits: torch.Tensor

    @property
    def forward_probability(self) -> float:
        if self.log_forward == -math.inf:
            return 0.0
        if self.log_backward == -math.inf:
            return 1.0
        return 1.0 / (1.0 + math.exp(self.log_backward - self.log_forward))


def local_moves(
    k: int,
    n: int,
    log_rho: torch.Tensor | None,
    *,
    lam: float,
    leaf_rule: str = "interior",
    log_value: float = 0.0,
    shortlist_scale: float = 1.0,
    backward: bool = True,
) -> LocalMoves:
    """Reduced Definition 4.2 masses at a depth-k state of an n-position walk.

    `log_rho` holds log rho_i for the candidate parents (all revealed positions,
    or a shortlist whose mass is scaled by `shortlist_scale` = k / |shortlist|).
    It may be None when lam = 0. `log_value` is log V(z); only the exact leaf
    rule reads it. Leaves (k = n) are included for the stationary-law tests;
    the sampler itself stops before taking a move from one.
    """
    if not 0 <= k <= n:
        raise ValueError(f"depth {k} outside [0, {n}]")
    candidates = k if log_rho is None else int(log_rho.numel())
    if k == 0 or not backward or candidates == 0:
        empty = torch.empty(0, dtype=torch.float64)
        return LocalMoves(0.0 if k < n else -math.inf, -math.inf, empty)
    if log_rho is None:
        if lam != 0:
            raise ValueError("log_rho is required when lam > 0")
        logits = torch.zeros(k, dtype=torch.float64)
    else:
        logits = lam * log_rho.to(torch.float64)

    if k == n:
        # Leaf. Definition 4.2's leaf backward weight s_{n-1,1} V(y) carries no
        # lambda: uniform. The interior rule keeps V(y^{-i})^lambda.
        if leaf_rule == "exact":
            logits = torch.zeros_like(logits)
        return LocalMoves(-math.inf, 0.0, logits)

    log_backward = float(torch.logsumexp(logits, dim=0)) + math.log(shortlist_scale)
    log_forward = math.log(k)
    if k == n - 1 and leaf_rule == "exact":
        # Forward child is a leaf: F = s_{n-1,1} V(z) against the interior
        # backward mass s_{n-2,1} V(z)^(1+lambda) sum rho^lambda.
        log_forward -= lam * log_value
    return LocalMoves(log_forward, log_backward, logits)


@dataclass(frozen=True)
class DirectionChoice:
    move: Literal["forward", "backward", "stay", "switch"]
    direction: str


def choose_direction(
    moves: LocalMoves,
    direction: str,
    chi: float | None,
    uniform: float,
) -> DirectionChoice:
    """Plain chain (chi None) or the Definition 4.3 momentum lift.

    `uniform` is one U(0,1) draw, so the caller owns the randomness.
    """
    p_forward = moves.forward_probability
    p_backward = 1.0 - p_forward
    if chi is None:
        move = "forward" if uniform < p_forward else "backward"
        return DirectionChoice(move, DOWN if move == "forward" else UP)
    cancelled = chi * min(p_forward, p_backward)
    along, against = (p_forward, p_backward) if direction == DOWN else (p_backward, p_forward)
    if uniform < along:
        return DirectionChoice("forward" if direction == DOWN else "backward", direction)
    if uniform < along + cancelled:
        return DirectionChoice("stay", direction)
    return DirectionChoice("switch", UP if direction == DOWN else DOWN)


def tempered_log_probs(
    logits: torch.Tensor, excluded: set[int], temperature: float
) -> torch.Tensor:
    """log-softmax over the allowed vocabulary, at `temperature` (0 = keep raw)."""
    scores = logits.to(torch.float32, copy=True)
    bad = sorted(index for index in excluded if 0 <= index < scores.shape[-1])
    if bad:
        scores.index_fill_(-1, torch.tensor(bad, device=scores.device), -torch.inf)
    if temperature > 0:
        scores = scores / temperature
    return torch.log_softmax(scores, dim=-1)


# ----------------------------------------------------------------------
# One block's walk
# ----------------------------------------------------------------------


@dataclass
class BlockWalk:
    state: torch.Tensor
    steps: int = 0
    forward_moves: int = 0
    backward_moves: int = 0
    stays: int = 0
    switches: int = 0
    target_calls: int = 0
    target_rows: int = 0
    reference_calls: int = 0
    reference_rows: int = 0
    forced_completion: bool = False
    forced_positions: int = 0
    # (position, token) of every remask, in order: what the chain undid.
    remasked: list[tuple[int, int]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    # Leaf-start only: positions changed by each completed excursion.
    excursion_changes: list[int] = field(default_factory=list)


def _score_chunked(
    scorer: Scorer, states: torch.Tensor, rows: torch.Tensor, positions: torch.Tensor, chunk: int
) -> tuple[torch.Tensor, int]:
    """Scorer over row chunks. Returns (log-probs, forward calls)."""
    if states.shape[0] <= chunk:
        return scorer(states, rows, positions), 1
    outputs, calls = [], 0
    for low in range(0, states.shape[0], chunk):
        high = low + chunk
        pick = (rows >= low) & (rows < high)
        outputs.append((pick, scorer(states[low:high], rows[pick] - low, positions[pick])))
        calls += 1
    result = None
    for pick, values in outputs:
        if result is None:
            result = values.new_empty((rows.shape[0], values.shape[-1]))
        result[pick] = values
    return result, calls


def _sample(log_probs: torch.Tensor, temperature: float, generator: torch.Generator) -> torch.Tensor:
    """Draw from untempered log-probs at `temperature` (0 = argmax)."""
    if temperature == 0:
        return log_probs.argmax(dim=-1)
    probs = torch.softmax(log_probs / temperature, dim=-1) if temperature != 1 else log_probs.exp()
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


def walk_block(
    state: torch.Tensor,
    editable: torch.Tensor,
    target: Scorer,
    reference: Scorer | None,
    config: VGBConfig,
    *,
    mask_token_id: int,
    generator: torch.Generator,
) -> BlockWalk:
    """Run the chain on one block from `state` to its first leaf (or budget).

    `state` is [L] and holds the block (prompt tail included); `editable` is
    [L] bool and marks the walk's n coordinates. Both scorers see [S, L] states.
    """
    state = state.clone()
    device = state.device
    n = int(editable.sum())
    walk = BlockWalk(state=state)
    leaf_start = not bool((editable & (state == mask_token_id)).any())
    # From a leaf the only move is a remask; start the lift pointing up.
    direction = UP if leaf_start else DOWN
    snapshot = state.clone()
    excursion_from = 0  # moves made when the current excursion began
    log_value = 0.0
    cached: tuple[LocalMoves, torch.Tensor] | None = None  # valid while state is unchanged
    if config.needs_reference and reference is None:
        raise ValueError(f"{config.label} needs a reference scorer")
    # Trace-only detail of the last unmask (the shortlist it chose from). Filled
    # from tensors the move already computed; it draws no random numbers, so a
    # traced walk takes exactly the moves of an untraced one.
    forward_info: dict[str, Any] = {}

    def forward_move(masked_positions: torch.Tensor):
        """One unmask. Returns (position, token, log pi_b(token), log pi_s(token) or None)."""
        forward_info.clear()
        rows = torch.zeros(len(masked_positions), dtype=torch.long, device=device)
        if config.forward_selection == "shortlist":
            small_all = reference(state.unsqueeze(0), rows, masked_positions)
            walk.reference_calls += 1
            walk.reference_rows += 1
            keep = small_all.max(dim=-1).values.topk(
                min(config.shortlist_positions, len(masked_positions))).indices
            positions, small = masked_positions[keep], small_all[keep]
            big = target(state.unsqueeze(0), rows[: len(keep)], positions)
            walk.target_calls += 1
            walk.target_rows += 1
            if config.temperature == 0:
                draws = small.argmax(dim=-1, keepdim=True)
            else:
                draws = torch.multinomial(torch.softmax(small / config.temperature, dim=-1),
                                          config.shortlist_tokens, replacement=True,
                                          generator=generator)
            slot = torch.arange(len(keep), device=draws.device)[:, None].expand_as(draws)
            pairs, counts = torch.unique(
                torch.stack((slot.reshape(-1), draws.reshape(-1)), dim=1), dim=0, return_counts=True)
            slots, tokens = pairs[:, 0], pairs[:, 1]
            log_weights = (counts.float() / draws.shape[1]).log() + big[slots, tokens] - small[slots, tokens]
            pick = int(torch.multinomial(torch.softmax(log_weights, 0), 1, generator=generator))
            slot_, token = int(slots[pick]), tokens[pick]
            if config.trace:
                share = torch.softmax(log_weights.double(), 0)
                top = share.topk(min(6, len(share))).indices.tolist()
                forward_info.update(candidates=len(share), pick_share=float(share[pick]), top=[
                    [int(positions[int(slots[j])]), int(tokens[j]), float(big[int(slots[j]), tokens[j]]),
                     float(small[int(slots[j]), tokens[j]]), int(counts[j]), float(share[j])]
                    for j in top])
            return positions[slot_], token, float(big[slot_, token]), float(small[slot_, token])
        if config.forward_selection == "uniform":
            pick = int(torch.randint(len(masked_positions), (), generator=generator, device=generator.device))
            masked_positions, rows = masked_positions[pick : pick + 1], rows[:1]
        big = target(state.unsqueeze(0), rows, masked_positions)
        walk.target_calls += 1
        walk.target_rows += 1
        drawn = _sample(big, config.temperature, generator)
        best = int(big.gather(-1, drawn[:, None]).squeeze(-1).argmax()) \
            if config.forward_selection == "confidence" else 0
        return masked_positions[best], drawn[best], float(big[best, drawn[best]]), None

    def rand() -> float:
        return float(torch.rand((), generator=generator, device=generator.device))

    while True:
        masked = editable & (state == mask_token_id)
        revealed = editable & ~masked
        k = n - int(masked.sum())
        moves_made = walk.forward_moves + walk.backward_moves
        if k == n:
            if not leaf_start or not config.backward:
                break
            if moves_made > excursion_from:  # an excursion just returned
                walk.excursion_changes.append(int((state != snapshot).sum()))
                if walk.excursion_changes[-1] == 0:
                    break
                snapshot = state.clone()
                excursion_from = moves_made
            if len(walk.excursion_changes) >= config.max_excursions:
                break
            if config.max_steps_per_block is not None and moves_made >= config.max_steps_per_block:
                break  # out of budget, but at a leaf: nothing to complete
        if config.max_steps_per_block is not None and moves_made >= config.max_steps_per_block:
            walk.forced_completion = True
            break

        masked_positions = masked.nonzero().squeeze(-1)

        if cached is None:
            parents = revealed.nonzero().squeeze(-1)
            scale = 1.0
            # Without rho every parent costs nothing, so shortlisting only
            # adds noise; it applies when rho has to be computed.
            if (config.needs_rho and config.backward_shortlist is not None
                    and len(parents) > config.backward_shortlist):
                order = torch.randperm(len(parents), generator=generator, device=generator.device)
                parents = parents[order[: config.backward_shortlist].to(parents.device)].sort().values
                scale = k / len(parents)
            log_rho = torch.zeros(len(parents), dtype=torch.float64)  # rho^0 = 1
            if config.needs_rho and len(parents):
                parent_states = state.unsqueeze(0).repeat(len(parents), 1)
                parent_states[torch.arange(len(parents), device=device), parents] = mask_token_id
                tokens = state[parents]
                rows = torch.arange(len(parents), device=device)
                big, calls = _score_chunked(target, parent_states, rows, parents, config.row_chunk)
                walk.target_calls += calls
                walk.target_rows += len(parents)
                small, calls = _score_chunked(reference, parent_states, rows, parents, config.row_chunk)
                walk.reference_calls += calls
                walk.reference_rows += len(parents)
                log_rho = (small.gather(-1, tokens[:, None]) - big.gather(-1, tokens[:, None])).squeeze(-1)
                log_rho = log_rho.double().cpu()
            moves = local_moves(
                k, n, log_rho,
                lam=config.lam if config.needs_rho else 0.0,
                leaf_rule=config.leaf_rule, log_value=log_value,
                shortlist_scale=scale, backward=config.backward,
            )
            cached = (moves, parents)
        moves, parents = cached

        choice = choose_direction(moves, direction, config.momentum_chi, rand())
        direction = choice.direction
        walk.steps += 1
        record: dict[str, Any] | None = None
        if config.trace:
            record = {
                "k": k, "move": choice.move, "direction": direction,
                "log_forward": moves.log_forward, "log_backward": moves.log_backward,
                "log_value": log_value if config.tracks_value else None,
            }

        if choice.move == "stay":
            walk.stays += 1
        elif choice.move == "switch":
            walk.switches += 1
        elif choice.move == "forward":
            position, token, log_big, log_small = forward_move(masked_positions)
            if config.tracks_value:
                if log_small is None:
                    small = reference(state.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=device),
                                      position.view(1))
                    walk.reference_calls += 1
                    walk.reference_rows += 1
                    log_small = float(small[0, token])
                log_value += log_big - log_small
            state[position] = token
            walk.forward_moves += 1
            cached = None
            if record is not None:
                record.update(position=int(position), token=int(token), log_big=log_big,
                              log_small=log_small, **forward_info)
        else:
            weights = torch.softmax(moves.backward_logits, 0).to(generator.device)
            index = int(torch.multinomial(weights, 1, generator=generator))
            position = parents[index]
            if config.tracks_value:
                log_value += float(moves.backward_logits[index]) / config.lam
            walk.remasked.append((int(position), int(state[position])))
            if record is not None:
                # Why this token: its log rho = log pi_s - log pi_b with itself
                # masked, and the most likely other remasks, (position, token,
                # log rho, pick probability). Read before the token is masked.
                share = torch.softmax(moves.backward_logits, 0)
                lam = config.lam if config.needs_rho and config.lam > 0 else 1.0
                top = share.topk(min(6, len(share))).indices.tolist()
                record.update(
                    position=int(position), token=int(state[position]),
                    log_rho=float(moves.backward_logits[index]) / lam, pick_share=float(share[index]),
                    parents=len(parents),
                    top=[[int(parents[j]), int(state[parents[j]]), float(moves.backward_logits[j]) / lam,
                          float(share[j])] for j in top])
            state[position] = mask_token_id
            walk.backward_moves += 1
            cached = None
        if record is not None:
            walk.trace.append(record)

    if walk.forced_completion:
        # The paper's fallback: finish forward-only from the target.
        while True:
            masked_positions = (editable & (state == mask_token_id)).nonzero().squeeze(-1)
            if not len(masked_positions):
                break
            position, token, _, _ = forward_move(masked_positions)
            state[position] = token
            walk.forced_positions += 1
            # Forced fills are not walk steps, and the runner checks one trace
            # row per step, so they ride on the block's last row.
            if config.trace and walk.trace:
                walk.trace[-1].setdefault("forced", []).append([int(position), int(token)])
    walk.state = state
    return walk


def draft_block(
    state: torch.Tensor,
    editable: torch.Tensor,
    scorer: Scorer,
    *,
    mask_token_id: int,
) -> tuple[torch.Tensor, int]:
    """Greedy fixed_k_1 fill: argmax token at the most confident masked position.

    The SDAR fixed_k_1 decoder, one committed token per forward. Returns the
    filled block and the number of forwards.
    """
    state = state.clone()
    calls = 0
    while True:
        masked = (editable & (state == mask_token_id)).nonzero().squeeze(-1)
        if not len(masked):
            return state, calls
        rows = torch.zeros(len(masked), dtype=torch.long, device=state.device)
        log_probs = scorer(state.unsqueeze(0), rows, masked)
        calls += 1
        best, tokens = log_probs.max(dim=-1)
        pick = int(best.argmax())
        state[masked[pick]] = tokens[pick]


# ----------------------------------------------------------------------
# Block-diffusion driver (two models, two caches)
# ----------------------------------------------------------------------


def block_scorer(scope: BlockCacheProvider, excluded: set[int]) -> Scorer:
    """Untempered log-probs; temperature is applied only when drawing."""

    def score(states: torch.Tensor, rows: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden = scope.hidden_states(states)
        logits = scope.logits_from_hidden(hidden, rows.to(hidden.device), positions.to(hidden.device))
        return tempered_log_probs(logits, excluded, 1.0).to(states.device)

    return score


def generate_vgb(
    target: Any,
    reference: Any | None,
    prompt_ids: torch.Tensor,
    config: VGBConfig,
    *,
    generator: torch.Generator | None = None,
    timer: StageTimer | None = None,
) -> GenerationResult:
    """Blockwise density-ratio VGB with SDAR-8B (target) and SDAR-1.7B (reference).

    Mirrors `generate_blockwise`: same canvas rounding, same block-causal
    mask, same store_kv discipline. Every finished block is committed to both
    models' caches. Early stop happens only at a block's leaf -- inside the
    walk a stop token can still be remasked.
    """
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("one prompt at a time")
    timer = timer if timer is not None else NULL_TIMER
    block = target.conventions.block_length
    if block is None:
        raise ValueError("generate_vgb needs a block-diffusion target")
    models = [target] if reference is None else [target, reference]
    for other in models[1:]:
        if other.conventions.block_length != block or other.mask_token_id != target.mask_token_id:
            raise ValueError("target and reference must share block length and mask token")
    device = target.device
    generator = generator if generator is not None else torch.Generator(device=device)

    prompt_length = int(prompt_ids.shape[1])
    generated = config.continuation_length
    num_blocks = math.ceil((prompt_length + generated) / block)
    total_length = num_blocks * block
    mask_id = target.mask_token_id
    canvas = torch.full((total_length,), mask_id, dtype=torch.long, device=device)
    canvas[:prompt_length] = prompt_ids[0].to(device)
    attention = block_causal_mask(total_length, block, device=device)
    positions = torch.arange(total_length, device=device)

    from transformers.cache_utils import DynamicCache

    caches = [DynamicCache() for _ in models]
    prefill = (prompt_length // block) * block
    commit_passes = 0

    def commit(low: int, high: int) -> None:
        nonlocal commit_passes
        with timer("commit"):
            for model, cache in zip(models, caches):
                model.commit(
                    canvas[low:high].unsqueeze(0).to(model.device),
                    attention[low:high, :high].to(model.device),
                    positions[low:high].to(model.device),
                    cache,
                )
                commit_passes += 1

    if prefill:
        commit(0, prefill)

    stop_ids = stop_token_ids(target)
    totals: dict[str, Any] = {
        "steps": 0, "forward_moves": 0, "backward_moves": 0, "stays": 0, "switches": 0,
        "target_calls": 0, "target_rows": 0, "reference_calls": 0, "reference_rows": 0,
        "forced_blocks": 0, "forced_positions": 0, "draft_calls": 0,
        "changed_vs_draft": 0, "excursions": 0,
    }
    per_block: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    early_stopped = False

    for index in range(prefill // block, num_blocks):
        low, high = index * block, (index + 1) * block
        editable = canvas[low:high] == mask_id
        if not bool(editable.any()):
            continue
        scorers = []
        for model, cache in zip(models, caches):
            scope = BlockCacheProvider(
                model, cache,
                attention[low:high, :high].to(model.device),
                positions[low:high].to(model.device),
            )
            scorers.append(block_scorer(scope, model.excluded_token_ids))
        start_state = canvas[low:high]
        draft_calls = 0
        draft = None
        if config.start == "draft":
            if len(scorers) < 2:
                raise ValueError("draft start needs the reference model")
            with timer("draft"):
                draft, draft_calls = draft_block(
                    canvas[low:high], editable, scorers[1], mask_token_id=mask_id)
            start_state = draft
        with timer("walk"):
            walk = walk_block(
                start_state, editable, scorers[0],
                scorers[1] if len(scorers) > 1 else None, config,
                mask_token_id=mask_id, generator=generator,
            )
        canvas[low:high] = walk.state
        summary = {
            key: getattr(walk, key)
            for key in ("steps", "forward_moves", "backward_moves", "stays", "switches",
                        "target_calls", "target_rows", "reference_calls", "reference_rows",
                        "forced_positions")
        }
        summary.update(block_index=index, n=int(editable.sum()),
                       forced=walk.forced_completion, remasked=walk.remasked,
                       draft_calls=draft_calls, excursions=walk.excursion_changes,
                       changed_vs_draft=(int((walk.state != draft).sum())
                                         if draft is not None else None))
        per_block.append(summary)
        for key in totals:
            if key == "forced_blocks":
                totals[key] += int(walk.forced_completion)
            elif key == "excursions":
                totals[key] += len(walk.excursion_changes)
            elif key == "changed_vs_draft":
                totals[key] += summary[key] or 0
            elif key in summary:
                totals[key] += summary[key]
        if config.trace:
            traces.extend(dict(record, block_index=index) for record in walk.trace)

        if config.early_stop and finished_at(
            canvas[prompt_length : prompt_length + generated],
            mask_token_id=mask_id, stop_ids=stop_ids,
        ) is not None:
            early_stopped = True
            break
        commit(low, high)

    continuation = canvas[prompt_length : prompt_length + generated].cpu().tolist()
    scored, eos_position, stop_token = truncate_at_stop(continuation, stop_ids)
    return GenerationResult(
        continuation_token_ids=[int(value) for value in scored],
        traces=traces,
        denoising_steps=totals["steps"],
        proposer_passes=totals["target_calls"],
        # 1.7B work: leave-one-out/shortlist rows plus the draft's own forwards.
        verifier_rows=totals["reference_rows"] + totals["draft_calls"],
        verifier_forward_passes=totals["reference_calls"] + totals["draft_calls"],
        # One token per unmask move (r = 1), including forced completions.
        commitment_histogram=[totals["forward_moves"] + totals["forced_positions"]],
        eos_position=eos_position,
        maximum_length_termination=eos_position is None,
        full_continuation_token_ids=[int(value) for value in continuation],
        stop_token_id=stop_token,
        early_stopped=early_stopped and mask_id in continuation,
        decoded_positions=sum(1 for value in continuation if value != mask_id),
        extras={"vgb": totals, "vgb_blocks": per_block, "vgb_config": config.label},
        commit_passes=commit_passes,
        stage_seconds=timer.totals(),
    )
