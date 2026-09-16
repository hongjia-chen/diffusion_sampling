"""Algorithm 3 of "Parallel Sampling via Autospeculation" (arXiv:2511.07869).

The algorithm, verbatim (paper p. 5)
-----------------------------------

    sigma <- uniformly random permutation of [n]      # ONCE, at the top
    rho   <- 1 / log2(n)
    Z     <- RS2([n], Z_<S = empty)

    function RS2(S = {a+1..b}, Z_<S):
        X ~ nu_S(. | Z_<S) := prod_{r=a+1}^{b} pi_{sigma(r)}(. | Z_<S)
        dmu/dnu (X | Z_<S) = prod_{s=a+1}^{b}
              pi_{sigma(s)}( X_{sigma(s)} | Z_<S, X_{sigma(a+1)}..X_{sigma(s-1)} )
            / pi_{sigma(s)}( X_{sigma(s)} | Z_<S )
        u <- min(1, dmu/dnu(X | Z_<S))
        if coin(u):
            return X                       # if b <= a+1 then u = 1 always
        else:
            for r = 0, 1, 2, ...:
                for i = ceil((1+rho)^r) .. ceil((1+rho)^(r+1)) - 1 IN PARALLEL:
                    Y(i) <- FallbackSampler(S, Z_<S)
                    p(i) <- max(0, 1 - dnu/dmu(Y(i) | Z_<S))
                    c(i) <- coin(p(i))
                if any c(i) is heads:
                    return Y(i*) for i* the first head

    function FallbackSampler(S = {a+1..b}, Z_<S):
        T1 <- {a+1 .. floor((a+b)/2)},  T2 <- {floor((a+b)/2)+1 .. b}
        for i in 1, 2:                     # SEQUENTIAL, this is the whole point
            Z_{T_i} <- RS2(T_i, Z);  append to Z
        return Z_S

In masked-diffusion terms
-------------------------

`nu_S` is the product of per-position marginals over the block, which is
exactly **one fully parallel denoising step that unmasks all of S at once** --
what `fixed:|S|` does. `mu_S` is the chain rule over S in sigma order, which is
**|S| steps unmasking one position each** -- `fixed:1` restricted to S. So
Algorithm 3 is: try the all-at-once unmasking, accept it with exactly the
probability that makes the result distributed as if you had gone one at a time,
and on rejection split the block and recurse.

Blocks are contiguous ranges of *sigma indices*, so their canvas positions are
scattered. sigma is drawn once and never redrawn: `dmu_S/dnu_S` telescopes only
because the order within a block is fixed.

Design decisions are recorded in ../DESIGN.md. The ones this file implements:

    D1  temperature 1.0, no top-p -- proposer and verifier score the same
        distribution, which is what makes the coin meaningful.
    D2  sigma uniformly random, fixed up front, independent of the sample.
    D3  lockstep batching: the rejection loop's members run as rows of one
        pooled forward, via `_interleave`, not one after another.
    D4  physical rounds are the headline, and the only round counter kept.
    D5  canvas sized so the root block is one physical round where possible.

Parallelism
-----------

Every function that needs the model is a generator that *yields* a forward
request and receives logits back. `_interleave` merges several such generators
so their requests pool into one, recursively, and `_drive` is the only place
that touches the provider. One `_drive` iteration is one parallel round no
matter how many block evaluations it contains; how many forwards that actually
costs is counted separately.

Why the rejection loop needs no extra forwards
----------------------------------------------

`RS2` returns `log mu_S(Z_S | Z_<S)` alongside the block it filled, and
`FallbackSampler` composes the two halves by the chain rule. So when the
rejection loop needs `dnu/dmu(Y(i))`, the `mu` half is already known and the
`nu` half is a lookup into the block's cached marginals. Evaluating a candidate
is free; only *drawing* it costs anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Generator, Literal

import torch

from autospec.scheduler import GenerationResult, LogitsProvider, _filtered_log_probs, _uses_lean_path
from autospec.timing import NULL_TIMER, StageTimer


# A forward request: (states [k, S], rows [N], positions [N]). Forward the k
# state rows, then return logits at (rows[j], positions[j]) as [N, V]. Keeping
# reads separate from rows is what lets a block read m positions out of a
# single forward instead of forwarding m copies of the same state.
Request = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Reply = torch.Tensor


@dataclass(frozen=True)
class AutospecConfig:
    continuation_length: int
    # Paper default: rho = 1 / log2(n). Exposed because it is the one knob the
    # paper leaves free (DESIGN.md D6).
    rho: float | None = None
    # Physical batching budget: the most state rows in one forward. This does
    # not change any result, only how many physical rounds a parallel one is
    # chunked into.
    max_batch_rows: int = 256
    temperature: float = 1.0
    top_p: float | None = None
    method: Literal["autospec"] = "autospec"
    order: Literal["sigma"] = "sigma"

    def __post_init__(self) -> None:
        if self.continuation_length < 1:
            raise ValueError("continuation_length must be positive")
        if self.temperature != 1.0 or self.top_p is not None:
            # Not a correctness requirement -- Algorithm 3 stays exact at any
            # fixed T > 0 -- but at T != 1 the target is an order-dependent
            # object with no independent characterization, and the O(sqrt(n))
            # round bound no longer applies. See DESIGN.md D1.
            raise ValueError(
                "autospec runs at temperature 1.0 with no top-p; see DESIGN.md D1"
            )
        if self.rho is not None and not self.rho > 0:
            raise ValueError("rho must be positive")
        if self.max_batch_rows < 1:
            raise ValueError("max_batch_rows must be positive")

    @property
    def effective_rho(self) -> float:
        if self.rho is not None:
            return self.rho
        return 1.0 / max(math.log2(max(self.continuation_length, 2)), 1e-9)

    # A block may commit up to the whole canvas, so this is the histogram width.
    @property
    def max_candidates(self) -> int:
        return self.continuation_length

    @property
    def verifier_microbatch_size(self) -> int:
        return self.max_batch_rows

    @property
    def label(self) -> str:
        # An explicit rho is used verbatim, so it can be named. The paper's
        # default cannot: it is 1/log2(n), and under block diffusion the n
        # Algorithm 3 actually runs on is the BLOCK, not the canvas -- 1/log2(64)
        # where a canvas-derived name would claim 1/log2(512). Naming the arm
        # after a value no block uses is worse than not naming it, and the last
        # block is shorter than the rest, so no single number is right either.
        if self.rho is None:
            return "autospec_rho_auto"
        value = f"{self.rho:.4f}".replace(".", "p")
        return f"autospec_rho_{value}"


@dataclass(frozen=True)
class AutospecResult(GenerationResult):
    """`GenerationResult` plus the counters D4 asks for."""

    oracle_rounds_physical: int = 0
    oracle_rows: int = 0
    blocks_opened: int = 0
    speculations_accepted: int = 0
    speculations_rejected: int = 0
    fallback_draws: int = 0
    max_fallback_draws_in_call: int = 0
    max_pooled_request_rows: int = 0
    singleton_blocks: int = 0


class _Context:
    """Everything the recursion needs that is not a block."""

    def __init__(self, provider: LogitsProvider, config: AutospecConfig,
                 sigma: torch.Tensor, canvas_start: int, device: torch.device,
                 timer: Any = None):
        self.timer = timer if timer is not None else NULL_TIMER
        self.provider = provider
        self.config = config
        self.sigma = sigma                 # sigma[j] = canvas position visited j-th
        self.canvas_start = canvas_start
        self.device = device
        self.excluded = provider.excluded_token_ids
        self.rho = config.effective_rho
        self.commitments: dict[int, int] = {}
        self.traces: list[dict[str, Any]] = []
        self.counters = {
            "oracle_rounds_physical": 0,
            "oracle_rows": 0,
            "blocks_opened": 0,
            "speculations_accepted": 0,
            "speculations_rejected": 0,
            "fallback_draws": 0,
            "max_fallback_draws_in_call": 0,
            "max_pooled_request_rows": 0,
            "singleton_blocks": 0,
        }

    def positions(self, lo: int, hi: int) -> torch.Tensor:
        """Canvas-absolute positions of sigma indices [lo, hi)."""
        return self.sigma[lo:hi] + self.canvas_start

    def commit(self, width: int) -> None:
        self.commitments[width] = self.commitments.get(width, 0) + 1


def _coin(bias: torch.Tensor) -> bool:
    """One Bernoulli draw, taken on the global RNG so runs stay seed-reproducible."""
    return bool(torch.rand((), device=bias.device, dtype=bias.dtype) < bias)


def _interleave(
    subgenerators: list[Generator[Request, Reply, Any]],
) -> Generator[Request, Reply, list[Any]]:
    """Run sub-generators in lockstep, pooling their requests into one (D3a).

    Members diverge as soon as their coins differ, so the pool shrinks over
    time and the finished members simply stop contributing rows. Nothing is
    masked or padded: a member that is done is absent from the next pool.
    """
    results: list[Any] = [None] * len(subgenerators)
    pending: dict[int, Request] = {}
    for index, generator in enumerate(subgenerators):
        try:
            pending[index] = generator.send(None)
        except StopIteration as stop:
            results[index] = stop.value
    while pending:
        keys = sorted(pending)
        states_parts, rows_parts, position_parts = [], [], []
        offset = 0
        for key in keys:
            states, rows, positions = pending[key]
            states_parts.append(states)
            rows_parts.append(rows + offset)
            position_parts.append(positions)
            offset += int(states.shape[0])
        reply = yield (
            torch.cat(states_parts, dim=0),
            torch.cat(rows_parts, dim=0),
            torch.cat(position_parts, dim=0),
        )
        cursor = 0
        following: dict[int, Request] = {}
        for key in keys:
            width = int(pending[key][2].numel())
            slice_ = reply[cursor : cursor + width]
            cursor += width
            try:
                following[key] = subgenerators[key].send(slice_)
            except StopIteration as stop:
                results[key] = stop.value
        pending = following
    return results


def _rs2(
    lo: int, hi: int, state: torch.Tensor, ctx: _Context
) -> Generator[Request, Reply, tuple[torch.Tensor, torch.Tensor]]:
    """RS2 over sigma indices [lo, hi).

    `state` is [1, prompt+canvas] with the block's positions still masked.
    Returns the state with the block filled, and `log mu_S` of what it filled.
    """
    width = hi - lo
    positions = ctx.positions(lo, hi)
    ctx.counters["blocks_opened"] += 1

    # --- One parallel round: every marginal pi(. | Z_<S), from ONE state row.
    marginal_logits = yield (
        state,
        torch.zeros(width, dtype=torch.long, device=ctx.device),
        positions,
    )
    log_marginal = _filtered_log_probs(marginal_logits, ctx.excluded)     # [m, V]
    del marginal_logits
    # X ~ nu_S: independent draws from the marginals, all positions at once.
    speculation = torch.multinomial(log_marginal.exp(), num_samples=1).squeeze(-1)
    log_nu = log_marginal.gather(1, speculation.unsqueeze(1)).squeeze(1).sum()

    if width == 1:
        # Base case. nu_S == mu_S, so u == 1 and this is ordinary ancestral
        # sampling of one position. The paper's "if b <= a+1 then u = 1".
        filled = state.clone()
        filled[0, positions] = speculation
        ctx.commit(1)
        ctx.counters["singleton_blocks"] += 1
        return filled, log_nu

    # --- One parallel round: the chain-rule conditionals at X.
    # Row s carries X_{lo..lo+s-1} and is read at the s-th block position, so
    # row 0 is the untouched state and its conditional IS the marginal there.
    probe = state.repeat(width, 1)
    for row in range(1, width):
        probe[row, positions[:row]] = speculation[:row]
    conditional_logits = yield (
        probe,
        torch.arange(width, device=ctx.device),
        positions,
    )
    log_conditional = _filtered_log_probs(conditional_logits, ctx.excluded)
    del conditional_logits, probe
    log_mu = log_conditional.gather(1, speculation.unsqueeze(1)).squeeze(1).sum()
    del log_conditional

    # --- The speculative coin: accept the whole block with min(1, dmu/dnu).
    acceptance = torch.clamp((log_mu - log_nu).exp(), max=1.0)
    if _coin(acceptance):
        filled = state.clone()
        filled[0, positions] = speculation
        ctx.commit(width)
        ctx.counters["speculations_accepted"] += 1
        return filled, log_mu

    ctx.counters["speculations_rejected"] += 1

    # --- Rejection: geometrically growing parallel batches of true mu_S draws.
    #
    # The paper indexes candidates by i and round r takes
    # i in [ceil((1+rho)^r), ceil((1+rho)^(r+1)) - 1], so after round r exactly
    # ceil((1+rho)^(r+1)) - 1 candidates have been drawn. For small rho those
    # two ceilings often coincide and the round draws nothing: at rho = 1/8,
    # rounds 1 through 4 are all empty. Rather than step r through them one at
    # a time, solve for the next r whose cumulative target actually moves --
    # (1+rho)^(r+1) > drawn + 1 -- which is the same schedule, reached directly.
    drawn = 0
    exponent = 0
    while True:
        exponent = max(
            exponent,
            math.floor(math.log(drawn + 1) / math.log1p(ctx.rho)) + 1,
        )
        target = math.ceil((1.0 + ctx.rho) ** exponent) - 1
        exponent += 1
        count = target - drawn
        if count <= 0:
            raise RuntimeError("rejection schedule failed to advance")
        candidates = yield from _interleave(
            [_fallback(lo, hi, state, ctx) for _ in range(count)]
        )
        drawn += count
        ctx.counters["fallback_draws"] += count
        ctx.counters["max_fallback_draws_in_call"] = max(
            ctx.counters["max_fallback_draws_in_call"], drawn
        )
        for candidate_state, candidate_log_mu in candidates:
            values = candidate_state[0, positions]
            candidate_log_nu = (
                log_marginal.gather(1, values.unsqueeze(1)).squeeze(1).sum()
            )
            # p = max(0, 1 - dnu/dmu(Y)). No forward pass: mu came back with
            # the candidate and nu is a lookup into the cached marginals.
            keep = torch.clamp(
                1.0 - (candidate_log_nu - candidate_log_mu).exp(), min=0.0
            )
            if _coin(keep):
                ctx.commit(width)
                return candidate_state, candidate_log_mu


def _fallback(
    lo: int, hi: int, state: torch.Tensor, ctx: _Context
) -> Generator[Request, Reply, tuple[torch.Tensor, torch.Tensor]]:
    """One true draw from mu_S, by splitting the block and recursing.

    The two halves are SEQUENTIAL -- the second conditions on the first. That
    sequencing is what the whole construction is built around; only the
    rejection batch above it is parallel.
    """
    middle = (lo + hi) // 2
    left_state, left_log_mu = yield from _rs2(lo, middle, state, ctx)
    right_state, right_log_mu = yield from _rs2(middle, hi, left_state, ctx)
    # Chain rule over blocks: mu_S = mu_T1 * mu_T2(. | T1).
    return right_state, left_log_mu + right_log_mu


def _forward_pooled(request: Request, ctx: _Context) -> torch.Tensor:
    """Run one pooled request, chunked to `max_batch_rows`. Returns [N, V].

    One parallel round is chunked into as many physical rounds as the budget
    requires, and every chunk here is counted as one. Under block diffusion the
    budget never binds -- see `max_batch_rows` -- so this loop runs once.
    """
    states, rows, positions = request
    provider = ctx.provider
    total = int(states.shape[0])
    ctx.counters["max_pooled_request_rows"] = max(
        ctx.counters["max_pooled_request_rows"], total
    )
    budget = ctx.config.max_batch_rows
    collected: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, total, budget):
        stop = min(start + budget, total)
        chunk = states[start:stop]
        selected = torch.nonzero((rows >= start) & (rows < stop), as_tuple=False).flatten()
        if selected.numel() == 0:
            # No read lands in this chunk, so it is never forwarded and must
            # not be charged. Cannot happen for requests `_rs2` builds, where
            # every state row is read, but pooling makes it representable.
            continue
        ctx.counters["oracle_rounds_physical"] += 1
        ctx.counters["oracle_rows"] += stop - start
        local_rows = rows[selected] - start
        local_positions = positions[selected]
        # Algorithm 3 has no separate proposer: nu comes from the same oracle
        # call the verifier's first row would have made, so there is one stage
        # to attribute here, not two.
        with ctx.timer("oracle"):
            if _uses_lean_path(provider):
                hidden = provider.hidden_states(chunk)
                logits = provider.logits_from_hidden(hidden, local_rows, local_positions)
                del hidden
            else:
                logits = provider.forward_logits(chunk)[local_rows, local_positions, :]
        collected.append((selected, logits))
    if not collected:
        raise RuntimeError("pooled request read no positions")
    width = int(rows.numel())
    output = torch.empty(
        (width, collected[0][1].shape[-1]),
        dtype=collected[0][1].dtype,
        device=collected[0][1].device,
    )
    for selected, logits in collected:
        output[selected] = logits
    return output


def _drive(
    generator: Generator[Request, Reply, Any], ctx: _Context
) -> Any:
    """The only place that touches the provider. One iteration = one round."""
    try:
        request = generator.send(None)
        while True:
            request = generator.send(_forward_pooled(request, ctx))
    except StopIteration as stop:
        return stop.value


def run_autospec_block(
    provider: LogitsProvider,
    state: torch.Tensor,
    config: AutospecConfig,
    order: torch.Tensor,
    timer: Any = None,
) -> tuple[torch.Tensor, "_Context"]:
    """Run Algorithm 3 over ONE block, in place, and hand back its counters.

    `state` is block-local, `[1, block]`, and `order` holds the block-local
    indices of the positions to fill, in sigma order -- so `canvas_start` is 0
    and `ctx.positions` indexes the block directly. Positions absent from
    `order` (the prompt's remainder in the first decode block) are never
    written.

    Separate from `generate_autospeculative`, which owns the full-canvas case,
    because under block diffusion the canvas is a loop over blocks and the
    cache, masks and commits belong to that loop (DESIGN.md D8).
    """
    ctx = _Context(provider, config, order, 0, provider.device, timer=timer)
    filled, _ = _drive(_rs2(0, int(order.numel()), state, ctx), ctx)
    return filled, ctx


def generate_autospeculative(
    provider: LogitsProvider,
    prompt_ids: torch.Tensor,
    config: AutospecConfig,
    *,
    capture_traces: bool = False,
    sigma_rank: torch.Tensor | None = None,
) -> AutospecResult:
    if sigma_rank is None:
        raise ValueError("autospec requires a sigma_rank permutation (DESIGN.md D2)")
    if sigma_rank.numel() != config.continuation_length:
        raise ValueError(
            f"sigma_rank has {sigma_rank.numel()} entries for a canvas of "
            f"{config.continuation_length}"
        )
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("Authoritative latency generation requires one prompt at a time")

    device = provider.device
    canvas_start = int(prompt_ids.shape[1])
    length = config.continuation_length
    canvas = torch.full((1, length), provider.mask_token_id, dtype=torch.long, device=device)
    state = torch.cat((prompt_ids.to(device), canvas), dim=1)
    # sigma_rank[p] is when position p is visited; sigma[j] is what is visited
    # j-th, which is what the block ranges index into.
    sigma = torch.argsort(sigma_rank.to(device))

    ctx = _Context(provider, config, sigma, canvas_start, device)
    final_state, _ = _drive(_rs2(0, length, state, ctx), ctx)

    continuation = final_state[0, canvas_start:].detach().cpu().tolist()
    try:
        eos_position = continuation.index(provider.eos_token_id)
    except ValueError:
        eos_position = None
    scored = continuation if eos_position is None else continuation[:eos_position]
    histogram = [0] * length
    for width, count in ctx.commitments.items():
        histogram[width - 1] += count
    counters = ctx.counters
    return AutospecResult(
        continuation_token_ids=[int(value) for value in scored],
        traces=ctx.traces if capture_traces else [],
        # D4: the headline is what the hardware did.
        denoising_steps=counters["oracle_rounds_physical"],
        # There is no separate proposer; nu comes from the same oracle call
        # that the verifier's first row would have made.
        proposer_passes=0,
        verifier_rows=counters["oracle_rows"],
        verifier_forward_passes=counters["oracle_rounds_physical"],
        commitment_histogram=histogram,
        eos_position=eos_position,
        maximum_length_termination=eos_position is None,
        full_continuation_token_ids=[int(value) for value in continuation],
        **counters,
    )
