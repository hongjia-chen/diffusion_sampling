"""Block-diffusion generation: the canvas as a loop over blocks.

A full-canvas dLLM like Dream denoises the whole canvas under bidirectional
attention, so nothing can be cached and every step re-encodes everything. A
block diffusion model like SDAR is causal *across* blocks and bidirectional
*within* one, so the committed prefix has fixed K/V and only the current block
is re-encoded. That is the whole difference, and it is what this module exists
to express.

Transcribed from SDAR's own reference inference, `generate.py` in JetAstra/SDAR
(DESIGN.md D8). The pieces that matter, none of them inferred:

* the canvas is rounded UP to a whole number of blocks,
  `total_length = ceil((prompt + gen) / B) * B`;
* the prompt is not special-cased -- it occupies the leading blocks under the
  same block-causal mask, and its remainder (when `prompt % B != 0`) simply
  lands in the first decode block, where only still-masked positions are filled;
* `position_ids` are passed explicitly, sliced per block;
* every denoising forward runs `store_kv=False` so masked-token K/V never
  enters the cache, and one extra `store_kv=True` forward commits each finished
  block. A block therefore costs `steps + 1` forwards, not `steps`.

The scheduler is reused unchanged. `_propose` and `_score_all_prefixes` only
ever index relative to a canvas offset, so handing them a **block-local** state
of length B with `canvas_start=0` makes them operate on one block while
`BlockCacheProvider` transparently supplies the cached prefix.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import torch

from autospec.autospeculation import AutospecConfig, AutospecResult, run_autospec_block
from autospec.model import block_causal_mask
from autospec.scheduler import GenerationConfig, GenerationResult, generation_step
from autospec.timing import NULL_TIMER, StageTimer


def _cache_base():
    """`transformers.cache_utils.Cache`, which the forward isinstance-checks."""
    from transformers.cache_utils import Cache

    return Cache


class ExpandedCache(_cache_base()):
    """A batch-1 KV cache presented at batch N, without copying it.

    The verifier scores several probe rows that differ only in which in-block
    positions are already filled; they all share one committed prefix. SDAR's
    attention reads the cache as `past_key_value[layer_idx]` and concatenates,
    which requires matching batch dimensions -- so a batch-1 cache cannot be
    used directly by an N-row batch.

    Materialising N copies would cost 112 KiB per token per row (4.8 GiB for a
    64-row batch over a 704-token prefix). It is unnecessary: with
    `store_kv=False` the attention path only ever *reads* the cache, so a
    stride-0 `expand` view is enough and costs nothing. The concatenation the
    model then performs is a per-layer transient, freed before the next layer.

    Subclasses `Cache` because `SDARModel.forward` rejects anything else.
    """

    def __init__(self, cache: Any, batch_size: int):
        super().__init__()
        # Bypass Module.__setattr__. `Cache` subclasses `nn.Module` in
        # transformers < 4.52 and does not in >= 4.52; a plain assignment would
        # try to register the delegate cache as a submodule on the old one.
        object.__setattr__(self, "_cache", cache)
        object.__setattr__(self, "_batch_size", batch_size)

    def __getitem__(self, layer_index: int):
        keys, values = self._cache[layer_index]
        if keys.shape[0] == self._batch_size:
            return keys, values
        return (
            keys.expand(self._batch_size, -1, -1, -1),
            values.expand(self._batch_size, -1, -1, -1),
        )

    def __len__(self) -> int:
        return len(self._cache)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._cache.get_seq_length(layer_idx)

    def get_max_cache_shape(self, *args: Any, **kwargs: Any):
        getter = getattr(self._cache, "get_max_cache_shape", None)
        return getter(*args, **kwargs) if getter else None

    def update(self, *args: Any, **kwargs: Any):
        # Reached only if something asks to write through an expanded view,
        # which would silently corrupt the shared prefix. store_kv=False must
        # prevent it; if it does not, fail loudly rather than write garbage.
        raise RuntimeError(
            "ExpandedCache is read-only. A forward reached cache.update() with "
            "an expanded batch, which means store_kv was not False."
        )


class BlockCacheProvider:
    """Presents one block, against a cached prefix, with the scheduler's API.

    Wraps the real provider so `_propose` and `_score_all_prefixes` can be used
    verbatim: they call `hidden_states` / `forward_logits` on a `[rows, B]`
    block-local tensor and index positions in `[0, B)`.
    """

    def __init__(
        self,
        provider: Any,
        cache: Any,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        self._provider = provider
        self._cache = cache
        self._attention_mask = attention_mask       # [B, keys] bool
        self._position_ids = position_ids           # [B]
        self.mask_token_id = provider.mask_token_id
        self.eos_token_id = provider.eos_token_id
        self.device = provider.device
        self.forward_passes = 0

    @property
    def excluded_token_ids(self) -> set[int]:
        return self._provider.excluded_token_ids

    @property
    def supports_lean_logits(self) -> bool:
        return self._provider.supports_lean_logits

    def _kwargs(self, block_ids: torch.Tensor) -> dict[str, Any]:
        rows = int(block_ids.shape[0])
        return {
            "attention_mask": self._attention_mask.unsqueeze(0),
            "position_ids": self._position_ids.unsqueeze(0).expand(rows, -1),
            "past_key_values": ExpandedCache(self._cache, rows) if len(self._cache) else None,
            "use_cache": True,
            "store_kv": False,
        }

    @torch.inference_mode()
    def hidden_states(self, block_ids: torch.Tensor) -> torch.Tensor:
        block_ids = block_ids.to(self.device, non_blocking=True)
        self.forward_passes += 1
        outputs = self._provider.backbone(input_ids=block_ids, **self._kwargs(block_ids))
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]
        return hidden

    def logits_from_hidden(
        self, hidden: torch.Tensor, rows: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        # Block diffusion is position-aligned; there is no shift to undo.
        return self._provider.lm_head(hidden[rows, positions, :])

    @torch.inference_mode()
    def forward_logits(self, block_ids: torch.Tensor) -> torch.Tensor:
        block_ids = block_ids.to(self.device, non_blocking=True)
        self.forward_passes += 1
        return self._provider.model(input_ids=block_ids, **self._kwargs(block_ids)).logits



def block_sigma_order(
    sigma_rank: torch.Tensor,
    masked_local: torch.Tensor,
    low: int,
    prompt_length: int,
    generated: int,
) -> torch.Tensor:
    """Visit order for one block's masked positions, block-local indices.

    `sigma_rank` is the ONE global permutation D2 draws per task, over the
    generation canvas `[prompt_length, prompt_length + generated)`. Restricting
    a uniform permutation to a subset and re-ranking leaves a uniform
    permutation of that subset, so each block gets a valid sigma out of the
    single global draw -- no second RNG, and the same seed reproduces it.

    Rounding the canvas up to a whole number of blocks leaves a tail past
    `prompt_length + generated` with no `sigma_rank` entry. Those positions are
    still masked and still get filled (every sampler fills the whole block, and
    the tail is dropped from the scored continuation), so they are ordered
    after every real canvas position rather than being skipped.
    """
    canvas_index = low + masked_local - prompt_length
    inside = (canvas_index >= 0) & (canvas_index < generated)
    # Sort the two groups separately and concatenate, rather than folding them
    # into one key: a single key would have to encode "after every real rank"
    # as a magnitude, which silently depends on sigma_rank's values staying in
    # [0, generated). They do -- it is a permutation -- but the ordering should
    # not rest on that.
    real = masked_local[inside]
    ranks = sigma_rank.to(masked_local.device)[canvas_index[inside]]
    tail = masked_local[~inside]
    return torch.cat(
        (real[torch.argsort(ranks)], tail[torch.argsort(canvas_index[~inside])])
    )


def generate_blockwise(
    provider: Any,
    prompt_ids: torch.Tensor,
    config: GenerationConfig,
    *,
    capture_traces: bool = False,
    sigma_rank: torch.Tensor | None = None,
    timer: StageTimer | None = None,
) -> GenerationResult:
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("Authoritative latency generation requires one prompt at a time")
    timer = timer if timer is not None else NULL_TIMER
    block = provider.conventions.block_length
    if block is None:
        raise ValueError("generate_blockwise requires a block_length")

    device = provider.device
    prompt_length = int(prompt_ids.shape[1])
    generated = config.continuation_length
    num_blocks = math.ceil((prompt_length + generated) / block)
    total_length = num_blocks * block

    canvas = torch.full(
        (1, total_length), provider.mask_token_id, dtype=torch.long, device=device
    )
    canvas[:, :prompt_length] = prompt_ids.to(device)
    attention = block_causal_mask(total_length, block, device=device)
    positions = torch.arange(total_length, device=device)

    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    prefill_blocks = prompt_length // block
    prefill_length = prefill_blocks * block
    commit_passes = 0

    if prefill_length:
        # The prompt's own blocks. Not all-ones: block 0 must not see block 1.
        with timer("commit"):
            provider.commit(
                canvas[:, :prefill_length],
                attention[:prefill_length, :prefill_length],
                positions[:prefill_length],
                cache,
            )
        commit_passes += 1

    traces: list[dict[str, Any]] = []
    steps = 0
    verifier_rows = 0
    verifier_passes = 0
    proposer_passes = 0

    # Algorithm 3 is recursive over a block rather than a loop of steps, so it
    # takes a different per-block call. Everything around it -- the cache, the
    # block-causal slices, the commit -- is identical, which is why it lives
    # here rather than in a parallel copy of this loop (DESIGN.md D8).
    autospec = isinstance(config, AutospecConfig)
    if autospec:
        if sigma_rank is None:
            raise ValueError("autospec requires a sigma_rank permutation (DESIGN.md D2)")
        if sigma_rank.numel() != generated:
            raise ValueError(
                f"sigma_rank has {sigma_rank.numel()} entries for a canvas of {generated}"
            )
        histogram = [0] * block
        counters: dict[str, int] = {}
        commitments: dict[int, int] = {}
        block_config = None
    else:
        histogram = [0] * config.max_candidates
        # One block's config: the scheduler sees a canvas exactly one block wide.
        block_config = replace(
            config,
            continuation_length=block,
            max_candidates=min(config.max_candidates, block),
        )

    for index in range(prefill_blocks, num_blocks):
        low, high = index * block, (index + 1) * block
        state = canvas[:, low:high].clone()
        if not bool((state == provider.mask_token_id).any()):
            continue
        # Keys are sliced to exactly the attendable set, so this slice is
        # all-ones and the checkpoint takes its fast bidirectional path. The
        # block structure is carried by WHICH keys are present, not by zeros.
        scoped = BlockCacheProvider(
            provider, cache, attention[low:high, :high], positions[low:high]
        )
        if autospec:
            masked_local = torch.nonzero(
                state[0] == provider.mask_token_id, as_tuple=False
            ).flatten()
            order = block_sigma_order(
                sigma_rank, masked_local, low, prompt_length, generated
            )
            # n is this block's masked count, so rho = 1/log2(n) is the paper's
            # rho for the block Algorithm 3 actually runs on.
            state, block_ctx = run_autospec_block(
                scoped,
                state,
                replace(config, continuation_length=int(order.numel())),
                order,
                timer=timer,
            )
            # The counters ARE the accounting for this path -- the result reads
            # them directly, so there is nothing to mirror into the scheduler's
            # step tallies.
            for name, value in block_ctx.counters.items():
                counters[name] = counters.get(name, 0) + value
            for width, count in block_ctx.commitments.items():
                commitments[width] = commitments.get(width, 0) + count
            if capture_traces:
                for trace in block_ctx.traces:
                    entry = dict(trace)
                    entry["block_index"] = index
                    traces.append(entry)
            canvas[:, low:high] = state
            with timer("commit"):
                provider.commit(
                    canvas[:, low:high],
                    attention[low:high, :high],
                    positions[low:high],
                    cache,
                )
            commit_passes += 1
            continue

        while bool((state == provider.mask_token_id).any()):
            step = generation_step(
                scoped,
                state,
                canvas_start=0,
                config=block_config,
                step_index=steps,
                capture_trace=capture_traces,
                timer=timer,
            )
            state = step.state
            steps += 1
            proposer_passes += 1
            verifier_rows += step.verifier_rows
            verifier_passes += step.verifier_forward_passes
            histogram[step.accepted_prefix_length - 1] += 1
            if capture_traces:
                trace = dict(step.trace)
                trace["block_index"] = index
                traces.append(trace)

        canvas[:, low:high] = state
        # Commit the finished block's K/V. This forward is real cost and is
        # counted: a block costs `steps + 1` passes, not `steps`.
        with timer("commit"):
            provider.commit(
                canvas[:, low:high], attention[low:high, :high], positions[low:high], cache
            )
        commit_passes += 1

    continuation = canvas[0, prompt_length : prompt_length + generated].detach().cpu().tolist()
    try:
        eos_position = continuation.index(provider.eos_token_id)
    except ValueError:
        eos_position = None
    scored = continuation if eos_position is None else continuation[:eos_position]
    common = dict(
        continuation_token_ids=[int(value) for value in scored],
        traces=traces,
        eos_position=eos_position,
        maximum_length_termination=eos_position is None,
        full_continuation_token_ids=[int(value) for value in continuation],
        commit_passes=commit_passes,
        stage_seconds=timer.totals(),
    )
    if autospec:
        for width, count in commitments.items():
            histogram[width - 1] += count
        # `.get`: a canvas with nothing to fill opens no block and so records
        # no counters, and an empty result is still a valid one.
        return AutospecResult(
            denoising_steps=counters.get("oracle_rounds_physical", 0),
            # There is no separate proposer: nu comes from the same oracle call
            # the verifier's first row would have made.
            proposer_passes=0,
            verifier_rows=counters.get("oracle_rows", 0),
            verifier_forward_passes=counters.get("oracle_rounds_physical", 0),
            commitment_histogram=histogram,
            **common,
            **counters,
        )
    return GenerationResult(
        denoising_steps=steps,
        # Every denoising step is one proposer pass. The commit forwards are
        # reported separately rather than folded in here, so a stage breakdown
        # can attribute them; `model_forward_passes` still sums all three.
        proposer_passes=proposer_passes,
        verifier_rows=verifier_rows,
        verifier_forward_passes=verifier_passes,
        commitment_histogram=histogram,
        **common,
    )
