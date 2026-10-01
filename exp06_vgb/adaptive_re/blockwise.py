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

from adaptive_re.model import block_causal_mask
from adaptive_re.scheduler import (
    GenerationConfig,
    GenerationResult,
    finished_at,
    generation_step,
    stop_token_ids,
    truncate_at_stop,
)
from adaptive_re.timing import NULL_TIMER, StageTimer


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
        self.stop_token_ids = stop_token_ids(provider)
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
        return self._provider.project(hidden[rows, positions, :])

    @torch.inference_mode()
    def forward_logits(self, block_ids: torch.Tensor) -> torch.Tensor:
        block_ids = block_ids.to(self.device, non_blocking=True)
        self.forward_passes += 1
        logits = self._provider.model(input_ids=block_ids, **self._kwargs(block_ids)).logits
        return self._provider.limit_vocab(logits)



def generate_blockwise(
    provider: Any,
    prompt_ids: torch.Tensor,
    config: GenerationConfig,
    *,
    capture_traces: bool = False,
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

    stop_ids = stop_token_ids(provider)
    canvas_low, canvas_high = prompt_length, prompt_length + generated
    early_stopped = False
    traces: list[dict[str, Any]] = []
    steps = 0
    verifier_rows = 0
    verifier_passes = 0
    proposer_passes = 0

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
            if config.early_stop:
                # Earlier blocks are complete, so writing this block back is
                # all it takes for the canvas to be current. 64 tokens; cheap.
                canvas[:, low:high] = state
                if finished_at(
                    canvas[0, canvas_low:canvas_high],
                    mask_token_id=provider.mask_token_id,
                    stop_ids=stop_ids,
                ) is not None:
                    early_stopped = True
                    break

        canvas[:, low:high] = state
        if early_stopped:
            # The answer is settled. The block's K/V would only ever be read by
            # a later block, and there is none, so its commit forward is skipped.
            break
        # Commit the finished block's K/V. This forward is real cost and is
        # counted: a block costs `steps + 1` passes, not `steps`.
        with timer("commit"):
            provider.commit(
                canvas[:, low:high], attention[low:high, :high], positions[low:high], cache
            )
        commit_passes += 1

    continuation = canvas[0, prompt_length : prompt_length + generated].detach().cpu().tolist()
    scored, eos_position, stop_token = truncate_at_stop(continuation, stop_ids)
    # "Early" means decoding returned with canvas positions never decoded, not
    # merely that the answer was short.
    early_stopped = early_stopped and provider.mask_token_id in continuation
    common = dict(
        continuation_token_ids=[int(value) for value in scored],
        traces=traces,
        eos_position=eos_position,
        maximum_length_termination=eos_position is None,
        full_continuation_token_ids=[int(value) for value in continuation],
        commit_passes=commit_passes,
        stage_seconds=timer.totals(),
        stop_token_id=stop_token,
        early_stopped=early_stopped,
        decoded_positions=sum(1 for value in continuation if value != provider.mask_token_id),
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
