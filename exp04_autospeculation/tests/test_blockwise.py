"""Block-diffusion generation against a fake that mirrors SDAR's cache protocol.

The fake reproduces the exact branch from `modeling_sdar.py`:

    if past_key_value is not None and store_kv:      cache.update(...)
    elif past_key_value is not None and not store_kv: read and concatenate

so the store_kv discipline, the expanded read and the block-causal slicing are
all exercised rather than assumed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.cache_utils import DynamicCache

from autospec.blockwise import ExpandedCache, generate_blockwise
from autospec.model import FAMILIES, MaskedDiffusionProvider
from autospec.scheduler import GenerationConfig

VOCAB, MASK, EOS, HIDDEN = 7, 5, 6, 4


class FakeBackbone:
    """Returns hidden states that encode the position and the cached prefix."""

    def __init__(self, log: list):
        self.log = log

    def __call__(self, *, input_ids, attention_mask, position_ids,
                 past_key_values=None, use_cache=None, store_kv=None):
        rows, width = input_ids.shape
        prefix = 0
        if past_key_values is not None and len(past_key_values):
            keys, _ = past_key_values[0]
            if keys.shape[0] != rows:
                raise AssertionError(
                    f"cache batch {keys.shape[0]} != input batch {rows}; the "
                    "expanded view is what makes these agree"
                )
            prefix = int(keys.shape[-2])
        if store_kv:
            new = torch.zeros(rows, 1, width, HIDDEN)
            past_key_values.update(new, new.clone(), 0)
        self.log.append({
            "rows": rows, "width": width, "store_kv": bool(store_kv),
            "prefix": prefix,
            "mask_all_true": bool(attention_mask.all()),
            "mask_shape": tuple(attention_mask.shape),
            "first_position": int(position_ids[0, 0]),
        })
        hidden = torch.zeros(rows, width, HIDDEN)
        hidden[..., 0] = position_ids.to(hidden.dtype)
        hidden[..., 1] = float(prefix)
        return SimpleNamespace(last_hidden_state=hidden)


class FakeHead(torch.nn.Module):
    """Predicts `position % 5`, with confidence falling along the block."""

    def forward(self, hidden):
        position = hidden[..., 0]
        logits = torch.full(hidden.shape[:-1] + (VOCAB,), -9.0)
        target = (position.long() % 5)
        peak = 6.0 - 0.1 * (position % 7)
        logits.scatter_(-1, target.unsqueeze(-1), peak.unsqueeze(-1))
        return logits


class FakeSDAR:
    def __init__(self):
        self.log: list = []
        self.model = FakeBackbone(self.log)
        self.lm_head = FakeHead()
        self.config = SimpleNamespace(max_position_embeddings=4096)

    def __call__(self, **kwargs):
        return SimpleNamespace(logits=self.lm_head(self.model(**kwargs).last_hidden_state))


class Tok:
    mask_token_id, eos_token_id, pad_token_id, bos_token_id = MASK, EOS, EOS, None
    name_or_path = "fake/sdar"


def build(block_length=4):
    model = FakeSDAR()
    conventions = FAMILIES["sdar"].__class__(
        logits_alignment="none", attention_mask="block_causal",
        loader="auto_model_for_causal_lm", block_length=block_length,
    )
    provider = MaskedDiffusionProvider(model, Tok(), torch.device("cpu"), conventions)
    return model, provider


def run(k=1, block_length=4, prompt_length=8, generated=8):
    model, provider = build(block_length)
    prompt = torch.arange(prompt_length, dtype=torch.long).unsqueeze(0) % 5
    config = GenerationConfig(
        method="fixed", continuation_length=generated, fixed_k=k,
        max_candidates=block_length, verifier_microbatch_size=2,
    )
    result = generate_blockwise(provider, prompt, config)
    return model, result


def test_canvas_is_filled_and_matches_the_position_rule() -> None:
    _, result = run(k=1)
    assert MASK not in result.full_continuation_token_ids
    # Generated span starts at absolute position 8, and the fake predicts p % 5.
    assert result.full_continuation_token_ids == [p % 5 for p in range(8, 16)]


def test_store_kv_discipline() -> None:
    """Denoising must never write K/V; each finished block commits exactly once."""
    model, _ = run(k=1, block_length=4, prompt_length=8, generated=8)
    writes = [e for e in model.log if e["store_kv"]]
    reads = [e for e in model.log if not e["store_kv"]]
    # 1 prefill (2 prompt blocks at once) + 1 commit per generated block.
    assert len(writes) == 3
    assert writes[0]["width"] == 8, "prefill covers both whole prompt blocks"
    assert all(e["width"] == 4 for e in writes[1:]), "commits are one block wide"
    assert all(e["rows"] == 1 for e in writes), "a block is committed at batch 1"
    assert reads, "denoising forwards happened"
    assert all(not e["store_kv"] for e in reads)


def test_prefill_is_block_causal_but_decode_slices_are_all_ones() -> None:
    model, _ = run(k=1)
    prefill = model.log[0]
    assert prefill["store_kv"] and prefill["width"] == 8
    # Prompt block 0 must not see prompt block 1.
    assert not prefill["mask_all_true"]
    # Decode forwards slice keys to exactly the attendable set, so the slice is
    # all-ones and the checkpoint takes its fast bidirectional path. The block
    # structure is carried by WHICH keys are present.
    decode = [e for e in model.log if not e["store_kv"]]
    assert all(e["mask_all_true"] for e in decode)


def test_the_cache_actually_grows_and_is_read() -> None:
    model, _ = run(k=1, block_length=4, prompt_length=8, generated=8)
    decode = [e for e in model.log if not e["store_kv"]]
    # First generated block reads the 8-token prompt prefix; the second reads 12.
    assert decode[0]["prefix"] == 8
    assert decode[-1]["prefix"] == 12


def test_a_block_costs_steps_plus_one_forward() -> None:
    for k in (1, 2, 4):
        model, result = run(k=k, block_length=4, prompt_length=8, generated=8)
        steps_per_block = 4 // k
        assert result.denoising_steps == 2 * steps_per_block
        commits = len([e for e in model.log if e["store_kv"]]) - 1   # minus prefill
        assert commits == 2
        # Commit forwards are real cost and are counted, but they are reported
        # apart from the denoising forwards so a stage breakdown can attribute
        # them. The prefill commit is the "+ 1".
        assert result.proposer_passes == result.denoising_steps
        assert result.commit_passes == commits + 1
        # The NFE total every earlier run recorded is unchanged.
        assert (
            result.proposer_passes + result.commit_passes
            == result.denoising_steps + commits + 1
        )


def test_fixed_k_commits_k_per_step_within_the_block() -> None:
    for k in (1, 2, 4):
        _, result = run(k=k, block_length=4)
        assert result.commitment_histogram[k - 1] == 2 * (4 // k)
        assert sum(result.commitment_histogram) == 2 * (4 // k)


def test_a_prompt_that_is_not_a_whole_number_of_blocks() -> None:
    """The remainder lands in the first decode block and is not overwritten."""
    model, result = run(k=1, block_length=4, prompt_length=6, generated=8)
    prefill = [e for e in model.log if e["store_kv"]]
    # Only ONE whole prompt block can be prefilled; positions 4 and 5 stay in
    # the first decode block alongside two masked positions.
    assert prefill[0]["width"] == 4
    assert MASK not in result.full_continuation_token_ids
    assert result.full_continuation_token_ids == [p % 5 for p in range(6, 14)]


def test_expanded_cache_is_a_free_view_and_refuses_writes() -> None:
    cache = DynamicCache()
    keys = torch.zeros(1, 2, 16, 8)
    cache.update(keys, keys.clone(), 0)
    expanded = ExpandedCache(cache, 64)
    view, _ = expanded[0]
    assert view.shape == (64, 2, 16, 8)
    assert view.untyped_storage().nbytes() == keys.untyped_storage().nbytes(), \
        "expand must be a stride-0 view, not a copy"
    with pytest.raises(RuntimeError, match="read-only"):
        expanded.update(keys, keys, 0)


def test_adaptive_runs_inside_a_block_too() -> None:
    model, provider = build(block_length=4)
    prompt = torch.arange(8, dtype=torch.long).unsqueeze(0) % 5
    config = GenerationConfig(
        method="adaptive", continuation_length=8, delta=0.5,
        max_candidates=4, verifier_microbatch_size=2,
    )
    result = generate_blockwise(provider, prompt, config)
    assert MASK not in result.full_continuation_token_ids
    assert result.verifier_rows > 0, "the verifier ran"
    assert len([e for e in model.log if e["store_kv"]]) == 3


# ----------------------------------------------------------------------
# Algorithm 3 under the block loop (DESIGN.md D8)
# ----------------------------------------------------------------------

from autospec.autospeculation import AutospecConfig, AutospecResult
from autospec.blockwise import block_sigma_order


def test_block_sigma_order_is_a_permutation_of_the_blocks_masked_positions() -> None:
    # Canvas positions 8..16, block [8, 12): global ranks pick the order.
    sigma_rank = torch.tensor([3, 0, 2, 1, 7, 6, 5, 4])
    masked_local = torch.arange(4)
    order = block_sigma_order(sigma_rank, masked_local, low=8, prompt_length=8, generated=8)

    assert sorted(int(v) for v in order) == [0, 1, 2, 3]
    # rank[1]=0 is smallest, then rank[3]=1, rank[2]=2, rank[0]=3.
    assert [int(v) for v in order] == [1, 3, 2, 0]


def test_block_sigma_order_handles_a_prompt_that_straddles_a_block() -> None:
    """The bug this replaces sliced with a NEGATIVE index and came back empty.

    With prompt 6 and block 4 the first decode block is [4, 8), whose first two
    positions are prompt. Only positions 6 and 7 are masked, and they must map
    to canvas indices 0 and 1 -- not to a silently empty tensor.
    """
    sigma_rank = torch.tensor([1, 0, 2, 3])
    masked_local = torch.tensor([2, 3])          # local 2,3 == absolute 6,7
    order = block_sigma_order(sigma_rank, masked_local, low=4, prompt_length=6, generated=4)

    assert order.numel() == 2
    assert sorted(int(v) for v in order) == [2, 3]
    # canvas index of local 3 is 1, whose rank 0 beats local 2's rank 1.
    assert [int(v) for v in order] == [3, 2]


def test_block_sigma_order_puts_the_rounding_tail_last() -> None:
    """Rounding the canvas up leaves positions with no sigma_rank entry."""
    sigma_rank = torch.tensor([1, 0])            # generated == 2
    masked_local = torch.arange(4)               # block covers 2 real + 2 tail
    order = block_sigma_order(sigma_rank, masked_local, low=0, prompt_length=0, generated=2)

    # Real positions first, in sigma order; the two tail positions after them.
    assert [int(v) for v in order] == [1, 0, 2, 3]


def run_autospec(block_length=4, prompt_length=8, generated=8, seed=0):
    model, provider = build(block_length)
    prompt = torch.arange(prompt_length, dtype=torch.long).unsqueeze(0) % 5
    config = AutospecConfig(continuation_length=generated, max_batch_rows=16)
    torch.manual_seed(seed)
    sigma_rank = torch.randperm(generated)
    result = generate_blockwise(provider, prompt, config, sigma_rank=sigma_rank)
    return model, result


def test_autospec_runs_under_the_block_loop_and_fills_the_canvas() -> None:
    model, result = run_autospec()

    assert isinstance(result, AutospecResult)
    assert MASK not in result.full_continuation_token_ids
    assert len(result.full_continuation_token_ids) == 8
    # The oracle is the only forward kind Algorithm 3 makes; commits are the
    # block loop's, and both are counted.
    assert result.verifier_forward_passes == result.oracle_rounds_physical
    assert result.proposer_passes == 0
    assert result.commit_passes == len([e for e in model.log if e["store_kv"]])


def test_autospec_under_blocks_still_commits_every_block_to_the_cache() -> None:
    model, result = run_autospec(block_length=4, prompt_length=8, generated=8)
    commits = [e for e in model.log if e["store_kv"]]
    # One prefill commit for the prompt's two blocks, then one per decode block.
    assert len(commits) == 3
    assert result.blocks_opened > 0


def test_autospec_under_blocks_requires_a_sigma_matching_the_canvas() -> None:
    _, provider = build(4)
    prompt = torch.zeros(1, 8, dtype=torch.long)
    config = AutospecConfig(continuation_length=8)
    with pytest.raises(ValueError, match="sigma_rank"):
        generate_blockwise(provider, prompt, config)
    with pytest.raises(ValueError, match="entries for a canvas"):
        generate_blockwise(provider, prompt, config, sigma_rank=torch.arange(4))


def test_autospec_never_overwrites_the_prompt_remainder_in_the_first_block() -> None:
    """Prompt 6 with block 4 leaves 2 prompt tokens inside the first decode block."""
    model, provider = build(block_length=4)
    prompt = torch.full((1, 6), 3, dtype=torch.long)
    config = AutospecConfig(continuation_length=6, max_batch_rows=16)
    torch.manual_seed(2)
    result = generate_blockwise(
        provider, prompt, config, sigma_rank=torch.randperm(6)
    )
    assert MASK not in result.full_continuation_token_ids
    assert len(result.full_continuation_token_ids) == 6
