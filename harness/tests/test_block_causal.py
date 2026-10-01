"""The block-causal attention mask SDAR was trained with (DESIGN.md D8).

Getting this wrong is the failure mode that produces fluent text with
inconsistent conditionals, so it is checked against SDAR's own formula from
`generate.py` rather than against intuition.
"""
from __future__ import annotations

import pytest
import torch

from sdar_harness.model import FAMILIES, Conventions, ModelOptions, block_causal_mask


def reference(total_length: int, block_length: int) -> torch.Tensor:
    """Verbatim from JetAstra/SDAR generate.py."""
    blocks = total_length // block_length
    block_mask = torch.tril(torch.ones(blocks, blocks))
    return (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
    ).bool()


@pytest.mark.parametrize("total,block", [(8, 4), (16, 4), (64, 8), (256, 64), (12, 3)])
def test_matches_sdar_reference_formula(total: int, block: int) -> None:
    assert torch.equal(block_causal_mask(total, block), reference(total, block))


def test_bidirectional_within_a_block_and_causal_across() -> None:
    mask = block_causal_mask(8, 4)
    # Inside block 0 every position sees every other, both directions.
    assert bool(mask[0, 3]) and bool(mask[3, 0])
    # Block 1 sees block 0 ...
    assert bool(mask[4, 0])
    # ... and block 0 does NOT see block 1. This is the property that makes a
    # KV cache possible and that an all-ones mask would destroy.
    assert not bool(mask[0, 4])


def test_it_is_not_an_all_ones_mask_nor_token_causal() -> None:
    mask = block_causal_mask(8, 4)
    assert not bool(mask.all()), "an all-ones mask is what SDAR was NOT trained with"
    token_causal = torch.tril(torch.ones(8, 8, dtype=torch.bool))
    assert not torch.equal(mask, token_causal), "block-causal is looser than token-causal"
    # Strictly between the two: every token-causal edge is allowed, plus the
    # forward edges inside a block.
    assert bool((mask | token_causal).eq(mask).all())


def test_rejects_a_canvas_that_is_not_a_whole_number_of_blocks() -> None:
    # SDAR rounds the canvas UP to a block boundary; silently truncating would
    # drop generated positions.
    with pytest.raises(ValueError, match="whole number"):
        block_causal_mask(10, 4)
    with pytest.raises(ValueError, match="block_length must be positive"):
        block_causal_mask(8, 0)


def test_sdar_family_conventions() -> None:
    sdar = FAMILIES["sdar"]
    assert sdar.logits_alignment == "none"      # reference writes back unshifted
    assert sdar.attention_mask == "block_causal"
    assert sdar.loader == "auto_model_for_causal_lm"   # AutoModel has no lm_head
    assert sdar.block_length == 64
    # Dream must be untouched by any of this.
    assert FAMILIES["dream"].loader == "auto_model"
    assert FAMILIES["dream"].block_length is None


def test_block_length_follows_the_checkpoint() -> None:
    for length in (8, 16, 32, 64):
        options = ModelOptions(model_id="JetLM/x", family="sdar", block_length=length)
        assert options.conventions.block_length == length
        assert options.conventions.attention_mask == "block_causal"
    assert ModelOptions(model_id="x", family="sdar").conventions.block_length == 64
    with pytest.raises(ValueError, match="block_length must be positive"):
        ModelOptions(model_id="x", family="sdar", block_length=0)


def test_block_causal_requires_a_block_length() -> None:
    from sdar_harness.model import MaskedDiffusionProvider

    class Tok:
        mask_token_id, eos_token_id, pad_token_id, bos_token_id = 9, 8, 8, 7

    provider = MaskedDiffusionProvider(
        object(), Tok(), torch.device("cpu"),
        Conventions(logits_alignment="none", attention_mask="block_causal"),
    )
    with pytest.raises(ValueError, match="requires a block_length"):
        provider._attention_kwargs(torch.zeros(1, 8, dtype=torch.long))
