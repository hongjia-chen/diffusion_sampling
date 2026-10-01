from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sdar_harness.model import FAMILIES, MaskedDiffusionProvider, ModelOptions


class Tokenizer:
    mask_token_id = 9
    eos_token_id = 8
    pad_token_id = 8
    bos_token_id = 7


class Model:
    def __init__(self) -> None:
        self.attention_mask = "unset"

    def __call__(self, *, input_ids, attention_mask=None):
        self.attention_mask = attention_mask
        logits = torch.arange(
            input_ids.shape[1] * 3, dtype=torch.float32
        ).reshape(1, input_ids.shape[1], 3)
        return SimpleNamespace(logits=logits)


def provider_for(family: str) -> tuple[Model, MaskedDiffusionProvider]:
    model = Model()
    return model, MaskedDiffusionProvider(
        model, Tokenizer(), torch.device("cpu"), FAMILIES[family]
    )


def test_dream_family_uses_the_full_attention_sentinel_and_shifts_logits() -> None:
    model, provider = provider_for("dream")
    logits = provider.forward_logits(torch.tensor([[1, 2, 3]]))
    assert model.attention_mask == "full"
    assert logits.tolist() == [[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]]


def test_generic_family_passes_a_real_mask_and_does_not_shift() -> None:
    model, provider = provider_for("generic")
    logits = provider.forward_logits(torch.tensor([[1, 2, 3]]))
    assert torch.equal(model.attention_mask, torch.ones(1, 3, dtype=torch.long))
    assert logits.tolist() == [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]


def test_the_two_alignments_really_do_differ() -> None:
    """Guards the one mistake that produces fluent-but-wrong text silently."""
    _, dream = provider_for("dream")
    _, generic = provider_for("generic")
    ids = torch.tensor([[1, 2, 3]])
    assert not torch.equal(dream.forward_logits(ids), generic.forward_logits(ids))


def test_model_id_is_required_and_family_must_be_known() -> None:
    with pytest.raises(ValueError, match="model_id is required"):
        ModelOptions(model_id="")
    with pytest.raises(ValueError, match="unknown model family"):
        ModelOptions(model_id="some/checkpoint", family="llada")


class BlockModel:
    """A checkpoint exposing the .model/.lm_head pair the lean path needs."""

    HIDDEN = 4
    VOCAB = 3

    def __init__(self) -> None:
        self.probe_lengths: list[int] = []
        outer = self

        class Backbone:
            def __call__(self, *, input_ids, attention_mask=None):
                outer.probe_lengths.append(int(input_ids.shape[1]))
                if attention_mask is not None and not isinstance(attention_mask, str):
                    # The block-causal mask is square over the sequence.
                    assert attention_mask.shape[-1] == input_ids.shape[1]
                hidden = torch.zeros(
                    input_ids.shape[0], input_ids.shape[1], outer.HIDDEN
                )
                return SimpleNamespace(last_hidden_state=hidden)

        self.model = Backbone()
        self.lm_head = lambda hidden: torch.zeros(*hidden.shape[:-1], self.VOCAB)

    def __call__(self, *, input_ids, attention_mask=None, **kwargs):
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return SimpleNamespace(logits=self.lm_head(hidden.last_hidden_state))


def test_lean_self_check_probe_is_rounded_up_to_a_whole_block() -> None:
    """A 12-token probe has no block-causal mask, so it must be padded.

    Without this the self-check raises inside `_attention_kwargs`, the provider
    silently degrades to the full [B, S, V] path, and a run started with
    `require_lean_logits` refuses to load a checkpoint that is in fact fine.
    """
    model = BlockModel()
    provider = MaskedDiffusionProvider(
        model, Tokenizer(), torch.device("cpu"), FAMILIES["sdar"]
    )
    assert provider.verify_lean_path() is True
    block = FAMILIES["sdar"].block_length
    assert provider.probe_length_used % block == 0
    assert all(length % block == 0 for length in model.probe_lengths)


def test_lean_self_check_probe_is_untouched_for_a_full_canvas_model() -> None:
    model = BlockModel()
    provider = MaskedDiffusionProvider(
        model, Tokenizer(), torch.device("cpu"), FAMILIES["dream"]
    )
    assert provider.verify_lean_path() is True
    assert model.probe_lengths == [12, 12]
