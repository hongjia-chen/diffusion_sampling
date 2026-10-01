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

from adaptive_re.blockwise import ExpandedCache, generate_blockwise
from adaptive_re.model import FAMILIES, MaskedDiffusionProvider
from adaptive_re.scheduler import GenerationConfig

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


class EosAfterHead(FakeHead):
    """Like FakeHead, but predicts EOS at every absolute position >= `start`."""

    def __init__(self, start: int):
        super().__init__()
        self.start = start

    def forward(self, hidden):
        logits = super().forward(hidden)
        position = hidden[..., 0]
        eos = position >= self.start
        logits[eos] = -9.0
        logits[..., EOS] = torch.where(eos, 8.0, -9.0)
        return logits


def run_eos(early_stop: bool, start: int = 13):
    model = FakeSDAR()
    model.lm_head = EosAfterHead(start)
    conventions = FAMILIES["sdar"].__class__(
        logits_alignment="none", attention_mask="block_causal",
        loader="auto_model_for_causal_lm", block_length=4,
    )
    provider = MaskedDiffusionProvider(model, Tok(), torch.device("cpu"), conventions)
    prompt = torch.arange(8, dtype=torch.long).unsqueeze(0) % 5
    config = GenerationConfig(
        method="fixed", continuation_length=12, fixed_k=1,
        max_candidates=4, verifier_microbatch_size=2, early_stop=early_stop,
    )
    return model, generate_blockwise(provider, prompt, config)


def test_block_loop_stops_early_with_the_same_answer() -> None:
    # Prompt 8 = blocks 0-1; canvas 12 = blocks 2-4 (positions 8..19). EOS from
    # position 13 on, i.e. canvas index 5, inside block 3.
    full_model, full = run_eos(early_stop=False)
    early_model, early = run_eos(early_stop=True)
    assert early.continuation_token_ids == full.continuation_token_ids
    assert early.eos_position == full.eos_position == 5
    assert early.early_stopped and not full.early_stopped
    assert early.denoising_steps < full.denoising_steps
    # Block 4 is never opened, and the stopping block is never committed.
    assert full.commit_passes == 4          # prefill + blocks 2, 3, 4
    assert early.commit_passes == 2         # prefill + block 2
    assert all(entry["first_position"] < 16 for entry in early_model.log)


def test_block_loop_does_not_stop_while_a_position_before_eos_is_masked() -> None:
    # Confidence falls along each block, so position 12 (canvas index 4) is
    # committed before 13 -- but make EOS start at 12 and check the rule the
    # other way: the stop must be at the first EOS, with nothing masked before.
    _, early = run_eos(early_stop=True, start=12)
    assert early.eos_position == 4
    assert early.continuation_token_ids == [8 % 5, 9 % 5, 10 % 5, 11 % 5]


def test_vocab_limit_slices_every_read_path() -> None:
    _, provider = build(block_length=4)
    provider.vocab_limit = VOCAB - 1
    hidden = torch.zeros(1, 4, HIDDEN)
    rows, positions = torch.tensor([0]), torch.tensor([1])
    # the lean path (scheduler) and the block wrapper's path both go through project()
    assert provider.logits_from_hidden(hidden, rows, positions).shape[-1] == VOCAB - 1
    assert provider.project(hidden[0]).shape == (4, VOCAB - 1)
    # the full-logits path
    assert provider.limit_vocab(torch.zeros(1, 4, VOCAB)).shape[-1] == VOCAB - 1


def test_a_stop_token_outside_the_vocab_limit_is_refused() -> None:
    model = FakeSDAR()
    with pytest.raises(ValueError, match="outside vocab_limit"):
        MaskedDiffusionProvider(
            model, Tok(), torch.device("cpu"), FAMILIES["sdar"],
            vocab_limit=EOS, stop_token_ids=(EOS,),
        )


def test_declared_stop_tokens_come_from_generation_config() -> None:
    from adaptive_re.model import declared_stop_token_ids
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[151645, 151643]))
    tok = SimpleNamespace(eos_token_id=151643)
    assert declared_stop_token_ids(model, tok) == (151643, 151645)
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=151643))
    assert declared_stop_token_ids(model, tok) == (151643,)
