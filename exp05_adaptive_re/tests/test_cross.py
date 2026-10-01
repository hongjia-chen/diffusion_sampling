"""SDAR-proposes / Dream-verifies, against fakes whose preferences are set by hand.

The SDAR fake is the block-diffusion fake from test_blockwise: it predicts
`absolute position % 5`, most confident early in each block. The Dream fake is
a full-canvas lean-path provider whose preferred token at each canvas index is
chosen per test, so each rule -- acceptance, veto, early stop, remasking -- can
be switched on in isolation.
"""
from __future__ import annotations

import math

import pytest
import torch

from adaptive_re.cross import CrossConfig, dream_verify, generate_cross, remask_pass
from adaptive_re.model import FAMILIES, MaskedDiffusionProvider
from tests.test_blockwise import EOS, VOCAB, EosAfterHead, FakeSDAR, Tok

DREAM_MASK = 9          # outside the shared range, like Dream's real 151666
SDAR_PROMPT = 8


class FakeDream:
    """Dream stand-in. `preferred(k, row)` picks the token at canvas index k."""

    def __init__(self, prompt_length: int, preferred, coupled: bool = False):
        self.prompt_length = prompt_length
        self.preferred = preferred
        self.coupled = coupled
        self.mask_token_id = DREAM_MASK
        self.eos_token_id = EOS
        self.excluded_token_ids = {DREAM_MASK}
        self.device = torch.device("cpu")
        self.vocab_limit = VOCAB
        self.supports_lean_logits = True
        self.forwards: list[int] = []      # rows per forward

    def hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.forwards.append(int(input_ids.shape[0]))
        return input_ids.unsqueeze(-1).float()

    def logits_from_hidden(self, hidden, rows, positions) -> torch.Tensor:
        out = torch.full((int(rows.numel()), VOCAB), -9.0)
        for i, (row, position) in enumerate(zip(rows.tolist(), positions.tolist())):
            sequence = hidden[row, :, 0].long()
            canvas = sequence[self.prompt_length:]
            k = position - self.prompt_length
            target = self.preferred(k, canvas)
            out[i, target] = 5.0
            if self.coupled and k > 0 and int(canvas[k - 1]) != DREAM_MASK:
                # A filled left neighbour makes Dream less sure (p ~ 0.62 on
                # its target, still its argmax): the joint departs from the
                # product of marginals without changing Dream's preference.
                out[i, target] = 1.0
                out[i, (target + 1) % 5] = 0.5
        return out


def sdar_target(k: int) -> int:
    return (k + SDAR_PROMPT) % 5


def build_sdar(vocab_limit=VOCAB, eos_from=None):
    conventions = FAMILIES["sdar"].__class__(
        logits_alignment="none", attention_mask="block_causal",
        loader="auto_model_for_causal_lm", block_length=4,
    )
    model = FakeSDAR()
    if eos_from is not None:
        model.lm_head = EosAfterHead(SDAR_PROMPT + eos_from)
    return MaskedDiffusionProvider(
        model, Tok(), torch.device("cpu"), conventions,
        vocab_limit=vocab_limit, stop_token_ids=(EOS,),
    )


def run(preferred, *, generated=8, delta=10.0, remask=False, veto=True,
        coupled=False, dream_prompt=5, group_rows=4, traces=False, sdar_eos_from=None):
    sdar = build_sdar(eos_from=sdar_eos_from)
    dream = FakeDream(dream_prompt, preferred, coupled=coupled)
    config = CrossConfig(
        method="adaptive", delta=delta, continuation_length=generated,
        max_candidates=4, remask=remask, veto=veto, group_rows=group_rows,
    )
    sdar_prompt = torch.arange(SDAR_PROMPT, dtype=torch.long).unsqueeze(0) % 5
    dream_prompt_ids = torch.zeros(1, dream_prompt, dtype=torch.long)
    result = generate_cross(sdar, dream, sdar_prompt, dream_prompt_ids, config,
                            capture_traces=traces)
    return result, dream


def agree(k, canvas):
    return sdar_target(k)


def test_agreement_accepts_whole_blocks() -> None:
    result, _ = run(agree)
    assert result.continuation_token_ids == [sdar_target(k) for k in range(8)]
    # Independent and in agreement: every block goes in one step.
    assert result.denoising_steps == 2
    assert result.extras["veto_count"] == 0


def test_dream_coordinates_are_independent_of_the_prompt_lengths() -> None:
    for dream_prompt in (3, 5, 11):
        result, _ = run(agree, dream_prompt=dream_prompt)
        assert result.continuation_token_ids == [sdar_target(k) for k in range(8)]


def test_veto_commits_dreams_token_alone() -> None:
    def disagree(k, canvas):
        return (sdar_target(k) + 1) % 5

    result, _ = run(disagree)
    assert result.continuation_token_ids == [disagree(k, None) for k in range(8)]
    assert result.denoising_steps == 8
    assert result.extras["veto_count"] == 8
    assert result.commitment_histogram[0] == 8


def test_without_veto_sdars_tokens_stand() -> None:
    def disagree(k, canvas):
        return (sdar_target(k) + 1) % 5

    result, _ = run(disagree, veto=False)
    assert result.continuation_token_ids == [sdar_target(k) for k in range(8)]
    assert result.extras["veto_count"] == 0


def test_dependence_limits_the_accepted_prefix() -> None:
    loose, _ = run(agree, coupled=True, delta=100.0)
    tight, _ = run(agree, coupled=True, delta=0.01)
    assert loose.denoising_steps < tight.denoising_steps
    assert loose.continuation_token_ids == tight.continuation_token_ids


def test_group_size_changes_cost_but_not_the_answer() -> None:
    one, dream_one = run(agree, coupled=True, delta=0.5, group_rows=1)
    four, dream_four = run(agree, coupled=True, delta=0.5, group_rows=4)
    assert one.continuation_token_ids == four.continuation_token_ids
    assert one.commitment_histogram == four.commitment_histogram
    assert max(dream_one.forwards) == 1 and max(dream_four.forwards) <= 4


def test_early_stop_on_a_committed_stop_token() -> None:
    def eos_at_5(k, canvas):
        return EOS if k >= 5 else sdar_target(k)

    # Both models agree the answer ends at canvas index 5.
    result, _ = run(eos_at_5, generated=12, sdar_eos_from=5)
    assert result.eos_position == 5
    assert result.continuation_token_ids == [sdar_target(k) for k in range(5)]
    assert result.early_stopped
    # Prompt 8 = 2 blocks (1 prefill commit); canvas indices 0-3 are block 2
    # (committed), and the stop lands in block 3, which is never committed.
    assert result.commit_passes == 2


def test_remask_replaces_one_token_dream_rejects_in_hindsight() -> None:
    # Dream agrees while the block is being filled, but once every other
    # position of the block is visible it wants a different token at canvas
    # index 2 (SDAR's is sdar_target(2) == 0).
    replacement = 2

    def hindsight(k, canvas):
        others_filled = all(
            int(canvas[j]) != DREAM_MASK for j in range(4) if j != k
        )
        if k == 2 and others_filled:
            return replacement
        return sdar_target(k)

    plain, _ = run(hindsight, remask=False)
    fixed, _ = run(hindsight, remask=True, traces=True)
    assert plain.continuation_token_ids[2] == sdar_target(2)
    assert fixed.continuation_token_ids[2] == replacement
    others = [i for i in range(8) if i != 2]
    assert [fixed.continuation_token_ids[i] for i in others] == \
        [plain.continuation_token_ids[i] for i in others]
    assert fixed.extras["tokens_changed_by_remask"] == 1
    assert fixed.extras["remask_passes"] == 2           # once per block
    remasks = [t for t in fixed.traces if t.get("remask")]
    assert remasks[0]["chosen"] == 2 and remasks[0]["new_token"] == replacement


def test_remask_skips_when_dream_would_change_nothing() -> None:
    result, _ = run(agree, remask=True)
    assert result.extras["remask_passes"] == 2
    assert result.extras["remask_skipped"] == 2
    assert result.extras["tokens_changed_by_remask"] == 0


def test_mismatched_vocab_limits_are_refused() -> None:
    sdar = build_sdar(vocab_limit=VOCAB)
    dream = FakeDream(5, agree)
    dream.vocab_limit = VOCAB + 1
    config = CrossConfig(method="adaptive", delta=1.0, continuation_length=8, max_candidates=4)
    with pytest.raises(ValueError, match="vocab_limit"):
        generate_cross(sdar, dream, torch.zeros(1, 8, dtype=torch.long),
                       torch.zeros(1, 5, dtype=torch.long), config)


def test_dream_verify_matches_the_prefix_rule() -> None:
    dream = FakeDream(2, lambda k, canvas: 3, coupled=True)
    base = torch.tensor([[0, 0] + [DREAM_MASK] * 6])
    positions = torch.tensor([2, 3, 4, 5])
    tokens = torch.tensor([3, 3, 3, 3])
    verdict = dream_verify(dream, base, positions, tokens,
                           tau_nats=math.log1p(0.5), group_rows=4, veto=True)
    # Marginals are all confident; every conditional after the first sees a
    # filled left neighbour and drops, so the error grows from candidate 2.
    assert not verdict.vetoed
    assert verdict.accepted < 4
    assert verdict.errors[0] == 0


def test_remask_draw_is_weighted_by_inverse_probability() -> None:
    # Two survivors whose committed token has p = 0.4 and p = 0.1, Dream
    # preferring token 4 at both: 1/p weights 2.5 : 10, so the p = 0.1 one is
    # drawn 80% of the time.
    class Scripted(FakeDream):
        def logits_from_hidden(self, hidden, rows, positions):
            out = torch.full((int(rows.numel()), VOCAB), -60.0)
            for i, position in enumerate(positions.tolist()):
                p = 0.4 if position == 2 else 0.1
                out[i, 1] = math.log(p)          # the committed token
                out[i, 4] = math.log(1 - p)      # Dream's argmax
            return out

    dream = Scripted(2, None)
    base = torch.tensor([[0, 0, 1, 1]])
    picks = []
    torch.manual_seed(0)
    for _ in range(400):
        out = remask_pass(dream, base, torch.tensor([0, 1]), torch.tensor([2, 3]),
                          torch.tensor([1, 1]), bottom=4, group_rows=4)
        picks.append(out.chosen)
    share_low = picks.count(1) / len(picks)
    assert 0.74 < share_low < 0.86


def test_the_cli_variant_is_remask_without_veto() -> None:
    import sys
    sys.path.insert(0, "scripts")
    from run_benchmark import parse_method
    sampling = {"fixed": {}, "adaptive": {"temperature": 0.0, "top_p": None, "confidence": "logp"}}
    config = parse_method("xadaptive-remask:10.00", 256, 4, max_candidates=64, sampling=sampling)
    assert config.remask and not config.veto
    assert config.label == "xadaptive_remask_noveto_delta_10p00"
    vetoed = parse_method("xadaptive-remask:0.30", 256, 4, max_candidates=64,
                          sampling=sampling, veto=True)
    assert vetoed.veto and vetoed.label == "xadaptive_remask_delta_0p30"



class StopScripted(FakeDream):
    """Dream wants <|endoftext|> (EOS) where a different stop token (7) sits."""

    OTHER_STOP = 3

    def __init__(self, replace_with):
        super().__init__(2, None)
        self.replace_with = replace_with

    def logits_from_hidden(self, hidden, rows, positions):
        out = torch.full((int(rows.numel()), VOCAB), -30.0)
        for i, position in enumerate(positions.tolist()):
            if position == 3:                      # the stop position
                out[i, self.replace_with] = 0.0
                out[i, self.OTHER_STOP] = -3.0     # the committed stop token: p ~ 0.05
            else:
                out[i, 1] = 0.0                    # content: Dream agrees
        return out


def test_swapping_one_stop_token_for_another_is_not_a_change() -> None:
    dream = StopScripted(replace_with=EOS)
    out = remask_pass(dream, torch.tensor([[0, 0, 1, 3]]), torch.tensor([0, 1]),
                      torch.tensor([2, 3]), torch.tensor([1, 3]), bottom=4, group_rows=4,
                      stop_ids=frozenset({EOS, StopScripted.OTHER_STOP}))
    assert out.chosen is None and out.survivors == []
    # scored by the whole stop set, so the stop position is not "improbable"
    assert out.log_probs[1] > -0.1


def test_replacing_a_stop_token_with_content_is_still_a_change() -> None:
    dream = StopScripted(replace_with=2)       # Dream wants the answer to go on
    out = remask_pass(dream, torch.tensor([[0, 0, 1, 3]]), torch.tensor([0, 1]),
                      torch.tensor([2, 3]), torch.tensor([1, 3]), bottom=4, group_rows=4,
                      stop_ids=frozenset({EOS, StopScripted.OTHER_STOP}))
    assert out.chosen == 1 and out.new_token == 2
