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

from sdar_harness.cross import DREAM_GROUP_ROWS, CrossConfig, dream_verify, generate_cross, remask_pass
from sdar_harness.model import FAMILIES, MaskedDiffusionProvider
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


def build_sdar(vocab_limit=VOCAB, eos_from=None, head=None):
    conventions = FAMILIES["sdar"].__class__(
        logits_alignment="none", attention_mask="block_causal",
        loader="auto_model_for_causal_lm", block_length=4,
    )
    model = FakeSDAR()
    if eos_from is not None:
        model.lm_head = EosAfterHead(SDAR_PROMPT + eos_from)
    if head is not None:
        model.lm_head = head
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


def test_step_counters_say_what_bounded_each_step() -> None:
    whole, _ = run(agree)
    # Agreement passes all 4 candidates of each block: the block is the bound.
    assert whole.extras["all_candidates_accepted_steps"] == 2
    assert whole.extras["lone_candidate_steps"] == 0
    assert whole.extras["candidates_offered"] == 8
    tight, _ = run(agree, coupled=True, delta=0.01, veto=False, traces=True)
    steps = [t for t in tight.traces if not t.get("remask")]
    assert tight.extras["candidates_offered"] == sum(t["remaining_masks_before"] for t in steps)
    assert tight.extras["lone_candidate_steps"] == sum(
        t["remaining_masks_before"] == 1 for t in steps)
    assert tight.extras["all_candidates_accepted_steps"] == sum(
        1 < t["remaining_masks_before"] == t["accepted_prefix_length"] for t in steps)
    assert tight.extras["lone_candidate_steps"] > 0


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


def test_a_lone_candidate_skips_the_verifier_unless_it_can_veto() -> None:
    dream = FakeDream(2, lambda k, canvas: 4)
    base = torch.tensor([[0, 0] + [DREAM_MASK] * 6])
    positions, tokens = torch.tensor([2]), torch.tensor([3])
    verdict = dream_verify(dream, base, positions, tokens,
                           tau_nats=0.0, group_rows=4, veto=False)
    assert verdict.accepted == 1 and verdict.tokens.tolist() == [3]
    assert verdict.passes == 0 and verdict.rows == 0 and dream.forwards == []
    # A veto can overrule it, so then the forward still runs.
    vetoed = dream_verify(dream, base, positions, tokens,
                          tau_nats=0.0, group_rows=4, veto=True)
    assert vetoed.vetoed and vetoed.tokens.tolist() == [4] and dream.forwards == [1]


def test_skipping_lone_candidates_leaves_the_answer_unchanged() -> None:
    # delta 0 with coupling commits few tokens per step, so blocks end on a lone slot.
    result, dream = run(agree, delta=0.0, veto=False, coupled=True, traces=True)
    assert result.continuation_token_ids == [sdar_target(k) for k in range(8)]
    lone = [t for t in result.traces if t["remaining_masks_before"] == 1]
    rest = [t for t in result.traces if t["remaining_masks_before"] > 1]
    assert lone and all(t["dream_rows"] == 0 for t in lone)
    assert rest and all(t["dream_rows"] > 0 for t in rest)
    assert 1 not in dream.forwards
    assert result.verifier_forward_passes == len(rest)


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
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_benchmark import parse_method
    sampling = {"fixed": {}, "adaptive": {"temperature": 0.0, "top_p": None, "confidence": "logp"}}
    config = parse_method("xadaptive-remask:10.00", 256, 4, max_candidates=64, sampling=sampling)
    assert config.remask and not config.veto
    assert config.label == "xadaptive_remask_noveto_delta_10p00"
    vetoed = parse_method("xadaptive-remask:0.30", 256, 4, max_candidates=64,
                          sampling=sampling, veto=True)
    assert vetoed.veto and vetoed.label == "xadaptive_remask_delta_0p30"


def test_the_cli_group_rows_reaches_the_mi_rule() -> None:
    # Regression (2026-10-15): launch_mi.sh's group size went only to
    # --verifier-microbatch-size, which CrossConfig never reads, so every MI
    # run scored DREAM_GROUP_ROWS rows per call whatever it asked for.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_benchmark import parse_method
    sampling = {"fixed": {}, "adaptive": {"temperature": 0.0, "top_p": None, "confidence": "logp"}}
    default = parse_method("xadaptive:0.10", 256, 8, max_candidates=32, sampling=sampling)
    assert default.group_rows == DREAM_GROUP_ROWS == 4
    eight = parse_method("xadaptive:0.10", 256, 4, max_candidates=32, sampling=sampling, group_rows=8)
    assert eight.group_rows == 8


def test_a_wide_delta_grid_gives_every_arm_its_own_label() -> None:
    grid = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 1e3, 1e4, 1e6, 1e9, 1e12]
    labels = [CrossConfig(method="adaptive", delta=float(d), continuation_length=8,
                          max_candidates=4).label for d in grid]
    assert len(set(labels)) == len(grid)
    # The values earlier runs used keep their names.
    assert labels[0].endswith("_delta_0p00") and labels[5].endswith("_delta_0p10")
    assert labels[9].endswith("_delta_10p00") and labels[12].endswith("_delta_1000p00")
    assert labels[1].endswith("_delta_0p001") and labels[14].endswith("_delta_1e06")



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


# --- Block verifier (exp06: SDAR-8B verifies SDAR-1.7B) --------------------


class DisagreeAtHead(torch.nn.Module):
    """The fake SDAR head, except it prefers another token at one absolute position."""

    def __init__(self, position: int):
        super().__init__()
        from tests.test_blockwise import FakeHead
        self.base = FakeHead()
        self.position = position

    def forward(self, hidden):
        logits = self.base(hidden)
        at = hidden[..., 0].long() == self.position
        target = (self.position % 5 + 1) % 5
        logits[at] = -9.0
        logits[at, target] = 6.0
        return logits


def run_block_verifier(*, verifier_head=None, remask=False, generated=8, prompt=None):
    sdar = build_sdar()
    verifier = build_sdar(head=verifier_head)
    config = CrossConfig(
        method="adaptive", delta=1.0, continuation_length=generated,
        max_candidates=4, remask=remask, veto=False,
    )
    sdar_prompt = torch.arange(SDAR_PROMPT, dtype=torch.long).unsqueeze(0) % 5
    verifier_prompt = sdar_prompt if prompt is None else prompt
    result = generate_cross(sdar, verifier, sdar_prompt, verifier_prompt, config)
    return result, verifier


def test_block_verifier_agreeing_with_the_proposer_accepts_whole_blocks() -> None:
    result, verifier = run_block_verifier(remask=True)
    assert result.continuation_token_ids == [sdar_target(k) for k in range(8)]
    assert result.denoising_steps == 2
    assert result.extras["tokens_changed_by_remask"] == 0
    log = verifier.model.log
    # Its own cache: the prompt's 2 blocks, then each generated block.
    commits = [entry for entry in log if entry["store_kv"]]
    assert [entry["prefix"] for entry in commits] == [0, 8, 12]
    # Verification and remask rows are one block wide, read against that cache.
    reads = [entry for entry in log if not entry["store_kv"]]
    assert reads and all(entry["width"] == 4 and entry["prefix"] in (8, 12) for entry in reads)


def test_block_verifier_remasks_the_token_it_disagrees_with() -> None:
    # Absolute position 10 is canvas index 2; SDAR commits 10 % 5 = 0 there.
    head = DisagreeAtHead(10)
    plain, _ = run_block_verifier(verifier_head=head, remask=False)
    fixed, _ = run_block_verifier(verifier_head=head, remask=True)
    assert plain.continuation_token_ids[2] == sdar_target(2)
    assert fixed.continuation_token_ids[2] == 1
    assert fixed.extras["tokens_changed_by_remask"] == 1


def test_block_verifier_refuses_a_different_prompt() -> None:
    other = torch.zeros(1, SDAR_PROMPT, dtype=torch.long)
    with pytest.raises(ValueError, match="prompt"):
        run_block_verifier(prompt=other)


def test_mi_profiling_names_every_stage_and_leaves_the_answer_alone() -> None:
    from sdar_harness.timing import StageTimer
    plain, _ = run(agree, coupled=True, delta=0.01, veto=False)
    sdar = build_sdar()
    dream = FakeDream(5, agree, coupled=True)
    config = CrossConfig(
        method="adaptive", delta=0.01, continuation_length=8,
        max_candidates=4, remask=False, veto=False, group_rows=4,
    )
    sdar_prompt = torch.arange(SDAR_PROMPT, dtype=torch.long).unsqueeze(0) % 5
    timed = generate_cross(
        sdar, dream, sdar_prompt, torch.zeros(1, 5, dtype=torch.long), config,
        timer=StageTimer(torch.device("cpu"), enabled=True),
    )
    assert timed.continuation_token_ids == plain.continuation_token_ids
    assert plain.stage_seconds == {} and plain.stage_counts == {}
    counts = timed.stage_counts
    assert set(timed.stage_seconds) == set(counts)
    steps = timed.denoising_steps
    for stage in ("propose_forward", "propose_rank", "verify_test", "decide", "stop_check"):
        assert counts[stage] == steps, stage
    # Lone candidates skip the verifier, so its calls are the passes it made.
    assert counts.get("verify_forward", 0) == timed.verifier_forward_passes
    assert counts["prefill"] == 1 and counts["commit"] == timed.commit_passes - 1


# ---------------------------------------------------------------- prefix plus one
# The coupled fake drops Dream's confidence at canvas index k once k-1 is
# filled. With candidates ranked k = 0, 1, 2, 3, rank 1 fails given rank 0, but
# rank 2's left neighbour is still masked in row 1, so {0, 2} is independent.

def coupled_verdicts(**kwargs):
    base = torch.tensor([[0, 0] + [DREAM_MASK] * 6])
    positions, tokens = torch.tensor([2, 3, 4, 5]), torch.tensor([3, 3, 3, 3])
    out = {}
    for plus_one in (False, True):
        dream = FakeDream(2, lambda k, canvas: 3, coupled=True)
        out[plus_one] = (dream_verify(dream, base, positions, tokens, tau_nats=math.log1p(0.5),
                                      group_rows=4, veto=False, plus_one=plus_one, **kwargs), dream)
    return out


def test_plus_one_adds_a_later_candidate_independent_of_the_prefix() -> None:
    (prefix, _), (plus, dream) = coupled_verdicts().values()
    assert prefix.accepted == prefix.prefix == 1 and prefix.committed.tolist() == [0]
    assert plus.prefix == 1 and plus.extra == 2 and plus.accepted == 2
    assert plus.committed.tolist() == [0, 2] and plus.tokens.tolist() == [3, 3]
    # Read from row 1, already run: no extra forward, and the prefix test saw the same numbers.
    assert dream.forwards == [4] and plus.passes == prefix.passes == 1
    assert torch.equal(plus.errors, prefix.errors)
    # Ranks 2 and 3 both pass given rank 0 with the same log-ratio (0), so the
    # tie goes to the more confident.
    assert plus.later_errors.tolist() == [0.0, 0.0]
    assert plus.extra_passing == 2 and plus.extra_first == 2


def test_plus_one_tests_dependence_not_agreement() -> None:
    base = torch.tensor([[0, 0] + [DREAM_MASK] * 6])
    positions, tokens = torch.tensor([2, 3, 4, 5]), torch.tensor([3, 3, 3, 3])
    dream = FakeDream(2, lambda k, canvas: 3 if k != 2 else 4, coupled=True)
    verdict = dream_verify(dream, base, positions, tokens, tau_nats=math.log1p(0.5),
                           group_rows=4, veto=False, plus_one=True)
    # Dream rejects rank 2's token outright (log p ~ -14 on its own), but no
    # less so given rank 0, so the set's log-ratio is ~0 and the extra goes in.
    # Like the prefix rule, plus one measures dependence, not agreement; only
    # the veto ablation acts on disagreement.
    assert verdict.extra == 2 and verdict.marginal[2] < -10


def test_plus_one_saves_a_step_on_the_coupled_fake() -> None:
    def run_rule(rule):
        sdar = build_sdar()
        dream = FakeDream(5, agree, coupled=True)
        config = CrossConfig(method="adaptive", delta=0.01, continuation_length=8,
                             max_candidates=4, veto=False, subset_rule=rule)
        prompt = torch.arange(SDAR_PROMPT, dtype=torch.long).unsqueeze(0) % 5
        return generate_cross(sdar, dream, prompt, torch.zeros(1, 5, dtype=torch.long), config,
                              capture_traces=True)

    prefix, plus = run_rule("prefix"), run_rule("prefix_plus_one")
    assert plus.continuation_token_ids == prefix.continuation_token_ids == [sdar_target(k) for k in range(8)]
    # The SDAR fake ranks the second block's slots k = 6, 7, 4, 5. Rank 1 (k7)
    # leans on rank 0 (k6), so the prefix stops at k6; given k6 alone, k4 is as
    # likely as on its own, so plus one commits {k6, k4} and then {k7, k5}: two
    # steps where the prefix rule takes three. The first block's step tries an
    # extra too (k3 given {k0, k2}), but k3 leans on k2 and fails.
    assert prefix.denoising_steps == 5 and plus.denoising_steps == 4
    assert plus.commitment_histogram == [0, 4, 0, 0]
    assert plus.extras["extra_tried_steps"] == 2 and plus.extras["extra_committed_steps"] == 1
    assert plus.extras["extra_choice_steps"] == 1 and plus.extras["extra_not_most_confident_steps"] == 0
    assert "extra_committed_steps" not in prefix.extras
    tried, added = plus.traces[0], plus.traces[2]
    assert tried["extra_rank"] == -1 and tried["committed_positions"] == [0, 2]
    assert added["accepted_prefix_length"] == 1 and added["extra_rank"] == 2
    assert added["committed_positions"] == [6, 4]
    assert "extra_rank" not in prefix.traces[0]


class Leaning(FakeDream):
    """Every slot wants token 3. Given k0, k1 leans on it hard (p 0.27) and k2 a
    little (p 0.88); k3 never does."""

    def logits_from_hidden(self, hidden, rows, positions) -> torch.Tensor:
        out = torch.full((int(rows.numel()), VOCAB), -9.0)
        for i, (row, position) in enumerate(zip(rows.tolist(), positions.tolist())):
            canvas = hidden[row, :, 0].long()[self.prompt_length:]
            k, head = position - self.prompt_length, int(canvas[0]) != DREAM_MASK
            if k == 1 and head:
                out[i, 3], out[i, 4] = 1.0, 2.0
            elif k == 2 and head:
                out[i, 3], out[i, 4] = 3.0, 1.0
            else:
                out[i, 3] = 5.0
        return out


def test_plus_one_takes_the_smallest_log_ratio_not_the_most_confident() -> None:
    dream = Leaning(2, None)
    base = torch.tensor([[0, 0] + [DREAM_MASK] * 6])
    verdict = dream_verify(dream, base, torch.tensor([2, 3, 4, 5]), torch.tensor([3, 3, 3, 3]),
                           tau_nats=math.log1p(0.5), group_rows=4, veto=False, plus_one=True)
    # Rank 1 fails (log-ratio ~1.3); ranks 2 (~0.13) and 3 (~0) both pass.
    assert verdict.prefix == 1 and verdict.extra_passing == 2
    assert verdict.later_errors[0] == pytest.approx(0.127, abs=0.01) and verdict.later_errors[1] < 1e-4
    assert verdict.extra_first == 2 and verdict.extra == 3
    assert verdict.committed.tolist() == [0, 3]


def test_the_cli_plus1_variant_and_its_label() -> None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_benchmark import parse_method
    sampling = {"fixed": {}, "adaptive": {"temperature": 0.0, "top_p": None, "confidence": "logp"}}
    plain = parse_method("xadaptive:0.10", 256, 8, max_candidates=16, sampling=sampling)
    assert plain.subset_rule == "prefix" and plain.label == "xadaptive_noveto_delta_0p10"
    plus = parse_method("xadaptive-plus1:0.10", 256, 8, max_candidates=16, sampling=sampling)
    assert plus.subset_rule == "prefix_plus_one" and plus.label == "xadaptive_plus1_noveto_delta_0p10"
    both = parse_method("xadaptive-remask-plus1:0.10", 256, 8, max_candidates=16, sampling=sampling)
    assert both.remask and both.label == "xadaptive_remask_plus1_noveto_delta_0p10"
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        parse_method("fixed-plus1:1", 256, 8, max_candidates=16, sampling=sampling)
    with pytest.raises(ValueError):
        CrossConfig(method="adaptive", delta=0.1, continuation_length=8, max_candidates=4, subset_rule="all")


def test_the_confidence_floor_keeps_low_confidence_candidates_out_of_the_extra() -> None:
    base = torch.tensor([[0, 0] + [DREAM_MASK] * 6])
    positions, tokens = torch.tensor([2, 3, 4, 5]), torch.tensor([3, 3, 3, 3])

    def verdict(floor):
        return dream_verify(Leaning(2, None), base, positions, tokens, tau_nats=math.log1p(0.5),
                            group_rows=4, veto=False, plus_one=True,
                            confidences=torch.tensor([0.99, 0.98, 0.95, 0.50]), min_confidence=floor)

    # Unfloored, rank 3 has the smallest log-ratio; at 0.50 the floor rules it out.
    assert verdict(0.0).extra == 3
    floored = verdict(0.9)
    assert floored.extra == 2 and floored.extra_passing == 1 and floored.committed.tolist() == [0, 2]
    # Nothing clears 0.99 past the prefix: the prefix alone is committed.
    none = verdict(0.99)
    assert none.extra == -1 and none.extra_passing == 0 and none.accepted == 1


def test_the_floor_reaches_generate_cross() -> None:
    def run_floor(floor):
        config = CrossConfig(method="adaptive", delta=0.01, continuation_length=8, max_candidates=4,
                             veto=False, subset_rule="prefix_plus_one", extra_min_confidence=floor)
        prompt = torch.arange(SDAR_PROMPT, dtype=torch.long).unsqueeze(0) % 5
        return generate_cross(build_sdar(), FakeDream(5, agree, coupled=True), prompt,
                              torch.zeros(1, 5, dtype=torch.long), config)

    # A floor no candidate can miss behaves as no floor (4 steps); one none can
    # reach leaves the prefix rule (5 steps, as in the plus-one test above).
    assert run_floor(1e-9).denoising_steps == 4
    assert run_floor(1.0).denoising_steps == 5 and run_floor(1.0).extras["extra_committed_steps"] == 0


def test_the_cli_floor_variant_and_its_label() -> None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_benchmark import parse_method
    sampling = {"fixed": {}, "adaptive": {"temperature": 0.0, "top_p": None, "confidence": "logp"}}
    floored = parse_method("xadaptive-plus1c90:0.10", 256, 8, max_candidates=16, sampling=sampling)
    assert floored.extra_min_confidence == 0.9 and floored.subset_rule == "prefix_plus_one"
    assert floored.label == "xadaptive_plus1_c90_noveto_delta_0p10"
    plain = parse_method("xadaptive-plus1:0.10", 256, 8, max_candidates=16, sampling=sampling)
    assert plain.extra_min_confidence == 0.0 and plain.label == "xadaptive_plus1_noveto_delta_0p10"
    with pytest.raises(ValueError):
        CrossConfig(method="adaptive", delta=0.1, continuation_length=8, max_candidates=4,
                    extra_min_confidence=0.9)


# ---------------------------------------------------------------- one-sided test
class Scripted2(FakeDream):
    """Every slot wants token 3. `effect[k]` says what a filled left neighbour
    does to slot k: "support" lifts it from p ~0.62 to ~1, "conflict" drops it
    from ~1 to ~0.62, None leaves it at ~1."""

    def __init__(self, prompt_length, effect):
        super().__init__(prompt_length, None)
        self.effect = effect

    def logits_from_hidden(self, hidden, rows, positions) -> torch.Tensor:
        out = torch.full((int(rows.numel()), VOCAB), -9.0)
        for i, (row, position) in enumerate(zip(rows.tolist(), positions.tolist())):
            canvas = hidden[row, :, 0].long()[self.prompt_length:]
            k = position - self.prompt_length
            left = k > 0 and int(canvas[k - 1]) != DREAM_MASK
            effect = self.effect.get(k)
            weak = (effect == "support" and not left) or (effect == "conflict" and left)
            if weak:
                out[i, 3], out[i, 4] = 1.0, 0.5
            else:
                out[i, 3] = 5.0
        return out


def verify_sided(effect, one_sided):
    base = torch.tensor([[0, 0] + [DREAM_MASK] * 6])
    return dream_verify(Scripted2(2, effect), base, torch.tensor([2, 3, 4, 5]), torch.tensor([3, 3, 3, 3]),
                        tau_nats=math.log1p(0.5), group_rows=4, veto=False, one_sided=one_sided)


def test_one_sided_lets_supportive_dependence_through() -> None:
    support = {1: "support", 2: "support", 3: "support"}
    # Each filled neighbour lifts the next slot by ~+0.47 nats: two-sided stops
    # at rank 1, one-sided takes the whole block.
    assert verify_sided(support, one_sided=False).accepted == 1
    one = verify_sided(support, one_sided=True)
    assert one.accepted == 4 and one.passes == 1
    # The trace quantity is the same two-sided |cumulative log-ratio| either way.
    assert torch.equal(one.errors, verify_sided(support, one_sided=False).errors)


def test_one_sided_still_stops_at_conflict() -> None:
    conflict = {1: "conflict", 2: "conflict", 3: "conflict"}
    assert verify_sided(conflict, one_sided=False).accepted == 1
    assert verify_sided(conflict, one_sided=True).accepted == 1


def test_one_sided_is_per_token_so_support_cannot_pay_for_conflict() -> None:
    # Rank 1 is supported (+0.47), rank 2 conflicts (-0.47): their cumulative
    # log-ratio is ~0, but rank 2 alone is a conflict and stops the prefix.
    mixed = {1: "support", 2: "conflict"}
    one = verify_sided(mixed, one_sided=True)
    assert one.accepted == 2
    assert one.errors[2] < 1e-4


def test_one_sided_through_generate_cross_and_the_cli() -> None:
    def run_test(pmi_test):
        config = CrossConfig(method="adaptive", delta=0.5, continuation_length=8, max_candidates=4,
                             veto=False, pmi_test=pmi_test)
        dream = Scripted2(5, {k: "support" for k in range(1, 8)})
        prompt = torch.arange(SDAR_PROMPT, dtype=torch.long).unsqueeze(0) % 5
        return generate_cross(build_sdar(), dream, prompt, torch.zeros(1, 5, dtype=torch.long), config)

    two, one = run_test("two_sided"), run_test("one_sided")
    assert one.continuation_token_ids == two.continuation_token_ids
    assert one.denoising_steps < two.denoising_steps
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_benchmark import parse_method
    sampling = {"fixed": {}, "adaptive": {"temperature": 0.0, "top_p": None, "confidence": "logp"}}
    os_config = parse_method("xadaptive-os:0.10", 256, 8, max_candidates=16, sampling=sampling)
    assert os_config.pmi_test == "one_sided" and os_config.subset_rule == "prefix"
    assert os_config.label == "xadaptive_os_noveto_delta_0p10"
    assert parse_method("xadaptive:0.10", 256, 8, max_candidates=16, sampling=sampling).pmi_test == "two_sided"
    with pytest.raises(ValueError):
        parse_method("xadaptive-os-plus1:0.10", 256, 8, max_candidates=16, sampling=sampling)


class StraddleHead(torch.nn.Module):
    """Targets `position % 5` like FakeHead, with a set confidence per block slot.
    The default is near-certain on the first two slots and below 0.95 on the
    last two (~0.90 and ~0.88, each against one runner-up)."""

    def __init__(self, peaks=(6.0, 5.9, 2.2, 2.0)):
        super().__init__()
        self.peaks = torch.tensor(peaks)

    def forward(self, hidden):
        position = hidden[..., 0].long()
        logits = torch.full(hidden.shape[:-1] + (VOCAB,), -9.0)
        target = position % 5
        peak = self.peaks[position % 4]
        logits.scatter_(-1, target.unsqueeze(-1), peak.unsqueeze(-1))
        runner = torch.where(peak < 5.0, torch.tensor(0.0), torch.tensor(-9.0))
        logits.scatter_(-1, ((target + 1) % 5).unsqueeze(-1), runner.unsqueeze(-1))
        return logits


def run_dynamic_first(delta, *, threshold=0.95, peaks=(6.0, 5.9, 2.2, 2.0), traces=False):
    config = CrossConfig(method="adaptive", delta=delta, continuation_length=8, max_candidates=4,
                         veto=False, dynamic_threshold=threshold)
    dream = FakeDream(5, agree, coupled=True)
    prompt = torch.arange(SDAR_PROMPT, dtype=torch.long).unsqueeze(0) % 5
    sdar = build_sdar(head=StraddleHead(peaks) if peaks else None)
    result = generate_cross(sdar, dream, prompt, torch.zeros(1, 5, dtype=torch.long), config,
                            capture_traces=traces)
    return result, dream


def test_dynamic_first_commits_the_confident_then_the_mi_rules_first() -> None:
    result, dream = run_dynamic_first(0.0)
    # Per block: slots 0 and 1 (above 0.95) untested, slot 2 as the MI rule's
    # forced first candidate; the coupled verifier fails slot 3 given slot 2,
    # so it follows alone. Dynamic 0.95 would take 2, 1, 1.
    assert result.continuation_token_ids == [sdar_target(k) for k in range(8)]
    assert result.commitment_histogram == [2, 0, 2, 0] and result.denoising_steps == 4
    assert result.extras["dynamic_committed_tokens"] == 4 and result.extras["dynamic_steps"] == 2
    # The first verify scores the two candidates after slots 0 and 1, not four.
    assert dream.forwards[0] == 2


def test_dynamic_first_measures_marginals_on_the_canvas_with_the_dynamic_tokens() -> None:
    result, _ = run_dynamic_first(0.0, traces=True)
    first = result.traces[0]
    assert first["dynamic_committed"] == 2
    assert first["accepted_prefix_length"] == 3
    assert first["committed_positions"] == first["candidate_positions"][:3]
    assert first["marginal_log_factors"][:2] == [None, None]
    assert first["conditional_log_factors"][:2] == [None, None] and first["error_by_prefix_nats"][:2] == [None, None]
    # Slot 2 is the MI rule's first candidate: its conditional is its marginal.
    # Both see slot 1 filled (the coupled verifier is unsure next to a filled
    # neighbour), so the dynamic tokens were on the verifier's canvas.
    assert first["conditional_log_factors"][2] == first["marginal_log_factors"][2] < math.log(0.7)
    assert first["error_by_prefix_nats"][2] == 0
    plain, _ = run_dynamic_first(0.0, threshold=0.0, traces=True)
    assert plain.traces[0]["marginal_log_factors"][2] > math.log(0.99)
    assert "dynamic_committed" not in plain.traces[0]


def test_dynamic_first_without_a_confident_candidate_is_the_mi_rule() -> None:
    for delta in (0.0, 0.5):
        peaks = (2.6, 2.4, 2.2, 2.0)
        dyn, dyn_dream = run_dynamic_first(delta, peaks=peaks, traces=True)
        plain, plain_dream = run_dynamic_first(delta, threshold=0.0, peaks=peaks, traces=True)
        assert dyn.continuation_token_ids == plain.continuation_token_ids
        assert dyn.commitment_histogram == plain.commitment_histogram
        assert dyn_dream.forwards == plain_dream.forwards
        assert dyn.extras["dynamic_committed_tokens"] == 0
        for a, b in zip(dyn.traces, plain.traces):
            assert {k: v for k, v in a.items() if k != "dynamic_committed"} == b


def test_dynamic_first_skips_the_verifier_when_every_candidate_is_confident() -> None:
    # FakeHead is near-certain everywhere: each step is a whole block, as in dynamic.
    result, dream = run_dynamic_first(0.0, peaks=None, traces=True)
    assert result.continuation_token_ids == [sdar_target(k) for k in range(8)]
    assert result.commitment_histogram == [0, 0, 0, 2]
    assert dream.forwards == [] and result.verifier_forward_passes == 0
    assert [t["accepted_prefix_length"] for t in result.traces] == [4, 4]
    assert all(t["dream_rows"] == 0 and t["marginal_log_factors"] == [None] * 4 for t in result.traces)


def test_dynamic_first_with_a_wide_delta_takes_whole_blocks() -> None:
    result, _ = run_dynamic_first(10.0, traces=True)
    assert result.continuation_token_ids == [sdar_target(k) for k in range(8)]
    assert result.commitment_histogram == [0, 0, 0, 2]
    assert [t["accepted_prefix_length"] for t in result.traces] == [4, 4]
    assert result.extras["all_candidates_accepted_steps"] == 2


def test_the_cli_dynamic_first_variant_and_its_label() -> None:
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_benchmark import parse_method
    sampling = {"fixed": {}, "dynamic": {}, "adaptive": {"temperature": 0.0, "top_p": None, "confidence": "logp"}}
    config = parse_method("xadaptive-dyn95:0.10", 256, 8, max_candidates=16, sampling=sampling)
    assert config.dynamic_threshold == 0.95
    assert config.subset_rule == "prefix" and config.pmi_test == "two_sided"
    assert config.label == "xadaptive_dyn95_noveto_delta_0p10"
    assert parse_method("xadaptive:0.10", 256, 8, max_candidates=16, sampling=sampling).dynamic_threshold == 0.0
    for spec in ("xadaptive-dyn95-os:0.10", "xadaptive-dyn95-plus1:0.10", "xadaptive-remask-dyn95:0.10"):
        with pytest.raises(ValueError):
            parse_method(spec, 256, 8, max_candidates=16, sampling=sampling)
    with pytest.raises(ValueError):
        parse_method("xadaptive-dyn95:0.10", 256, 8, max_candidates=16, sampling=sampling, veto=True)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_method("dynamic-dyn95:0.10", 256, 8, max_candidates=16, sampling=sampling)
    with pytest.raises(ValueError):
        CrossConfig(method="adaptive", delta=0.1, continuation_length=8, dynamic_threshold=1.0)


def run_banded(delta, *, floor=0.85, threshold=0.95, peaks=(6.0, 5.9, 2.2, 2.0), traces=False):
    config = CrossConfig(method="adaptive", delta=delta, continuation_length=8, max_candidates=4,
                         veto=False, dynamic_threshold=threshold, band_floor=floor)
    dream = FakeDream(5, agree, coupled=True)
    prompt = torch.arange(SDAR_PROMPT, dtype=torch.long).unsqueeze(0) % 5
    sdar = build_sdar(head=StraddleHead(peaks) if peaks else None)
    result = generate_cross(sdar, dream, prompt, torch.zeros(1, 5, dtype=torch.long), config,
                            capture_traces=traces)
    return result, dream


def run_dynamic(threshold, peaks):
    from sdar_harness.blockwise import generate_blockwise
    from sdar_harness.scheduler import GenerationConfig
    config = GenerationConfig(method="dynamic", threshold=threshold, continuation_length=8, max_candidates=4)
    prompt = torch.arange(SDAR_PROMPT, dtype=torch.long).unsqueeze(0) % 5
    return generate_blockwise(build_sdar(head=StraddleHead(peaks)), prompt, config)


def test_banded_anchor_at_delta_zero_is_dynamic_at_the_threshold() -> None:
    # Per block: slot 0 untested, slot 1 (the least confident above 0.95) as
    # the anchor; the coupled verifier fails slot 2 given slot 1, then slot 3
    # given slot 2, so each follows alone: dynamic 0.95's 2, 1, 1.
    peaks = (6.0, 5.9, 2.2, 2.0)
    result, dream = run_banded(0.0)
    dynamic = run_dynamic(0.95, peaks)
    assert result.continuation_token_ids == dynamic.continuation_token_ids
    assert result.commitment_histogram == dynamic.commitment_histogram == [4, 2, 0, 0]
    assert result.extras["dynamic_committed_tokens"] == 2 and result.extras["anchored_steps"] == 2
    assert result.extras["band_offered_tokens"] == 2 * (2 + 1) and result.extras["band_committed_tokens"] == 0
    # The first verify scores the anchor and both band slots after slot 0.
    assert dream.forwards[0] == 3


def test_banded_anchor_with_a_wide_delta_is_dynamic_at_the_floor() -> None:
    peaks = (6.0, 5.9, 2.2, 2.0)       # ~1, ~1, ~0.90, ~0.88
    for floor, shape in ((0.85, [0, 0, 0, 2]), (0.89, [2, 0, 2, 0])):
        result, _ = run_banded(10.0, floor=floor)
        dynamic = run_dynamic(floor, peaks)
        assert result.continuation_token_ids == dynamic.continuation_token_ids
        assert result.commitment_histogram == dynamic.commitment_histogram == shape


def test_banded_anchor_is_measured_on_the_canvas_with_the_rest_of_the_confident_set() -> None:
    result, _ = run_banded(0.0, traces=True)
    first = result.traces[0]
    assert first["dynamic_committed"] == 1 and first["anchored"] and first["band_end"] == 4
    assert first["accepted_prefix_length"] == 2
    assert first["committed_positions"] == first["candidate_positions"][:2]
    assert first["marginal_log_factors"][0] is None
    # Slot 1 is the anchor: its marginal is read with slot 0 filled (the
    # coupled verifier is unsure next to a filled neighbour) and equals its
    # conditional; slot 2 is read unsure only once the anchor is filled.
    assert first["conditional_log_factors"][1] == first["marginal_log_factors"][1] < math.log(0.7)
    assert first["error_by_prefix_nats"][1] == 0
    assert first["marginal_log_factors"][2] > math.log(0.99) > math.log(0.7) > first["conditional_log_factors"][2]


def test_banded_anchor_without_a_confident_candidate_anchors_on_the_top_one() -> None:
    # ~0.93, ~0.92, ~0.90, ~0.88: nothing above 0.95, so the top candidate is
    # the anchor (dynamic's one-token fallback) and the band follows it.
    peaks = (2.6, 2.4, 2.2, 2.0)
    result, _ = run_banded(10.0, peaks=peaks, traces=True)
    assert result.extras["anchored_steps"] == 0 and result.extras["dynamic_committed_tokens"] == 0
    assert result.commitment_histogram == run_dynamic(0.85, peaks).commitment_histogram == [0, 0, 0, 2]
    assert [t["band_end"] for t in result.traces] == [4, 4]
    assert all(not t["anchored"] for t in result.traces)


def test_an_empty_band_never_calls_the_verifier() -> None:
    # Floor 0.92 leaves only the ~0.93 slot above it: each step is the anchor
    # alone, which the MI rule commits without a forward (fixed:1 here).
    peaks = (2.6, 2.4, 2.2, 2.0)
    result, dream = run_banded(10.0, floor=0.92, peaks=peaks)
    assert result.commitment_histogram == run_dynamic(0.92, peaks).commitment_histogram == [8, 0, 0, 0]
    assert dream.forwards == [] and result.verifier_forward_passes == 0
    assert result.extras["band_steps"] == 0 and result.extras["band_offered_tokens"] == 0
    # When every candidate is confident, the anchor is the last of them.
    whole, whole_dream = run_banded(0.0, peaks=None)
    assert whole.commitment_histogram == [0, 0, 0, 2] and whole_dream.forwards == []
    assert whole.extras["dynamic_committed_tokens"] == 6 and whole.extras["anchored_steps"] == 2


def test_the_cli_banded_variant_and_its_label() -> None:
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_benchmark import parse_method
    sampling = {"fixed": {}, "dynamic": {}, "adaptive": {"temperature": 0.0, "top_p": None, "confidence": "logp"}}
    config = parse_method("xadaptive-dyn95-band80:0.10", 256, 8, max_candidates=16, sampling=sampling)
    assert config.dynamic_threshold == 0.95 and config.band_floor == 0.80
    assert config.subset_rule == "prefix" and config.pmi_test == "two_sided"
    assert config.label == "xadaptive_dyn95_band80_noveto_delta_0p10"
    assert parse_method("xadaptive-dyn95:0.10", 256, 8, max_candidates=16, sampling=sampling).band_floor == 0.0
    for spec in ("xadaptive-dyn95-band80-os:0.10", "xadaptive-dyn80-band95:0.10"):
        with pytest.raises(ValueError):
            parse_method(spec, 256, 8, max_candidates=16, sampling=sampling)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_method("xadaptive-band80:0.10", 256, 8, max_candidates=16, sampling=sampling)
    with pytest.raises(ValueError):
        CrossConfig(method="adaptive", delta=0.1, continuation_length=8, band_floor=0.8)
