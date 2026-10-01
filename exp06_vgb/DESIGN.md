# exp06_vgb: density-ratio VGB (r = 1), SDAR-8B target / SDAR-1.7B reference

Status: probing. Framework, toy-model tests, and GPU probes (§6–§8); nothing run on
benchmarks yet. Decisions taken so far are in §0; the rest of §4 is open.

## 0. Decisions (2026-09-26)

- **Leaf rule: `exact`** (Def 4.2 as written, running log V, target π_b).
  Waiting on the walk probe to see whether the depth n−1 stall (§1c) is a
  problem in practice.
- **Model pair: SDAR-8B-b64 / SDAR-1.7B-b64**, per-block walk (n ≤ 64), exact
  leave-one-out, no shortlist (D6). The b64 block keeps exact LOO affordable,
  which a full-canvas model (Dream, LLaDA) would not. Neither has a smaller
  same-tokenizer sibling anyway.
- **Forward move: SDAR-consistent** (D4/D5). Pick the position by logp
  confidence and take the argmax token (`forward_selection="confidence"`,
  `temperature=0`). This is exactly the 8B fixed_k_1 decoder. ρ is always
  computed from the untempered (T = 1) conditionals. With backward moves off,
  this reproduces the 8B fixed_k_1 records, so any change is due to
  remasking. It is no longer an exact sampler of π_b; it is VGB's remasking
  rule on top of the greedy decoder.
- **EOS is a boundary, not a hard stop** (D8). Inside the current block a stop
  token is an ordinary token: it can be committed, remasked, and placed
  elsewhere or not at all. The run ends only when a *completed* block
  contains a stop token.
- **Scope: GSM8K, HumanEval+, MBPP+** (D10), on the same 100 task IDs as the
  8B fixed-k frontier.
- **Placement: two cards per replica** (D9). 1.7B shares the second 8B card.
  The measured peak is 7.7 GB / 12.3 GB, so this fits.
- **Numerics: bf16 LM head as-is.** No fp32 projection.
- **Budget: 1024 moves per block (16 × 64)** (D7), then forward-only
  completion with the 8B decoder. It counts moves, not steps: momentum
  stay/switch steps reuse the cached ρ and cost nothing, so they don't count
  toward `max_steps_per_block`. Report the budget-hit rate.
- Latency is not a goal. The aim is the quality ceiling.

Measured per-row cost (GSM8K prefix, cards 0+1): 8B ≈ 26 ms/row, 1.7B ≈
8 ms/row once batched. That is ≈ 34 ms × k per exact-LOO step.

- Paper: Jeon, Vuong & Tao, *VGB for Masked Diffusion Model*, arXiv:2606.28301,
  Definition 4.2 (plain chain) and 4.3 (momentum lift). Official code:
  <https://github.com/KraitGit/MDM-VGB> (`src/algorithms/vgb_candidates.py`).
- Derivation: [vgb_density_ratio_decoding.tex](vgb_density_ratio_decoding.tex).
- Code: [adaptive_re/vgb.py](adaptive_re/vgb.py);
  tests: [tests/test_vgb.py](tests/test_vgb.py);
  probe: [scripts/probe_vgb.py](scripts/probe_vgb.py).

## 1. What the derivation gives, plus three additions

The tex derivation is correct for interior states. Three points it does not
state, all checked against exact toy joints in `tests/test_vgb.py`:

**1a. The direction probability is simpler.** With `(n-k) s_{k,1} = k s_{k-1,1}`,

    P(forward | z) = k / (k + Σ_i ρ_i^λ) = 1 / (1 + mean_i ρ_i^λ).

So the depth coefficients never have to be computed. Also, the depth drift has
the sign of `1 - mean ρ^λ`.

**1b. λ = 0 makes the small model irrelevant.** Every `ρ^0 = 1`, so P(forward)
= 1/2, and remasking is uniform. The small model is never called. λ > 0 is the
only regime where 1.7B does anything.

**1c. The leaf boundary does not cancel V(z).** Definition 4.2 drops `V(z)^λ`
on the edge *into* a leaf. That is what makes the leaf law exactly π*.
At depth n-1 the forward child is a leaf, so

    P(forward | z, k = n-1) = k / (k + V(z)^λ Σ_i ρ_i^λ).

The absolute value comes back here. The tex pseudocode (and the official code,
which gates every forward child by `V(z)^λ`) instead uses the interior formula
at depth n-1 as well. That is a different reversible chain whose leaf law is

    π_b(y) · Σ_i V(y^{-i})^λ  ≈  π_b(y)^{1+λ} / π_s(y)^λ · (LOO correction),

a **sequence-level contrastive tilt** (8B expert, 1.7B amateur), not π_b.
`test_interior_leaf_rule_targets_contrastive_tilt` confirms this exactly.
`test_exact_leaf_rule_targets_big_model` confirms the as-written rule gives π_b.

For the exact rule, `log V(z)` is cheap to track exactly. At the block root
`V = 1`, because both models are normalized. Each forward move adds
`log π_b(a|z) − log π_s(a|z)`, and each backward move adds `log ρ_i`. The only
extra cost is one reference row per forward move.

## 2. Feasibility

### 2a. SDAR is a block model, so the walk is per block

SDAR is causal across 64-token blocks and bidirectional within one. Its
conditionals are in-distribution only when every earlier block is complete. A
leave-one-out parent inside the current block (63 revealed, 1 masked) is
in-distribution. A remask in an earlier, committed block is not. So:

- The chain runs **inside one block** (n ≤ 64). The prompt and committed blocks
  are the fixed context x. The block's root is fully masked, and the walk
  stops at the first leaf. The block is then committed to **both** models'
  KV caches.
- Per-block target: π_b(block | prefix). Chaining blocks gives SDAR-8B's own
  block-autoregressive joint.
- No cross-block backtracking. A mistake in block b can't be undone from block
  b+1. That is the main loss relative to the paper's any-order setting.

Both checkpoints are b64, and they share `tokenizer.json` (same hash) and the
151,936 vocab. So token IDs and block boundaries line up with no translation.

### 2b. The comparison the theory predicts

If both models' conditionals came from coherent joints, the exact chain samples
**π_b**, exactly what forward-only random-order sampling from 8B gives. That
baseline costs one 8B row per token. VGB forward moves draw from π_b directly:
the small model never proposes, so **there is no speculative-decoding style
saving**. The 8B model is called on every step, on up to k rows.

So any quality gain has to come from something the ideal theory ignores:

1. **Conditional inconsistency.** SDAR's masked conditionals are not the
   marginals of one joint. Random-order ancestral sampling then depends on the
   order, and remasking may correct order-induced errors. Plausible, but not
   predicted.
2. **The contrastive tilt of the interior leaf rule** (§1c). Token-level
   contrastive decoding is known to help reasoning. This would be a
   sequence-level version with the plausibility constraint built in: every
   committed token is a π_b draw, and the tilt acts only through what gets
   remasked. This is the most concrete reason to expect a gain.
3. **Finite-time behavior.** The first-leaf law is not the stationary law.

The honest framing is "VGB as a principled remasking rule driven by the
8B/1.7B disagreement", evaluated at a cost several times higher than 8B alone.
It is not "VGB makes 8B cheaper".

### 2c. Hitting time: plain Definition 4.2 is diffusive

Interior P(forward) = 1/(1 + mean ρ^λ). Each committed token is drawn from π_b.
In the context where it was drawn:

- `E_{a~π_b}[ρ^λ] = Σ_a π_s^λ π_b^{1-λ}`, which equals **1 at λ = 1**.
- It is < 1 for λ < 1 (a Bhattacharyya-type overlap at λ = ½).
- It is > 1 for λ > 1.

The leave-one-out context differs from the draw context, so this is only
indicative. Still, it means:

| λ | depth drift | steps to a 64-token leaf |
| --- | --- | --- |
| > 1 | backward | exponential; blocks hit the budget |
| = 1 | ≈ 0 | diffusive, ~n² ≈ 4,096 |
| < 1 | small forward (8B and 1.7B agree on most tokens) | still hundreds to thousands |

The paper sees the same thing: Theorem 4.5 gives Θ(n²) for the plain chain and
O(n) for the momentum lift near balance. The paper's theory also assumes
`m ≤ V ≤ M`. Here `log V` is a sum of per-token log-ratios, so `(M/m)^{2λ}`
is exponential in the block length and the mixing bound is vacuous. **The
momentum lift (Def 4.3, χ ≈ 1) is effectively required.** It uses the same
weights and preserves the stationary law (`test_momentum_lift_preserves_stationary_law`).
Near balance it walks straight down and goes up only while the local
`mean ρ^λ > 1`, meaning while there are committed tokens 1.7B likes much more
than 8B does.

### 2d. Cost per step

Exact leave-one-out needs k parent rows × 64 tokens through **both** models
each step, plus 1 target row for z. The parents change after every move, since
attention within the block is bidirectional, so nothing carries over between
steps. Momentum stay/switch moves leave z unchanged and are free (the result
is cached).

These are FLOP estimates on the A4000s, not measurements. The GPUs were taken
by another user's jobs before the probe could run.

| | 1 row | 8 rows | 32 rows | 64 rows |
| --- | --- | --- | --- | --- |
| 8B, split over 2 cards (~1 TFLOP/row; measured 1-row step 57 ms) | ~57 ms | ~0.25 s | ~1 s | ~2 s |
| 1.7B (~0.2 TFLOP/row) | ~25 ms | ~50 ms | ~0.2 s | ~0.4 s |

Per step, with average k ≈ 32, exact LOO costs ~1.2 s against 57 ms for 8B
fixed_k_1. That is ~20×, before counting extra steps. For a momentum walk of
~100–200 steps per block, that is ~2–4 min per block. GSM8K (~3–4 blocks,
early stop) comes to ~10 min per task, or ~17 h per 100-task arm per replica.

With the official shortlist `L_b = 8` (8 random parents, backward mass scaled
by k/8): ~0.3 s per step, ~1 min per block, ~3–5 h per arm. **Exact LOO is a
small-scale reference. Any sweep needs L_b.**

Memory: rows go through `BlockCacheProvider` + `ExpandedCache`, so the prefix
K/V is a stride-0 view and is not copied. Only the k gathered positions are
projected through the LM head. `row_chunk` bounds activations.

The first thing to run once the cards are free is
`python scripts/probe_vgb.py --timing --walk`. It measures the table above and
the real `mean ρ^λ` / steps-to-leaf on GSM8K first blocks.

## 3. What is implemented

`adaptive_re/vgb.py`:

- `local_moves` / `choose_direction`: the reduced rule (§1), both leaf rules,
  shortlist scaling, and the Def 4.3 momentum lift. These are pure functions,
  tested against the raw Def 4.2 weights and exact stationary laws.
- `walk_block`: one block's chain. The target batch is the parents plus z.
  The reference batch is the parents, plus z on forward moves under the exact
  rule. It includes budget fallback, forward-only completion, traces, and
  row/call accounting.
- `generate_vgb`: the block loop, with two models, two caches, and commit to
  both. It returns the harness's `GenerationResult`: `proposer_passes` = 8B
  calls, `verifier_rows` / `verifier_forward_passes` = 1.7B rows / calls, and
  per-block VGB stats in `extras`.
- `VGBConfig(backward=False)` is the forward-only π_b control.

Not done yet: a `vgb:` method kind and `--reference-model` in
`scripts/run_benchmark.py`, and a launcher. Both wait on §4.

## 4. Decisions to make

**D1. Leaf rule: `exact` (target π_b) or `interior` (contrastive tilt).**
This decides what the experiment claims. Under `exact`, if the average
per-token log-ratio is positive (it is in expectation: KL(π_b‖π_s) ≥ 0),
`V(z)^λ` grows over the block and depth n-1 becomes hard to leave forward.
Expect budget hits unless λ is small. `interior` is scale-free and matches the
official code. Proposal: `interior` as primary, `exact` at one λ as the
π_b-faithful check.

**D2. Plain (Def 4.2) or momentum (Def 4.3).** §2c says plain is ~n² steps
per block. Proposal: momentum χ = 1 as primary, and plain only in the probe to
show the diffusive behavior.

**D3. λ grid.** λ > 1 drifts backward (§2c). Proposal: {0.25, 0.5, 1}. λ = 0
is not interesting (§1b), apart from serving as the uniform-remask control.

**D4. Temperature and the greedy baseline.** ρ needs proper distributions, so
T > 0. The existing 8B frontier is greedy. Options:
(a) T = 1: faithful, but it will likely trail 8B greedy on its own.
(b) A tempered target, T < 1 on both models. This is a heuristic, since tempered
conditionals are not consistent, but it is closer to greedy.
(c) ρ at T = 1 with forward draws at T < 1. This breaks the derivation.
Whichever we pick, the matching forward-only arm (same T, same order rule) is
the real control.

**D5. Forward position rule.** `uniform` is the exact Def 4.2 rule. The
existing SDAR baselines unmask by logp confidence, which matters a lot for
SDAR quality. `confidence` only changes which forward child is taken, never
the forward/backward split. Proposal: run both. Compare each against the
forward-only arm with the same rule.

**D6. Backward neighborhood.** Exact LOO (k rows), or shortlist L_b (official
default 8, random positions). A third option is a stale commit-time ratio:
`π_s(a|z_then)/π_b(a|z_then)`, recorded when the token was committed. It needs
no parent passes, but it is not VGB. Proposal: L_b = 8 for sweeps, exact LOO
on a 20-task slice to measure the shortlist's effect.

**D7. Budget.** Steps per block before forward-only completion (the paper
uses n×N overall). Proposal: 4n = 256 per block, and report the budget-hit
rate. A high rate means the chain is not working, not that we should raise the
budget.

**D8. Stopping.** Currently: stop at the first block leaf that contains a stop
token with everything before it decoded. The official code treats an EOS
choice as a terminal leaf immediately. That saves decoding the rest of the
block, but it stops before tokens after the EOS are settled. Proposal: keep
the block-leaf rule, since it is cleaner.

**D9. Placement.** 8B needs 2 cards. 1.7B (~3.5 GB) can share the second 8B
card, which has ~8 GB free at the measured 7.7 GB peak, minus LOO activations
(bounded by `row_chunk`). That allows 2 replicas on 4 cards. The alternative
is 1 replica with 1.7B on its own card. Check with the probe's peak memory.

**D10. Scope.** Start with GSM8K (canvas 256 = 4 blocks) on the same 100 task
IDs as the 8B frontier. Add MATH-500 only if GSM8K shows a signal. The code
benchmarks (HumanEval+/MBPP+) are poor fits: long blocks of low-entropy code
tokens, where ρ ≈ 1 almost everywhere.

**D11. Variance.** Every arm is stochastic. 100 tasks gives ±~4–5 points SE at
these accuracies. Plan 2–3 seeds for any arm we want to claim a difference on.

**D12. Metrics.** Accuracy. Wall time and tokens/s. 8B rows and calls, and
1.7B rows and calls. Commits. Steps, remasks, stays/switches per block.
Budget-hit rate. The distribution of `log mean ρ^λ`. Which tokens get
remasked; exp04's trajectory player could render these.

## 5. Arms, once D1–D6 are fixed (sketch)

| Arm | Purpose |
| --- | --- |
| 8B fixed_k_1 greedy (exists) | quality ceiling of the current decoder |
| 8B forward-only, T, uniform / confidence | the theory-equivalent control (same law as exact VGB) |
| VGB interior, momentum, λ ∈ {0.25, 0.5, 1}, L_b = 8 | main sweep |
| VGB λ = 0, momentum | uniform remasking at the same cost (is it the 1.7B signal or just remasking?) |
| VGB exact, λ = 0.5, momentum | π_b-faithful version |
| VGB exact LOO vs L_b = 8, 20 tasks | shortlist error |

## 6. Probe results (2026-09-26, `outputs/vgb_probe/walk_all.jsonl`)

Setup: 3 GSM8K tasks from the 100-task sample (1104, 790, 23), full
256-token canvas, momentum χ = 1, confidence order, 256-move budget per
block, cards 0+1. 8B fixed_k_1 scores these 1 / 0 / 1.

- **The plumbing is exact.** Forward-only greedy reproduces the 8B fixed_k_1
  completions token for token (3/3).
- **No arm changed a score.** Every config got 1 / 0 / 1. Every greedy arm
  except the exact λ = 1, T = 1 one produced exactly the fixed_k_1 text on
  1104 and 790.
- **Confidence order cancels the temperature.** At T = 0.7 and T = 1 the
  interior arms made **zero** remasks. Their `log mean ρ^λ` p10–p90 stayed
  within ±0.07. The confidence rule commits whichever drawn token is most
  probable, which is almost always a near-deterministic token that both
  models agree on. So ρ ≈ 1, and a T > 0 draw is almost always the argmax.
- **Interior + greedy remasks rarely, and the refill is identical.**
  λ = 2: 23 remasks, on one task. λ = 4: 11 and 23 remasks. Every remasked
  position was refilled with the same greedy token, so the output was
  unchanged. Most remasks hit `<|endoftext|>` padding after `<|im_end|>`.
- **The exact rule stalls at depth n−1 in every block** (all 8 blocks forced
  at every λ), even with small final log V (0–7). It oscillates near the leaf:
  220–346 remasks per task, at 5–10× the interior arms' cost, with no score
  change.
- **Cost.** Interior arms take 85–235 s per task; fixed_k_1 takes 7–12 s.
  That's ~15–20×, almost all of it leave-one-out rows that did nothing.

Implication: under SDAR confidence decoding, the 8B/1.7B density ratio gives
the chain almost nothing to act on. Remasking needs committed tokens the
models disagree on. Those appear with uniform (Def 4.2) order at T = 1: the
first single-prompt probe remasked 16–58 tokens per block there.

## 7. Track B: 1.7B drafts, 8B-guided VGB edits (leaf-start)

This is the paper's editing mode (Figure 1b and the Dyck/DNA/Protein repair
runs), with the same reward τ = π_b/π_s and the same chain. Only the start
state differs. `VGBConfig(start="draft", max_excursions=E)`. For each block:

1. **Draft.** 1.7B fills the block with its greedy fixed_k_1 decoder
   (`draft_block`): at each step, the argmax token at the most confident
   masked position, 64 forwards. It is conditioned on the already-edited
   earlier blocks, so from block 2 on the draft differs from the standalone
   1.7B run.
2. **Edit.** The chain starts at that complete block (a leaf). From a leaf
   the only move is a remask, at position i with probability ∝ ρ_i^λ: the
   tokens 1.7B favours more than 8B does. Unmasks are 8B moves, using the
   config's forward rule (Track B uses confidence order + greedy, the 8B
   decoder).
3. **Excursions.** An excursion leaves the leaf and returns to one. They
   repeat until one returns the block unchanged (converged), or until E
   excursions (default 4), or until the move budget runs out. One excursion
   can contain many remasks and refills.
4. Commit the block to both caches. Stop if it contains a stop token.
   Inside the block, EOS is an ordinary token and can be remasked.

Constraints: the interior leaf rule only, since the exact rule would need log
V along the draft path, which means scoring the draft with 8B.
`max_excursions = 0` returns the draft; it reproduces 1.7B fixed_k_1 token
for token (checked).

**Smoke test** (2026-09-26; `outputs/vgb_probe/edit_smoke.jsonl`; λ = 1,
greedy confidence-ordered 8B refills, 256-move budget):

| Config | 667 (1.7B ✗, 8B ✓) | 1104 (both ✓) | Time per task |
| --- | --- | --- | --- |
| Draft only (E = 0) | ✗, identical to 1.7B | ✓, identical to 1.7B | 3–5 s |
| Plain chain, E = 4 | ✗; one no-op remask per block | ✓, unchanged | 8–14 s |
| Momentum χ = 1, E = 4 | **✓**; 47 tokens changed (44 in one block) | ✓; 72 remasks, 0 changed | 141–218 s |

The plain chain remasks one token and 8B refills the identical token. A
coherent but wrong draft pulls single-token refills back to the same token,
and the "unchanged → stop" rule ends the block. Momentum remasks runs of
tokens, so 8B can rewrite a stretch, and a correct draft survives because
every refill agrees. The one fix came mostly from 8B regenerating a large
part of a block, not from a surgical edit.

## 8. Forward rules and the shortlist (importance-reweighting) derivation

Every forward move reveals one position with one token (r = 1). The rules
differ only in which forward child (i, a) is taken. None changes the
forward/backward split: that always uses the exact reduced forward mass
F~ = k (§1a), so the direction and backward derivations hold unchanged
under all three.

Def 4.2's weight for child c_{i,a}, divided by the common factor
s_{k,1} V(z)^{1+λ}, is

    w(i, a) = π_s(a | z) · V(c_{i,a}) / V(z) = π_s(a | z) · π_b(a | z) / π_s(a | z) = π_b(a | z).

**`uniform` (exact Def 4.2).** Normalising w: i ~ Unif(masked), a ~ π_b(·|z).
Our density-ratio value gives this closed form. That is why the tex
derivation needs no enumeration.

**`confidence` (SDAR decoder, heuristic).** Draw a token at every masked
position, then commit the most confident draw. With T = 0 this is exactly
8B fixed_k_1. It is not Def 4.2's child distribution. In the root-start
probe it made the chain inert (§6).

**`shortlist` (the paper's `high_conf`, official
`build_mdm_forward_candidates`).** The paper's V is a learned verifier that
must be called on each child, so it can't normalise w over 150k tokens × 64
positions. It shortlists instead. With L_f = `shortlist_positions` and
K = `shortlist_tokens`:

1. Keep the L_f masked positions where the reference π_s is most confident
   (max probability).
2. At each kept position, draw K tokens from π_s(·|z) at the config's
   temperature (T = 1 in the paper). Merge duplicates, with count c(i, a).
3. Give each candidate the weight

       ŵ(i, a) = (c(i, a) / K) · V(c_{i,a}) / V(z) = (c(i, a) / K) · π_b(a | z) / π_s(a | z),

   and pick one candidate with probability ∝ ŵ.

The empirical frequency c/K stands in for π_s(a|z), and the value ratio
supplies the rest. This is **sampling–importance–resampling**: π_s is the
proposal, π_b/π_s is the importance weight, and π_b is the target. As
K → ∞, c/K → π_s(a|z), so ŵ → π_b(a|z), the exact weight. With every
position kept, it becomes the exact `uniform` rule;
`test_shortlist_with_full_coverage_is_ancestral_sampling_of_big` checks this.
At finite K the pick is biased toward tokens 1.7B finds likely. A token that
8B likes but 1.7B almost never proposes can't be a candidate. With K = 1 the
committed token is simply a 1.7B draft token
(`test_shortlist_single_draw_proposes_from_small`).

What still holds, and what doesn't:

- **Holds:** the direction probability P(forward) = 1/(1 + mean ρ^λ), the
  backward choice ∝ ρ_i^λ, the interior/exact leaf analysis, and the
  momentum lift. All are unchanged.
- **Approximate:** the forward child distribution. It is a K-sample SIR
  approximation of the π_b draw, restricted to L_f positions.
- **Not in Def 4.2:** ranking positions by π_s confidence. This is the paper's
  implementation heuristic.
- **Deliberate difference from the official code:** the official code sums
  the shortlisted weights, rescaled by (#positions / L_f), to get the forward
  mass. That estimate is biased when positions are chosen by confidence
  rather than at random. We keep the exact F~ = k.

Cost per unmask: one 1.7B row on z (all masked positions, for the ranking and
the proposals) and one 8B row on z (the L_f kept positions, for π_b). That is
the same order as the other rules; the leave-one-out rows still dominate.

This rule puts 1.7B in the proposing seat of the forward move, while the
target stays 8B. It is the natural forward rule for Track A (remasking
ceiling with stochastic, paper-faithful unmasking), and it could replace 8B
greedy refills in Track B.

**Smoke test** (2026-09-26; `outputs/vgb_probe/shortlist_smoke.jsonl`;
L_f = K = 8, T = 1, root start, interior rule, momentum χ = 1, 256-move
budget; tasks 667 and 1104; 8B greedy gets both right, 1.7B only 1104):

| Config | 667 | 1104 | Remasks (667 / 1104) | Budget hits | Time per task |
| --- | --- | --- | --- | --- | --- |
| Forward-only shortlist | ✓ | ✓ | 0 / 0 | 0 | 10–16 s |
| λ = 1 | ✓ | ✓ | 126 / 27 | 1 block | 138–343 s |
| λ = 0.5 | ✓ | ✓ | 24 / 2 | 0 | 103–200 s |

Unlike greedy confidence order (§6), the chain is active here: log mean
ρ^λ spans p10–p90 ≈ −0.15 to +0.17 at λ = 1. Remasks land on ordinary
content tokens (" quarters", "4", " Sarah", " times"). λ sets how much
remasking happens, roughly 5× more at λ = 1 than at 0.5. Two tasks can't
say anything about accuracy.

**Editing probe** (2026-09-26; `outputs/vgb_probe/edit_probe.jsonl`;
momentum χ = 1, λ = 1, E = 4, greedy confidence-ordered 8B refills,
256-move budget). The run hit its 75-minute cap after 8 of 12 tasks. All 8
are tasks 1.7B gets wrong and 8B right. The 1.7B-right/8B-wrong and
both-right tasks (the regression side) did not run.

| Task | Edited | Tokens changed vs draft | Excursions per block | Remasks | Blocks at budget | Time |
| --- | --- | --- | --- | --- | --- | --- |
| 278 | ✓ | 64 | [3,13,0] [51,7] [0] [0] | 190 | 1 | 385 s |
| 154 | ✓ | 19 | [7,0] [11] [1,0] [0] | 318 | 1 | 758 s |
| 1284 | ✗ | 13 | [1,0] [3,0] [9,0] | 190 | 0 | 494 s |
| 77 | ✗ | 2 | [0] [2,0] [0] | 89 | 0 | 262 s |
| 187 | ✓ | 17 | [12,0] [1,0] [4,0] | 185 | 0 | 431 s |
| 1123 | ✗ | 0 | [0] [0] | 43 | 0 | 127 s |
| 450 | ✗ | 34 | [0] [20,0] [0] [14,0] [0] | 307 | 0 | 809 s |
| 7 | ✓ | 102 | [8,2,0] [45,11] [2,2,46] [0] | 391 | 2 | 967 s |

**Result: 4 of 8 repaired** (1.7B 0/8, 8B 8/8 by selection). Some fixes are
small: 187 changed 17 tokens and 154 changed 19. Others rewrote much of a
block (278, 7). Failures converge early: 1123's first excursion in every
block changed nothing. Cost is 2–16 min per task, against ~10 s for 8B
alone. Nearly all of it is leave-one-out rows at k ≈ 63.

## 9. Track A full run (launched 2026-09-26 23:41; relaunched 23:5x)

**Revised plan (relaunch):** MATH-500 is deferred because its
~1,300-token prefixes are untested against the 16 GB cards; GSM8K
already shows allocator retries. λ = 1 is added:
methods `vgb:fwdonly:sl8x8`, `vgb:lam0.5:mom:sl8x8`, `vgb:lam1:mom:sl8x8` on
GSM8K → MBPP+ → HumanEval+. The original plan, kept for the record:

`bash launch_vgb.sh`: one replica on cards 0+1 (8B split over both, 1.7B on
card 1). Methods `vgb:fwdonly:sl8x8` (no-remasking control) and
`vgb:lam0.5:mom:sl8x8`: T = 1, interior rule, χ = 1, exact leave-one-out,
512 moves per block, 16 leave-one-out rows per forward. Benchmarks run in the
order GSM8K → MBPP+ → HumanEval+ → MATH-500, 100 tasks each, seed 271828
(the 8B fixed-k task IDs). Canvases 256 / 256 / 512 / 1024. Then EvalPlus
scoring and `summarize_results.py` → `outputs/vgb_trackA/comparison/`.
Resumable: rerunning the launcher skips finished tasks. Track B is on hold.

Runner integration: `run_benchmark.py --methods vgb:<spec>` (spec syntax in
`parse_vgb_spec`); the 1.7B reference loads via
`--verifier-model/--verifier-family sdar/--verifier-device`. VGB arms skip
the untimed warmup. The per-task sampler seed is derived from the
(seed, method, task) torch seed.

## 10. exp05 re-run with an aligned verifier (cross8b, 2026-09-29)

Question: exp05's remask pass mostly hit **style** mismatches between SDAR-1.7B
and Dream-7B (leading space at canvas 0, `\r\n` → `\n`, prose), not errors
(`../exp05_adaptive_re/report/remask_postmortem.html`, reason A). With SDAR-8B
as the verifier (same tokenizer, chat template, EOS and whitespace habits), do
the edits move from style to logic?

Setup: exp05's `xadaptive-remask` recipe unchanged except for the verifier.
SDAR-1.7B proposes (greedy, logp confidence, up to 64 candidates); SDAR-8B
scores them with the verifier-only ratio, no veto, groups of 4 rows; one
remask pass per block over the bottom 4. δ ∈ {0.1, 1}. Counterfactual:
`xadaptive` at the same δ (identical decoding without the remask pass), run in
full rather than exp05's edited-tasks-only shortcut. Same 100 task IDs and
canvases. `bash launch_cross8b.sh` → `outputs/cross8b`.

Implementation (`cross.py`): when the verifier is a block-diffusion checkpoint
it keeps its own KV cache, committed with SDAR's after every block, and scores
rows of the current block only (`BlockCacheProvider`). The rows see what
Dream's rows saw in the current block (the committed prefix plus the block);
future blocks are invisible instead of masked. Every remask edit is logged per
record (`remask_edits`: canvas index, old/new token, log p, bottom-4), so
edits can be classified on all tasks without traced re-runs. `--verifier-devices`
pipeline-splits the 8B over the card pair.

What it can and can't show: it isolates reason A. Reasons B (hindsight view),
C (one in-place substitution per block) and E (no veto, so the 8B never adds
its own tokens beyond the one remask) are unchanged. From §6, the two SDARs
agree almost everywhere under greedy decoding, so few edits is a likely outcome.

### 10a. Role-swapped ablation (cross8b_swap)

SDAR-8B proposes, SDAR-1.7B verifies (acceptance) and remasks, everything else
as §10: `ROLES=8b_proposes bash launch_cross8b.sh` → `outputs/cross8b_swap`.
With no veto, every committed token is an 8B greedy proposal. The 1.7B only
sets how many commit per step, plus one remask per block, where it replaces
the 8B token with its own argmax. The reference is 8B `fixed_k_1`
(`outputs/sdar8b`): the no-remask arm shows what 1.7B-gated parallel
commitment costs, and the remask arm shows whether the weaker model's edits
hurt. It is also a control for §10: if the swap edits look like the 8B-verifier
edits (wording, not logic), then the edits track model disagreement rather than
which model is right.

### 10b. Results (2026-09-29; `scripts/analyze_cross8b.py`, `outputs/cross8b/analysis.json`)

Both runs completed: the main run 11:01–14:19 and the swap run 14:19–16:06, cards 0+1, 100 tasks × 4 arms × 3 benchmarks each, no errors. Consistency checks before launch (`scripts/check_cross8b.py`): the cached block verifier is fp32-exact against an uncached full forward in every layout, and the bf16 gaps are rounding. Pilots reproduce across processes. Settings, task order, prompts and tokens match exp05. Everything runs in bf16, as exp05 did.

Score, no-remask → remask (8B greedy `fixed_k_1`: 0.91 / 0.40 / 0.60; 1.7B: 0.80 / 0.27 / 0.35):

| | exp05 δ0.1 | cross8b δ0.1 | swap δ0.1 | exp05 δ1 | cross8b δ1 | swap δ1 |
| --- | --- | --- | --- | --- | --- | --- |
| GSM8K | 77→81 | 76→79 | 88→89 | 72→76 | 84→82 | 89→90 |
| MBPP+ | 28→32 | 27→34 | 42→38 | 29→35 | 34→38 | 43→39 |
| HumanEval+ | 39→37 | 36→39 | 61→61 | 31→32 | 30→33 | 49→48 |

Remask fix/break, pooled over both δ (the p-value treats δ as independent, so it overstates):

| | fix / break | p | fixes direct / regen | breaks direct / regen |
| --- | --- | --- | --- | --- |
| exp05 (Dream verifies 1.7B) | 32 / 15 | 0.019 | 10 / 22 | 2 / 13 |
| cross8b (8B verifies 1.7B) | 23 / 5 | 0.0009 | 11 / 12 | 1 / 4 |
| swap (1.7B verifies 8B) | 15 / 22 | 0.32 | 3 / 12 | 9 / 13 |

"Direct" means the rest of the answer is ≥ 90% token-identical to the counterfactual, so the edit itself flipped the score; "regen" means SDAR's re-generation after the edit did. Per benchmark, only cross8b MBPP+ δ0.1 (7/0, p = 0.02) is significant. cross8b MBPP+ is 11/0 across both δ.

First-edit classes: Dream's artifacts are gone. GSM8K position-0 edits fell from 91% to 4–6%, and whitespace-only edits from 77% to 1–2%. A new style mismatch between the two SDAR sizes replaced them. On MBPP+, 62–72% of the 8B's first edits are `):\n`→`):\r\n`, and the swap run shows the reverse, `\r\n`→`\n`, at 46–50%. On GSM8K the edits became wording swaps (84–92%). Number edits are 1–6%, and the few that flipped a score broke it.

Reading:
- On code, alignment moved the edits partly to logic. Direct 8B fixes include `>`→`==`, `*`→`**`, `else`→`return`, `==`→`+`, `large`→`larg` (matching the test's function name), a missing `(`, and a prematurely ended answer restored.
- On GSM8K, no: score flips there are regeneration effects.
- The swap run is the control. The 1.7B makes the same kinds of edits to the 8B, and they hurt on net. Mbpp/637 shows it cleanly: the 8B fixes `>`→`==`, and the 1.7B "fixes" it back. Edits track where the models disagree, and they help only when the verifier is the stronger model.
- On MBPP+, most answers fit in one block, so an `\r\n` edit uses up that block's only remask.
