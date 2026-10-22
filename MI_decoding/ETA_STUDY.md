# The η study: does the 1.7B see the 8B's in-block dependence?

Designed 2026-10-18. Status: pilot and main run done (2026-10-19 00:13 UTC); results in §7a and §7b.

## 1. The question

At a proposal step with state `h` (prompt + committed tokens + the masked
slots of the current block), the proposer P (SDAR-8B-b16) ranks the masked
slots by confidence: slots `i_1 … i_k`, argmax candidates `y_1 … y_k`. For
each rank j, and for both P and the verifier V (SDAR-1.7B-b16):

```
q_j(·) = p(slot i_j = · | h)                      marginal
p_j(·) = p(slot i_j = · | h, y_1 .. y_j-1 filled)  conditional on the higher ranks

D_j^P = TV(p_j^P, q_j^P)      what we want to bound; needs extra 8B forwards
D_j^V = TV(p_j^V, q_j^V)      what the verifier can compute

η_j   = ½ ‖ (p_j^P − q_j^P) − (p_j^V − q_j^V) ‖₁      dependency-transfer error

triangle inequality:   D_j^P ≤ D_j^V + η_j
```

The hope is `η_j ≤ κ` for a constant κ, which then becomes the new error
threshold. This study measures η, at the candidates the decoder actually
commits.

## 2. Decisions

- **The claim is per slot, at inference time.** The bound must hold at the
  `y_<j` the decoder conditions on, which are the greedy argmax candidates in
  confidence order. So the candidates are greedy.
  - A bound on the joint law of sampled commits (`TV(P, ∏ q)`) would instead
    average over independent T=1 draws `y_<j ~ ∏ q`. That mode is not built,
    and could be added later.
- **States are on-policy.** The 8B decodes the prompt itself at fixed:1, the
  B=16 baseline's decode. Reference solutions are never teacher-forced.
  - The MI rule also commits the most confident slots first, so its partially
    filled blocks have the same shape.
  - The states depend on no threshold, so one run serves every δ, ε or κ the
    analysis tries.
- **Prompts are held out.** κ is a threshold the algorithm will run with.
  Fitting it on evaluation prompts would be tuning on test, even though η
  needs no labels. Calibration therefore uses splits no evaluation set
  touches (§3).
  - Once κ is frozen, the same measurement on the evaluation sets is a
    transfer check, not a fit.
  - `data/sample250` is drawn from the test sets, so it is not held out.
- **Distributions are full.** Every TV is over the whole vocabulary (151,936,
  shared tokenizer; only the mask token is barred, as in decoding), at T=1,
  in float64 from fp16 logits.
- **Models:** SDAR-8B-Chat-b16 (P) and SDAR-1.7B-Chat-b16 (V). These are the
  pair the B=16 MI work uses.

## 3. Data: `data/heldout/`

Built by `harness/scripts/prepare_heldout.py`, seed 271828: 150 prompts for
GSM8K and MATH, and all 320 for MBPP.
All use the OpenCompass templates.

| file | source | selection |
|---|---|---|
| `gsm8k_train.jsonl` | `openai/gsm8k` main **train** (7,473) | seeded sample |
| `math_train.jsonl` | `EleutherAI/hendrycks_math` **train**, 7 subjects (7,498 with a numeric level) | stratified by level in MATH-500's proportions: 13/27/32/38/40 |
| `mbpp_heldout.jsonl` | full MBPP prompt + validation + train (474), minus every MBPP+ task | all 320 that remain |

Each file is checked against `data/opencompass/` (IDs and problem text). The
MBPP file is also checked against `data/full/mbpp_plus.jsonl`, because EvalPlus
draws on every MBPP split. HumanEval has no train split; MBPP is the code
domain.

## 4. What runs

`measure_eta.py`, built on `harness/sdar_harness/dependency.py`
(`generate_measured`), launched by `launch_eta.sh`.

```
prefill the prompt into the 8B's cache and the 1.7B's cache
for each block (16 slots):
    choose 1 step at random among the block's steps with ≥ 2 masked slots
    for each step of the block:
        8B forward on the block (1 row)        ← the fixed:1 propose step, unchanged code
        if this is the chosen step: MEASURE, write one record
        commit the single most confident token
        stop once the answer has ended
    write the finished block into both caches
```

**MEASURE** at a step with k masked slots:
- Build k probe rows. Row j fills `y_1 … y_j-1`, the same construction as
  `cross.dream_verify`.
- Run all k rows through the 8B in one call, then through the 1.7B in one
  call (`--group-rows 16`).
- Row 1 is read at every slot; that gives `q`. Row j is read at slot `i_j`;
  that gives `p_j`.

The probe forwards are read-only (`store_kv=False`). The decode is fixed:1
token for token, so its completions must equal the B=16 fixed:1 baseline's.
`test_dependency.py` checks this against `generate_blockwise` on the fakes,
and `--check-against` checks it on the GPU.

**Measured steps per task:** one per block (four on MBPP), at most 64 per
task. A 4096-token runaway would otherwise contribute 256 blocks of one loop.
MBPP answers are short, about 3 blocks, so it measures 4 of each block's steps
and uses all 320 held-out prompts. Steps within one block share their context,
so extra steps per block add less independent information than extra prompts.
For both reasons MBPP gets more of each.

**Records** (`steps.jsonl`). There is one line per measured step. Every
per-rank field is an array over j = 1..k, in rank order.

| field | meaning |
|---|---|
| `block`, `block_step`, `step`, `block_start`, `answer_offset` | where the step is |
| `k`, `filled_in_block`, `context_tokens` | masked slots, filled slots in the block, visible tokens |
| `slots`, `tokens`, `confidence_ranked` | `i_j`, `y_j`, the decoder's confidence |
| `D_P`, `D_V`, `eta` | the three quantities of §1 |
| `tv_PV` | TV between the two models' marginals at `i_j` |
| `conf_P`, `conf_V` | `q_j(y_j)` under each model |
| `gap_P`, `gap_V` | top-1 minus top-2 probability of `q_j` |
| `ent_P`, `ent_V` | entropy of `q_j` |
| `lr_P`, `lr_V` | `log p_j(y_j) − log q_j(y_j)`, the MI rule's per-token term |
| `argmax_qP`, `argmax_pP`, `argmax_qV`, `argmax_pV` | argmaxes; a flip is `argmax p ≠ argmax q` |
| `dist_new` | distance from `i_j` to the nearest of `i_<j` |
| `first_stop_slot` | first committed stop token in the block's answer slots (−1 if none) |
| `eta_self`, `noise_q` | numeric floor, on the first 10 tasks per dataset (§6) |
| `measure_ms` | wall time of the measurement |

At rank 1 nothing is filled, so `D_P = D_V = η = 0` there by construction.

`tasks.jsonl` holds one line per prompt: the completion, lengths, the stop
token, steps, probe forwards, timing, peak memory, and `matches_baseline`
when checked.

## 5. Analysis (pre-specified; `analyze_eta.py`, CPU)

The unit is a (measured step, rank j ≥ 2) pair. Slots after a stop token are
padding; they are dropped from per-pair statistics but kept in prefix
replays. Confidence intervals bootstrap over tasks.

1. **Distributions.** Quantiles (50/90/99/99.9, max) of `D_P`, `D_V`, `η` and
   the gap `(D_P − D_V)+`, per domain and pooled.
2. **Binned, never only unconditional.** Most slots are near-deterministic,
   and `η ≤ D_P + D_V`, so η is trivially tiny there. Report η by `D_V` bin
   and by `D_P` bin. Within each bin report `Σ η / Σ D_P`: the share of the
   8B's dependence the 1.7B misses (1 means the bound is vacuous).
3. **κ.** The q90, q99 and q99.9 of η, with 95% CIs. Primary: q99 per domain.
   Report the same quantiles of the gap `(D_P − D_V)+`. The proof only needs
   `D_P ≤ D_V + κ`, and η ≥ |D_P − D_V| always, so the gap is the weaker
   sufficient assumption.
4. **Constant or affine.** A least-squares line through η's q99 per `D_V` bin.
   A clearly positive slope means a constant κ is the wrong form.
5. **False accepts.** For ε in {0.001, 0.003, 0.01, 0.03, 0.1, 0.3}, among
   slots with `D_V ≤ ε`, report the share with `D_P > ε + κ` (κ = 0, q90, q99)
   and the share where the 8B's argmax flips.
6. **Prefix replays.** Commit ranks 1..m while ranks 2..m pass. The rules are
   `D_V ≤ ε`, and the argmax-safe rule `D_V + κ < gap_P / 2` (if
   `TV < gap/2`, the argmax cannot change, so a parallel commit equals
   sequential greedy). Report tokens per step and the flip rate among
   accepted slots. Under the argmax-safe rule that rate is 0 wherever the
   bound holds.
7. **Accumulation.** The mean of `Σ_{j≤m} η_j` against m. The bound for the
   whole set pays roughly κ per committed token.
8. **Does V see it?** The AUROC of `D_V` (and `|lr_V|`) for 8B flips and for
   `D_P > 0.1`.
9. **Covariates.** η by rank, fill level, context length, and distance to the
   nearest newly filled slot.
10. **Stability.** Fit κ (q99) on half the tasks and report coverage on the
    other half. Later, apply the frozen κ to the evaluation sets.

**κ is fixed before looking: the q99 of η on the held-out prompts, per
domain.** There is no pass/fail line (decided 2026-10-18). Whether κ is small
enough is read from the distributions, the false-accept tables and the prefix
replays. A κ too large to use shows up there as about one token per step.

## 6. Controls

- **Numeric floor.** On the first 10 tasks of each dataset, the 8B's probe
  rows are read a second time in groups of 4. `eta_self` is η of the 8B
  against itself across the two groupings, and `noise_q` is the TV between
  the two readings of `q`. In exact arithmetic both are 0 (they are on the CPU
  fakes and on the CPU smoke test). η below this floor carries no signal.
- **Trivial verifier.** `D_V ≡ 0` gives `η = D_P`, which is the "share
  missed" baseline in §5.2.
- **The decode is the baseline's.** `--check-against` on a few evaluation
  prompts, against `outputs/opencompass/SDAR-8B-Chat-b16-ec2/<bench>/fixed_k_1`.

## 7. Run plan (each run needs an explicit go)

Run on EC2 (A10G, separate placement: 8B alone on one card, 1.7B on the next).
Both b16 checkpoints are there; the 1.7B-b16 is not on SDS. Timing is not a
result here, so any card works.

1. **Pilot, about 20 min on one pair** (launched 2026-10-18).
   - 5 GSM8K test prompts with `CHECK_ROOT`, for the identity check.
   - 10 prompts from each held-out file (MBPP at 4 steps per block).
   - Output `outputs/eta/pilot_b16`.
   - Outputs to read: identity, floor, probe ms per step, peak memory at long
     context.
2. **Main run: every held-out prompt** (150 GSM8K, 150 MATH, 320 MBPP).
   - Estimated from the B=16 fixed:1 baseline's lengths (GSM8K 187, MATH 595,
     MBPP 48 mean tokens), plus about 0.2 s per measured step:
     - GSM8K ≈ 0.6 h
     - MATH ≈ 1.8 h
     - MBPP (320 prompts, 4 steps per block) ≈ 0.5 h
   - Two replicas (MATH on cards 2,3; GSM8K then MBPP on 0,1) take about 2 h
     of wall time.
   - Output `outputs/eta/heldout_b16`.
   - Expected (step, rank) pairs: roughly 13k for GSM8K, 40k for MATH and
     35k for MBPP.
3. **Later, with κ frozen:** the same run on the evaluation sets
   (`data/sample250` GSM8K/MATH, HumanEval, sanitized MBPP), as the transfer
   check.

## 7a. Pilot result (2026-10-18, EC2, 22:11–22:22 UTC)

`outputs/eta/pilot_b16`. Ten prompts per held-out file, in file order; MATH's
first ten are all algebra. Also 5 GSM8K test prompts for the identity check.

**Plumbing:**
- Identity: 5/5 completions are byte-identical to the B=16 fixed:1 baseline.
- At rank 1, the 1.7B's argmax equals the 8B's candidate 95.6% of the time,
  in line with the earlier MI traces.
- The fp16 floor `eta_self` has median 2e-4 and q99 0.01. It scales with
  uncertainty: 3e-5 at q99 for 8B confidence ≥ 0.999, about 0.01 below 0.9.
- Peak memory: the 8B card reached 15.9 GiB at contexts up to 600 tokens; the
  1.7B card 4.2 GiB.
- Probing takes 0.3 s per measured step, 26% of wall time at one step per block.

**First look** (4,940 pairs, 35 tasks):
- **Unconditional η is large:** median 0.17, q99 1.15. A constant κ at the
  q99 is vacuous, since TV ≤ 1.
- **It is concentrated at slots the 8B is unsure of.** Half the pairs have
  8B confidence < 0.9. There both models' conditionals move a lot, in
  different directions: η median 0.27–0.58, and the 8B's argmax flips in
  3–67% of cases.
- **Where the decoder would commit, η is small:**

  | 8B confidence | η median | η q99 |
  |---|---|---|
  | ≥ 0.999 | 2e-4 | 0.11 |
  | 0.99–0.999 | 0.01 | 0.97 |

  The q99 of 0.97 in the second row comes from the 1.7B over-reporting
  (D_V ≫ D_P).
- **The direction that can break the bound is small.** At confidence
  ≥ 0.999 the gap `(D_P − D_V)+` has q99 6e-4. At D_V ≤ 0.01 the 8B's argmax
  flips on 0.08% of slots.
- **Implication for the analysis.** A single constant κ over all slots is the
  wrong object. The informative quantities are κ conditional on the regime
  the rule accepts in (by D_V or by confidence), and the one-sided gap. The
  report now also bins by the 8B's confidence. The main run's records already
  hold everything needed.

## 7b. Main run result (2026-10-18 22:32 – 10-19 00:13 UTC, EC2)

`outputs/eta/heldout_b16`, report in `analysis/report.md`.

**Audit:** clean.
- 150/150/320 tasks, IDs equal to the dataset files, no duplicates.
- Step counts match every task. All records have the same fields.
- 2 MATH runaways (4096 tokens), 12 tasks at the 64-step cap.
- Both replicas finished on their first attempt, and the analysis exited 0.
- One CUDA allocator OOM warning (fragmentation) appeared while decoding
  MATH train/counting_and_probability/225. The allocator retried and the
  task completed.

**Scale:** 91,584 (step, rank) pairs: GSM8K 12.6k, MATH 39.6k, MBPP 39.4k.

**Unconditional η: a constant κ is vacuous.**

| | η q50 | η q99 | (D_P − D_V)+ q90 | (D_P − D_V)+ q99 | split-half coverage at q99 |
|---|---|---|---|---|---|
| GSM8K | 0.19 | 1.06 | 0.080 | 0.36 | 0.99 |
| MATH | 0.09 | 1.06 | 0.051 | 0.36 | 0.99 |
| MBPP | 0.32 | 1.23 | 0.099 | 0.41 | 0.99 |

The distribution is stable (split halves agree), but its q99 exceeds 1.

**By the 8B's confidence `c` in y_j** (all domains; per-domain values within
about 2× of these):

| c | pairs | η q50 | η q99 | D_P q99 | (D_P − D_V)+ q99 | fp16 floor q99 | 8B argmax flips |
|---|---|---|---|---|---|---|---|
| < 0.3 | 13.6k | 0.60 | 1.32 | 0.98 | 0.35 | 0.011 (< 0.9) | 72% |
| 0.7–0.9 | 13.2k | 0.22 | 0.96 | 0.84 | 0.37 | 0.011 (< 0.9) | 3.7% |
| 0.9–0.99 | 16.1k | 0.066 | 0.92 | 0.20 | 0.082 | 0.0032 | 0.47% |
| 0.99–0.999 | 9.5k | 0.0098 | 0.81 | 0.016 | 0.0080 | 0.0005 | 0.01% |
| ≥ 0.999 | 15.5k | 0.0002 | 0.21 | 0.0010 | 0.00075 | 0.00002 | 0 |

- **Large η is mostly the 1.7B over-reporting.** At c ≥ 0.9, `Σ η / Σ D_P`
  is 4–40, because D_V ≫ D_P there. That direction is harmless to the bound.
- **The one-sided gap scales like 1 − c.** `(D_P − D_V)+` at q99 is ≈ 8e-4,
  8e-3 and 8e-2 in the three confident bins, the same in all three domains,
  and 30–40× above the fp16 floor.

**By D_V, the quantity the rule sees:**
- The gap's q99 is 0.013 at D_V < 1e-4, 0.10 at 1e-4 to 1e-3, 0.22 at 1e-3 to
  1e-2, and 0.41 at 0.01 to 0.03.
- So conditioning on a quiet verifier alone leaves a 1% tail where the 8B
  depends much more than the 1.7B says.
- The affine fit of η q99 against D_V has slope 1.4 and intercept 0.26.

**Greedy decisions are far safer than the TV bound suggests.**
- Among slots with D_V ≤ 0.01, the 8B's argmax flips on 0.16%.
- Prefix replay of "commit while D_V ≤ ε":
  - ε = 0.01: 2.7 tokens/step at a 0.08% flip rate (GSM8K 2.8, MATH 3.7,
    MBPP 1.8).
  - ε = 0.1: 3.8 tokens/step at 0.5%.
  - "D_V < gap_P/2" (κ = 0): 4.9 tokens/step at 0.75%.
- AUROC of D_V for 8B flips is 0.83–0.88; for |lr_V| it is 0.88–0.92.

**Accumulation.** Over ranks, Σ η grows about 0.4–0.5 per rank after the
first few, dominated by low-confidence ranks.

**Reading.** η ≤ κ with one constant fails. What holds, consistently across
domains, is a confidence-dependent one-sided version:
`D_P ≤ D_V + κ(c)`, with κ(c) of order 1 − c at the q99. The proposer knows c
at decode time.

## 8. Files

- `harness/sdar_harness/dependency.py`: probe rows, metrics, `generate_measured`.
- `harness/tests/test_dependency.py`: 21 CPU tests. They cover equality with
  fixed:1 `generate_blockwise`, η = 0 against an identical model, η = D_P
  against an independent model, both triangle bounds, step selection, the
  cap, grouping invariance, the floor, and hand-computed TVs.
- `harness/scripts/prepare_heldout.py` → `data/heldout/`.
- `MI_decoding/measure_eta.py`: the run, resumable per task.
- `MI_decoding/launch_eta.sh`: one replica. It refuses cards held by another
  user, and `DRY_RUN=1` prints the plan.
- `MI_decoding/analyze_eta.py` → `<output-root>/analysis/{report.md,summary.json}`.

## 9. Caveats

- η measures dependence on the higher-ranked candidates only, not on the
  slots still masked. The plus-one failures (B=16 dev run) came from the
  second kind.
- The states are greedy fixed:1 states. They are like MI states, but not
  identical.
- fp16 weights, as in every run. The floor (§6) says how much of η is
  arithmetic.

## 10. Follow-up: what an 8B self-check costs (2026-10-20)

**Question.** Could the 8B verify its own candidates? Replayed on the §7b steps, two rules accept far
more than dynamic at zero changed picks:
- "D^P < half the 8B's top-two margin": 5.6 tokens per measured step.
- "no 8B flip": 6.5 tokens per measured step.
- dynamic θ=0.95 accepts 4.0 on the same steps, at 0.07% changed picks.

That is 1.40–1.63× dynamic's tokens. The 6.5 is a ceiling: a byte-exact fixed:1 check also requires the
slot order to hold. Each step would be one 8B call with k probe rows; the rows double as the next step's
proposals. So the question is what k rows cost against one.

**Run.** `bench_verifier_rows.py` on EC2 card 0 (A10G), alone on the machine, host load ≤ 1, guard clear.
- Models: SDAR-8B-Chat-b16, plus the 1.7B-b16 for comparison.
- Grid: rows 1–16 × context 128–4096, 4 interleaved rounds × 8 timed calls per cell.
- The prefix cache is committed in 512-token chunks (`--prefill-chunk`, added for this run): a single
  4096-token commit ran the 8B out of memory. The timed call is unchanged.
- One allocator OOM warning appeared while building the prefixes; it recovered.
- Output: `outputs/verifier_rows_bench/ec2_card0_b16_20261020.json`; log in `logs/verifier_rows_bench/`.

Median ms per call:

| context | 8B ×1 | ×4 | ×8 | ×16 | 1.7B ×1 | ×4 | ×8 | ×16 |
|---|---|---|---|---|---|---|---|---|
| 128 | 49 | 57 | 69 | 123 | 37 | 37 | 37 | 39 |
| 512 | 53 | 76 | 108 | 200 | 36 | 37 | 42 | 74 |
| 1024 | 59 | 102 | 160 | 304 | 36 | 41 | 66 | 122 |
| 2048 | 72 | 156 | 267 | 517 | 36 | 66 | 115 | 218 |
| 4096 | 101 | 267 | 490 | 965 | 41 | 116 | 215 | 419 |

- **Sanity check.** The 8B's single-row cost matches the B=16 baselines' in-run fit (43–45 ms plus
  9–13 ms per 1k context), a few ms higher for the two gathers.
- **8B cost model.** Fit ms ≈ 40 + 2.8·rows + 13.3·rows·(context/1000). Almost all of the extra cost is
  the per-row copy of the prefix (`ExpandedCache`'s `torch.cat`), not compute.
- **8 rows vs 1 row of the 8B:** 1.42× at context 128, 2.05× at 512, 2.71× at 1024. The headroom is
  1.40–1.63×, so today a self-check loses to dynamic at every context the decoder spends time in.
- **The 1.7B (MI's current verifier)** is launch-bound and flat at about 37 ms up to 4–8 rows below 1k
  context. An MI step (8B ×1 plus 1.7B ×8) costs about the same as an 8B ×8 call: 95 vs 108 ms at 512.

**With shared-prefix attention** (rows folded into one sequence with a block-diagonal mask, so the prefix
is read once; not built), the fit's terms give 8 rows ≈ 66–76 ms, about 1.3–1.45× a single row from 256
to 1024 context. That is roughly break-even with dynamic: at most +15%, and only against the 6.5 ceiling.
Its case is quality: fixed:1's output, where dynamic loses points (MATH at B=16, −2.4), at about
dynamic's speed.
