# exp05: adaptive delta with a Dream-7B verifier, early stop and remasking

**Status (2026-09-19): implemented. B1 done. B2 dropped. The cross sweep (`xadaptive-remask`, 5 deltas x 3 benchmarks x 100 tasks) launched 01:13 CDT to `outputs/cross`; estimated about 8-11 h.**

- **Code:** early stop, the stop set and the vocabulary cut are in
  `adaptive_re/scheduler.py`, `blockwise.py` and `model.py`. The Dream-verified
  arms (`xadaptive`, `xadaptive-remask`) are in `adaptive_re/cross.py`.
  86 CPU tests pass.
- **B1 (SDAR fixed:k):** complete in `outputs/baseline/sdar`. It reproduces
  exp04 with 0 unexplained differences (`logs/baseline_b1/compare_exp04.txt`):
  - 1,176/1,200 completions are identical;
  - 12 stop at an `<|im_end|>` that exp04 decoded past (new text is an exact
    prefix of exp04's);
  - 3 are MBPP+ answers longer than the new 256 canvas.

  Scores match exp04 on all 12 arms. Latency drops 1.6x (GSM8K fixed:1) to
  7.9x (MBPP+ fixed:1).
- **Cross-model smoke test (real models, 2 tasks, GSM8K and MBPP+; 1 task
  HumanEval+):** runs end to end. Dream peaks at 14.5 GiB (canvas 256) and
  14.75 GiB (canvas 512) on its own card. About 6 Dream rows and 1.5 Dream
  forwards per step, about 4 tokens committed per step. 10-60 s/task on
  GSM8K/MBPP+ and 126 s for one 263-token HumanEval+ answer.
  The final variant (remasking on, veto off) was re-checked on real models on
  2 GSM8K tasks at delta 0.10 and 10.00. Going from 0.10 to 10 raised commits
  per step from about 3 to about 7 and cut time from 34-67 s to 25-40 s per
  task.

### Why this delta grid

Measured on the 371 steps of the smoke-test traces, all run at delta 0.5
(veto on). For each step that delta cut short, the delta needed to accept one
more token was:

| p10 | p25 | p50 | p75 | p90 |
| ---: | ---: | ---: | ---: | ---: |
| 0.57 | 0.71 | 1.35 | 3.7 | 549 |

So delta acts mostly between about 0.5 and 5, with a long tail beyond.
exp04's grid (0.1-1.5) sat mostly below that range. The same number of points,
log-spaced, 0.1 / 0.3 / 1 / 3 / 10, keeps exp04's low end and reaches the tail.

Caveats:
- Every trace was run at delta 0.5, so anything needing less than 0.5 was
  accepted and is invisible here.
- The sample is 5 tasks.
- With the veto removed, large delta tends toward "accept everything SDAR
  proposes, then let remasking fix it".

**Cost estimate** for the one variant (5 deltas x 3 benchmarks x 100 tasks) at
`LAYOUT=exclusive`: roughly 11 h. Piloting one delta at 100 tasks first is
advised.

Every number below is measured on this host (RTX A4000 16 GiB) unless it says
"estimate".

## Settled decisions

| | decision | section |
| --- | --- | --- |
| benchmarks | GSM8K 256, HumanEval+ 512, MBPP+ 256 (max canvas), early stop on all. MATH-500 (1024) deferred. | F2 |
| vocabulary | both models restricted to the shared IDs 0..151664 | F4 |
| verification | Dream rows in groups of 4, early exit at the first group with a violation; commit cap = block (64) | F1 |
| acceptance ratio | Dream-only: `log p_D(y_j | c, y_<j) - log p_D(y_j | c)` | F4 |
| first candidate | **no veto** (decided 2026-09-19; `--veto` keeps it as an ablation) | F4 |
| correction | per block: of the 4 least probable committed tokens, keep those Dream would change; if any, draw one with weight 1/p and replace it with Dream's argmax | F3 |
| stop tokens | `{<|im_end|> 151645, <|endoftext|> 151643}`, first occurrence | F2 |
| headline metric | tokens/s, as decoding papers report it | F1 |
| arms | one Dream-verified variant: `xadaptive-remask` (Dream-only ratio, no veto, remasking) | F3/F4 |
| delta grid | 0.1 / 0.3 / 1 / 3 / 10 (log-spaced) | below |
| baselines | SDAR fixed:k (B1) and exp04's Dream numbers only. **No SDAR-verified adaptive (B2)** (decided 2026-09-19) | Baselines |

## The setup

| role | model |
| --- | --- |
| proposer | SDAR-1.7B-Chat-b64 (`../models/SDAR-1.7B-Chat-b64-local`) |
| verifier and remasking detector | Dream-v0-Instruct-7B |

At each step, inside the current SDAR block, SDAR proposes a token for every
still-masked position in the block and ranks them by confidence (unchanged from
exp04). Dream scores the top of that ranking and commits the longest prefix
under the threshold. When a block is complete, Dream runs one remasking pass
over it before SDAR writes the block's K/V into its cache, replacing at most
one token.

**One process, two cards.** SDAR needs the transformers 4.52.4 overlay
(`vendor_env/tf452`), and exp03/exp04 ran Dream on 4.46.2. Checked: Dream's
logits are bit-identical under both versions (max abs diff 0.0), so both models
load in one process under the overlay.

---

## F1. Verification: how many Dream rows per forward

Row 0 is the unmodified state. It reads the Dream marginal at every candidate
position, which is also the conditional for candidate 1. Row `j` has
candidates `y_1..y_j` filled and reads `p_D(y_{j+1} | c, y_<=j)`.

**Settled: groups of 4 rows with early exit** (see "Recommended batching").
This replaces the earlier "up to 32 rows in one forward". The group size is
fixed at 4 for every arm, because in bf16 it can shift a borderline
acceptance.

### Memory

Dream-7B bf16, lean logits path, measured peaks:

| canvas (S = prompt ~140 + canvas) | Dream alone, 1 card | Dream split, card 1 / card 2 |
| --- | --- | --- |
| 256 (S=400): GSM8K, MBPP+ | 12 rows max (16 OOM) | 32 rows: 8.2 / 10.0 GiB |
| 512 (S=660): HumanEval+ | 6 rows max (8 OOM) | 32 rows: 9.5 / 11.3 GiB |
| 1024 (S=1180): MATH-500, deferred | 1 row fits, 8 OOM (2-6 not tested) | 32 rows: 12.1 / 13.9 GiB |

The transient is linear, about 164 KiB per row-token, with no `S^2` term.
Placing SDAR on card 1 (3.5 GiB weights, ~4.2 GiB peak in exp04 including its
old 4-row verifier) gives about 12.4 GiB at S=400 and 13.7 GiB at S=660. That
fits. MATH-500 would reach about 16.3 GiB on card 1 and needs Dream's layers
moved toward card 2. This is one more reason to defer it.

Layout: **2 cards per worker** (SDAR plus the first half of Dream, then the
second half of Dream plus lm_head), so **2 workers on the 4 A4000s**.

### What "cost" means, and why parallel rows do not save time here

"Cost" means GPU wall-clock time. On this card, time is set by how much
arithmetic a forward does.

Dream has bidirectional attention, so it has no KV cache. **Each verifier row is
a full forward over the whole prompt plus canvas** (400-660 tokens), not over
one new token. A single such row already keeps the A4000's compute units busy,
so stacking rows into one batch does not make them cheaper. The time grows with
the number of rows:

Measured with warmed-up kernels, median of 5 runs, time per row in seconds:

| rows B | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S=400, split | 0.132 | 0.117 | 0.106 | 0.106 | 0.105 | 0.105 | 0.105 | 0.104 | 0.104 | 0.107 |
| S=400, 1 card | 0.141 | 0.123 | 0.111 | 0.111 | 0.108 | 0.106 | 0.107 | OOM | | |
| S=660, split | 0.200 | 0.185 | 0.177 | 0.179 | 0.177 | 0.178 | 0.178 | 0.179 | 0.180 | 0.179 |
| S=660, 1 card | 0.209 | 0.189 | 0.182 | 0.182 | 0.180 | OOM | | | | |

(An earlier, un-warmed probe put split rows at about 0.139 s. These numbers
supersede it: splitting Dream across two cards costs nothing measurable.)

So yes, to your question: **doing the rows in parallel saves sequential depth
(1 Dream round instead of up to 32), but almost no time.** Per-row time stops
improving at about 3-4 rows. Going from 1 row to 4 saves 20% (S=400) or 11%
(S=660). Going beyond 4 saves nothing, while a larger group scores more rows
that end up thrown away. Autoregressive speculative decoding is the opposite
case. There each verifier row processes one new token against a KV cache, the
GPU is mostly idle at batch 1, and extra rows are nearly free.

What this means for tokens/s: a step costs about `rows x 0.14 s` and commits
`k` tokens. Dream `fixed:1` costs about 0.15 s per token (38.8 s for 256 GSM8K
tokens). So this design beats Dream `fixed:1` in tokens/s only when a step
commits more tokens than it scored rows. With a fixed 32 rows, that means about
32 tokens per step, which exp04 never came close to on GSM8K. **Estimate:** at
about 10 committed tokens per 32-row step, that is about 0.44 s per token,
roughly 3x slower than Dream `fixed:1`. The number of sequential rounds is still
much lower.

Levers that change this, each as an ablation:

- **Early-exit rows** (exp04's scheme): score rows in small groups and stop at
  the first violation. Rows per step then fall to about `k+1`, which is roughly
  break-even with Dream `fixed:1`.
- **Approximate Dream KV cache** (Fast-dLLM-style prefix or dual cache). Each
  row then processes only the active block (about 64 tokens) instead of
  400-660, so every row gets 6-10x cheaper. That, not free batching, is the
  win: a 64-token row sits below the A4000's ridge point of about 171 tokens
  (see "What limits rows per forward"), so only about 3 rows batch for free. It changes Dream's conditionals slightly, and a fair comparison
  needs Dream `fixed:1` with the same cache. This is the lever that could make
  the method win on tokens/s on this hardware.

### What limits rows per forward

Two different constraints, and it helps to keep them apart:

1. **Memory capacity sets the maximum rows** that fit: 12 at S=400 and 6 at
   S=660 with Dream alone on one card.
2. **Compute sets the time.** "Utilization" here means achieved tensor-core
   FLOP/s divided by the card's peak (~77 TFLOP/s bf16 dense on an A4000, spec
   sheet).

Dream spends about 13.1 GFLOP per token (2 x 6.5B non-embedding parameters).
One row at S=400 is therefore about 5.2 TFLOP. The measured times fit a simple
line almost exactly:

    t(B) ~= a + b * B * S      a ~= 30-34 ms,   b ~= 0.25 ms per token

- `a` is a fixed cost per forward. It matches the time to read Dream's
  14.2 GiB of weights once at the A4000's 448 GB/s (about 34 ms). Memory
  bandwidth only shows up here.
- `b` is the cost per token processed. It works out to about 53 TFLOP/s,
  roughly 70% of peak, which is about as high as dense bf16 matmuls get.

At B=1 a row runs at about 51% of peak, because the fixed ~32 ms is not yet
spread over enough work. By B=4 the fixed cost is under 8% of the total. From
there time grows with rows at about 70% utilization, and **no batch size can
beat that**. So yes, it is compute. Batching can only spread out the fixed
per-forward cost, and 4 rows already does that.

**Why this differs from autoregressive decoding.** A forward's arithmetic per
byte of weights read is roughly the number of tokens it processes. The
A4000's ridge point is 77 TFLOP/s / 448 GB/s ~= 171. Below ~171 tokens per
forward the card waits on memory, so extra rows are free. Above it, the card
is compute-bound and rows cost their full compute. An AR decode step processes
1 token per row, so it stays memory-bound until about 171 rows. A Dream row is
400-660 tokens on its own, so **a single row is already past the ridge point.**

### Would all 4 GPUs help?

**Splitting Dream's layers across cards (pipeline, as in the earlier probe):
no.** The cards run one after the other, and the measured per-row time is the
same as on one card.

**Tensor parallelism**, where each card computes a slice of every layer, does
cut the compute per card. The price is two activation all-reduces per layer.
There is no NVLink here, and the topology is two PCIe-switch pairs (GPU0-1,
GPU2-3) joined across CPU sockets. Measured cost of one forward's 56
all-reduces (NCCL):

| forward | TP-2 (GPU0-1, same switch) | TP-4 (across sockets) |
| --- | ---: | ---: |
| B=4, S=400 (11.5 MB each) | 36 ms | 70 ms |
| B=4, S=660 (18.9 MB each) | 59 ms | 114 ms |

Estimated B=4 forward, with compute split evenly plus that communication
(no overlap assumed):

| | 1 card | TP-2 | TP-4 |
| --- | ---: | ---: | ---: |
| S=400 | 425 ms | ~250 ms (1.7x) | ~176 ms (2.4x) |
| S=660 | 716 ms | ~417 ms (1.7x) | ~293 ms (2.4x) |

So TP would raise the batch-1 tokens/s, by about 1.7x on a switch pair and
about 2.4x on all four cards. That is an estimate: it needs a tensor-parallel
port of Dream's remote code, which does not exist here.

**Recommendation: don't, for now.**

- Published batch-1 tokens/s numbers are single-GPU. TP numbers are not
  comparable to them, and every baseline would have to be re-run under TP too.
- For finishing the sweep, **data parallelism is better use of the cards.**
  3 independent workers (F1 layout) give 3x the tasks per hour, against 2.4x
  for one TP-4 worker.
- TP-2 on a PIX pair is worth keeping as a later ablation if absolute latency
  becomes the point.

### Recommended batching: groups of 4 rows with early exit

Total verifier time is **(rows scored) x (time per row)**. Time per row is flat
from B=4 upward, so the best choice is the smallest group that reaches the flat
part (**4**), stopping at the first group that contains a violation. That is
exp04's scheme; exp04 also found microbatch 4 was the fastest for SDAR.

Rows scored per step are `4 * ceil((k+1)/4)`, where `k` is the number of
tokens accepted, against a fixed 32 in one forward. At `k` = 5 that is 8 rows
against 32, so **about 4x less verifier time**, at the cost of 2 Dream rounds
instead of 1. The accepted prefix is the same. The only exception is bf16
reduction order, which exp04 showed can flip a borderline acceptance, so the
group size is fixed once and kept identical across every arm.

This choice gives up round-parallelism to win on tokens/s. If the headline is
tokens/s, it is the right trade on this hardware. Rows per committed token
become about `(k+1)/k`, roughly break-even with Dream `fixed:1` in verifier
time. The pilot will tell whether SDAR's proposals get past that.

**The SDAR block does not need to change.** The block size (64) and the Dream
group size are independent. SDAR still ranks up to 64 candidates per step, and
Dream consumes that ranking 4 rows at a time until the first violation. b64
stays for the exp04 D8 reason: a smaller block caps how many tokens adaptive
can commit per step and censors the delta sweep. Moving to b16 would also need
a different checkpoint. With early exit the 32-row cap is unnecessary, so the
commit cap goes back to the block, 64.

**GPU layout changes.** With groups of 4, Dream fits on **one card by itself**
for all three benchmarks: 4 rows at S=660 fit, and 8 do not. Proposed layout:

- one card per Dream instance, with no split and no layer offloading;
- SDAR (~4 GiB each) on a shared card. Three SDAR copies fit on one 16 GiB
  card, giving **3 workers on 4 cards** instead of 2. SDAR's forwards are small
  and alternate with Dream's, so contention on the shared card should be low.
  The pilot has to measure that.
- MATH-500 (S=1180) would still need Dream split across two cards.

### Headline metric

**`tokens_per_second` = output tokens (before the first stop token) / wall
time, batch 1.** This is what exp04 already reports, and it matches vLLM's and
Fast-dLLM's aggregation (`sum tokens / sum time` over the corpus, never a mean
of per-task rates). With early stop it is also the honest measure of the
work saved. Report `dream_rounds` (sequential Dream forwards) next to it as the
depth metric, and `canvas_tokens_per_second` is retired.

## F2. Canvas, early stop and the stop tokens

| benchmark | max canvas | exp04 SDAR canvas | notes |
| --- | ---: | ---: | --- |
| GSM8K | 256 | 256 | unchanged |
| HumanEval+ | 512 | 512 | unchanged |
| MBPP+ | 256 | 512 | exp04's SDAR arms: 0/100 (adaptive) and 0-2/100 (fixed) completions ran past 254 tokens. Dream's own fixed:k did at 7-45%, but Dream does not propose here. |
| MATH-500 | 1024 | 1024 | deferred |

**Early stop:** stop when a stop token is committed at position `e` and every
position `< e` is committed. Return `canvas[:e]`. With no remasking this is
lossless, because committed tokens never change and everything from `e` on was
discarded before scoring. exp04 decoded 42% (GSM8K), 78% (HumanEval+) and 93%
(MBPP+ at canvas 512) of its canvas after the first EOS. With remasking (F3),
run the remasking pass on the block's committed tokens first, and stop only if
nothing before `e` was remasked.

### Stop tokens: honor both, truncate at the first

What SDAR itself does:

- `generation_config.json` of SDAR-1.7B-Chat-b64 declares
  `eos_token_id: [151645, 151643]`, i.e. `<|im_end|>` and `<|endoftext|>`.
- SDAR's official `generate.py` (JetAstra/SDAR) sets
  `stopping_criteria_idx = gen_cfg.eos_token_id` when none is given. After each
  **block** finishes it breaks if **any** of those IDs appears in the generated
  region. It returns the canvas untruncated and decodes with
  `skip_special_tokens=False`.
- Dream's `generation_config.json` declares only 151643.

What the exp04 SDAR canvases actually contain (all 76 traced tasks in
`exp04_autospeculation/report/data/trajectories`, rebuilt from the decision
traces):

| ending | tasks |
| --- | ---: |
| `<|im_end|>` `\n` `<|endoftext|>` `<|endoftext|>`... | 70 |
| `<|im_end|>` `<|im_end|>` `\n` `<|endoftext|>`... | 2 |
| `<|im_end|>` `<|endoftext|>`... | 1 |
| neither (ran to the canvas limit) | 3 |

So the two tokens are usually **not** adjacent. There is a `\n` between them;
the trajectory player just does not show it. This is the Qwen chat template's
turn terminator, `<|im_end|>\n`, followed by padding. The answer ends at
`<|im_end|>`.

**Rule:** the stop set is SDAR's declared `eos_token_id`, `{151645, 151643}`.
It applies to every arm, including any Dream-only baseline, so all arms share one
definition. The output is truncated at the **first** occurrence of either token,
and output tokens are counted before it. It is recorded as `stop_token_ids` in
each run's metadata. Why this rule:

- It is the model's own declaration and the official inference code's behavior,
  so it needs no judgement of ours.
- It does not depend on the two tokens being adjacent, and it handles the three
  odd endings above.
- Our stop condition is a finer version of SDAR's. SDAR stops at the end of the
  block that contains a stop token; we stop as soon as the stop token and
  everything before it are committed. The returned text is the same, and we skip
  the rest of the block.

Effect against exp04, which truncated at `<|endoftext|>` only: the scored text
loses only a trailing `\n`, since `<|im_end|>` was already dropped by
`skip_special_tokens=True`. Rescoring all 900 exp04 GSM8K records both ways
changed **0** scores. Token counts drop by about 2 per task. This matters when
comparing tokens/s with exp04, so exp04 numbers are recomputed under the new
rule rather than quoted.

## F3. Remasking: sample one weak token, replace it with Dream's argmax

**Trigger.** Exactly one pass per block, run before SDAR's commit forward,
whichever of these happens first:

- the block is complete (every position committed), or
- the early-stop condition holds inside it (F2): a stop token is committed at
  `e` and every position before `e` is committed. This is the common case for
  short answers that never reach 64 tokens. The final, partial block is
  **always** checked before the answer is returned.

A block that holds the prompt's remainder has fewer than 64 canvas positions;
it is checked the same way. Because every pass runs before SDAR's commit,
SDAR's cache holds only earlier blocks and nothing needs rolling back.

**Pass:** for every committed canvas token `z_i` in the block (up to 64;
at the early-stop point, only positions `<= e`), one row masks `z_i` and reads
Dream's full distribution `p_D(. | z^{-i})` over the shared vocabulary (F4),
with everything else visible: prompt, earlier blocks, the rest of this block and
the masked future. Rows run in groups of 4 like the verifier's. All of them
run, because ranking needs every `p_i`. Each row gives two things:

- `p_i = p_D(z_i | z^{-i})`, the committed token's probability, and
- `a_i = argmax_v p_D(v | z^{-i})`, Dream's preferred token at `i`.

**Rule:**

1. Take the 4 positions with the lowest `p_i`.
2. Draw one of them with probability proportional to `1 / p_i`, normalized over
   the 4.
3. Remask it and commit `a_i` there immediately. There is no SDAR re-proposal
   and no extra forward, since `a_i` comes from the same row.

That is one remasking pass per block, changing at most one token. The draw uses
the per-task seed stream, so runs stay reproducible.

Notes on the rule:

- **Rank and weight by probability, not raw logit.** Raw logits are
  unnormalized: they shift by an arbitrary constant per position, so they cannot
  be compared across positions and can be negative, which leaves `1/logit`
  undefined. `p_i` (equivalently `log p_i`) is comparable. `1/p_i` is the `T=1`
  member of the family `p_i^{-1/T}`, if a softer or sharper draw is wanted later.
- **The draw is usually close to an argmin.** With `p = (1e-4, 0.01, 0.1, 0.3)`
  the weights are `(10000, 100, 10, 3.3)`, so the lowest is picked 99% of the
  time. The sampling only matters when the bottom 4 are close in probability.
- **Replacing one token keeps the change consistent.** `a_i` is Dream's best
  token given everything else exactly as it now stands. Changing two positions
  at once, each chosen assuming the other stays fixed, would bring back the
  parallel-factorization error adaptive exists to control.
- **Skip when Dream would change nothing.** Take the bottom 4 by `p_i` first,
  then drop any position where `a_i == z_i`: the committed token is already
  Dream's top choice, and `p_i` is low only because Dream's distribution there
  is flat. If all 4 are dropped, nothing is remasked and decoding moves on.
  Otherwise draw one of the survivors with weight `1/p_i`, renormalized over the
  survivors. Log how often the pass is skipped.
- **Stop tokens count as one symbol (added 2026-09-19, before launch).** In the
  pre-launch smoke test, every stop-point pass was spent swapping SDAR's
  `<|im_end|>` for `<|endoftext|>`. Dream's only EOS is `<|endoftext|>`, so it
  always finds `<|im_end|>` improbable. The swap ends the answer at the same
  place, so it wasted the block's one remask. Now:
  - a committed stop token is scored by Dream's total probability on the whole
    stop set;
  - a stop-for-stop swap is not a change;
  - stop to content, and content to stop, still are.

  Re-run on the same 3 tasks, the stop-point passes were skipped and the real
  edits were unchanged.
- **Early stop.** The replacement is a committed token, so it never creates a
  mask before `e`. Three cases:
  - it changed a token before `e`: the stop condition still holds, return;
  - it turned an earlier token into a stop token: `e` moves earlier, return;
  - it replaced the stop token at `e` itself: the answer no longer ends there,
    so keep decoding the block. The pass has been used, so the next stop point
    in this block returns without another pass.
- **Cost: this pass is as expensive as Dream decoding the block by itself.**
  It needs one row per committed token, and ranking needs all of them, so early
  exit cannot help. 64 rows is about 6.8 s at S=400, or about 0.106 s per
  token. Dream `fixed:1` spends about 0.13 s per token. On its own the pass
  keeps the whole method slower than Dream `fixed:1` in tokens/s.
  **Accepted for this experiment:** the full pass. The cheaper shortlist
  below is recorded as a deferred ablation, not implemented here.

  **Deferred: shortlist of 8 (not in exp05).** The idea: pick the 8 committed
  tokens with the lowest commit-time Dream probability `q_i`, and re-score only
  those with full context (about 0.85 s instead of about 6.8 s at S=400).
  **Why it is not used:** `q_i` comes from the verifier row that accepted the
  token. That row sees earlier blocks and the candidates ranked above the token
  in its own step, but **not the tokens committed later in the block.** Seeing
  the later part of the block is the point of the remasking pass, and
  pre-selecting on `q_i` would drop exactly the tokens that only look wrong in
  hindsight. If it is revisited: `q_i` is already computed every step
  (`conditional` in `_score_all_prefixes`, `adaptive_re/scheduler.py`) and is
  only thrown away, so keeping a per-block buffer `q[64]` costs no forwards.
  Measure the miss rate against the full pass before adopting it.
- **More than one change per pass.** Replacing the top `r` draws instead of 1
  (sampled without replacement) is the obvious knob if the pilot shows the pass
  is worth more. `r > 1` reintroduces the joint-change caveat above.

Logged per checked token: `p_i`, `a_i`, the rank, and the commit-time
conditional (from the verifier row that accepted `z_i`). Logged per pass: the
drawn position, old and new token, and whether it was a no-op.

Deferred ablations: remasking and re-decoding through SDAR, `r > 1`,
re-checking earlier blocks, and the cheap subset-masking variant (8 rows each
masking 8 positions).

## F4. Dream as verifier

**Acceptance ratio (Dream-only):**
`err_j = | sum_{i<=j} [log p_D(y_i | c, y_<i) - log p_D(y_i | c)] |`. Accept the
longest prefix with `err_j <= log(1 + delta)`. This is Dream's own measure of
how much the proposed tokens depend on each other. SDAR's probabilities are
logged but not used.

**Veto on the first candidate.** In exp04, candidate 1 was always committed
(`passing[0] = True`). Now, if SDAR's top candidate `y_1` differs from Dream's
argmax at that position (read from row 0, so it costs nothing), commit **Dream's
argmax there and nothing else** this step. The other rows all contain SDAR's
`y_1`, so they no longer apply. Every step still commits at least one token, so
progress is guaranteed. This applies at T = 0, the exp04 recipe; a sampled
recipe would need a coin instead. Log `veto_count`.

A natural later variant is to apply the same check to every accepted position
(stop the prefix at the first SDAR/Dream argmax disagreement). It is not in
scope yet.

**Shared vocabulary.** IDs 0..151664 map to identical tokens in both tokenizers
(checked). Above that they differ, and SDAR's 151666 `</tool_response>` is
Dream's mask token. Both models therefore read logits as `logits[..., :151665]`
and renormalize over that range. This drops SDAR's
`<tool_response>`/`</tool_response>`/`<think>`/`</think>`/`<|MASK|>` and
Dream's `<|beginoftext|>`/`<|mask|>` plus its padding to 152064. Both mask tokens
fall outside the range, so the explicit mask exclusion is subsumed. In practice
this costs SDAR nothing: across all exp04 traced SDAR runs, no committed token
was ever >= 151665. It will be implemented as a provider option (`vocab_limit`)
with a load-time assertion that the two tokenizers agree on the shared range.

**Other bridging:**

- **EOS:** 151643 in both.
- **Prompts:** each model uses its own chat template. Canvas index `k` is SDAR
  position `P_S + k` and Dream position `P_D + k`.
- **Context:** SDAR proposes seeing earlier blocks plus the current one; Dream
  verifies seeing the whole canvas, with the future masked.

## Baselines to re-run from exp04

**Update 2026-09-19: B2 is dropped.** exp05 compares the Dream-verified arm
directly against SDAR fixed:k (B1) and exp04's Dream numbers. B2 was started
and stopped with 7 of 24 arms complete. Its outputs are kept, unused, in
`outputs/_archive/b2_stopped_2026-09-19/`. B2's workers had overwritten 7 of
B1's run-level `run_metadata_worker_*.json` files; those were rebuilt from B1's
plan and carry a `reconstructed` note. The per-arm `metadata.json` files were
never touched.

Every exp04 number was measured under rules exp05 changes:

- truncation at `<|endoftext|>` only (now the first of `{<|im_end|>, <|endoftext|>}`);
- the whole canvas decoded (now early stop);
- MBPP+ at canvas 512 (now 256);
- the full vocabulary (now IDs < 151665).

Its tokens/s are therefore not comparable. **Scores can carry over, but latency
cannot.** All arms that exp05 is compared against are re-run under the exp05
rules, on the same 100 tasks per benchmark (seed 271828), for GSM8K, HumanEval+
and MBPP+:

| # | arm | why | exp04 cost (GPU-h, 3 benchmarks) |
| --- | --- | --- | ---: |
| B1 | SDAR `fixed:{1,2,4,8}` | the proposer's own speed/quality frontier | 2.7 |

The costs are upper bounds: early stop removes 42-93% of the decoded canvas
for the SDAR arms, and MBPP+ halves its canvas.

**Dream is not re-run (decided 2026-09-18).** The Dream-7B reference is
exp04's deliverable 1 (`../exp04_autospeculation/outputs/final/1_dream7b_baseline`),
used as is. Read it with these differences in mind:

- **Recipe:** Dream/ADJUST (T=0.1, top-p 0.9, entropy ordering), not the greedy
  logp recipe Dream follows as verifier here.
- **Decoding and stopping:** the whole canvas is decoded, and truncation is at
  `<|endoftext|>` only.
- **MBPP+ canvas:** 512, not 256.

Its **scores** are the quality reference: they match Dream's published
numbers. Its **timings** are not comparable to exp05's early-stopped arms
and should not be put on the same tokens/s axis without that caveat.

**B1/B2 double as a correctness test of the new stop code.** SDAR is
block-causal: a block never attends to later blocks, so neither the canvas
length nor stopping early can change a token before the stop. Under exp05
rules, B1 must therefore reproduce exp04's SDAR completions **token for token**
up to the stop token, on GSM8K, HumanEval+ and on the MBPP+ tasks that ended
within 256 tokens (all but 1-2 per fixed arm). B2 should too. exp04 records keep
the decoded text, not token IDs, so the check compares `raw_completion` after
stripping trailing whitespace. The new stop rule drops exactly
`<|im_end|>` (already skipped when decoding) and the `\n` after it. It also
compares `eos_position - 2` against the new stop position for the 70/73
`<|im_end|>\n<|endoftext|>` endings. The one allowed
source of divergence is a borderline adaptive acceptance flipped by the
vocabulary renormalization, which changes log-ratios in their last bits. Every
divergence gets listed and explained before B2's numbers are used.

**Not re-run:**

- Algorithm 3 (dropped).
- exp04's T=1 reference (only existed for the autospec confound).
- MATH-500 (deferred).
- Dream-7B `fixed:k`: exp04's numbers are used as they are (see above), and so is
  `reference/dream7b_frontier.json`.

**Timing placement.** Every timed arm gets its cards to itself:

- B1/B2 use one process per GPU (`launch_sdar_baselines.sh`).
- The cross arms use `LAYOUT=exclusive` in `launch_cross.sh`: SDAR and Dream
  each on their own card, the two cards sharing one PCIe switch (GPU0+1,
  GPU2+3).
- `LAYOUT=shared` (three SDAR copies on one card, 3 workers) is for pilots and
  ablations only, because it contends SDAR's share of the latency.

## What to record

On top of exp04's record fields:

- `dream_rounds`, `dream_rows`, `sdar_passes`
- `at_cap_rate` (64-token block cap binding), `veto_count`, `remask_skipped`
- `remask_passes`, `tokens_changed_by_remask`, `remask_noops`
- `stop_position`, `stop_token`, `blocks_skipped_by_early_stop`

The traces carry the remasking statistics per checked token (F3).

## Before implementing

1. Build the provider (Dream alone on one card, SDAR on a shared card) and
   measure the Dream card's peak at S=660 with 4 rows, and the shared SDAR card with 3 copies.
2. ~~Implement early stop, the stop set and the vocabulary limit, then run B1
   and check it against exp04.~~ Done: the smoke test matched exp04 on 24/24
   completions. The full B1 run is in progress; its comparison is written to
   `logs/baseline_b1/compare_exp04.txt`. (B2 was dropped.)
3. Pilot 4 tasks per benchmark and check `at_cap_rate`, `veto_count`, remasking
   changes and no-op rate and s/task against the estimates in F1.
