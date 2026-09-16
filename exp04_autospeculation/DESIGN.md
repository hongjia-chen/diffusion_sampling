# Algorithm 3: design choices

**Settled:** D1 (T = 1.0, no top-p), D2 (uniform random sigma, autospec only),
D3 (lockstep batching, faithful), D4 (physical rounds are the headline),
D5 (canvas sized so the root block is one physical round).
**Open:** D6 (`rho` sweep), D7 (whether to re-run baselines at T = 1),
D8 (SDAR-1.7B-Chat-b64; needs the block-loop harness change).

**Implemented.** `autospec/autospeculation.py`. Every function that needs the
model is a generator that yields a forward request; `_interleave` merges
sub-generators so the rejection batch pools into one request recursively, and
`_drive` is the only place that touches the provider. One `_drive` iteration is
one parallel round however many blocks it contains; the chunking inside
`_forward_pooled` is what turns it into physical rounds.

One consequence worth knowing: `RS2` returns `log mu_S` alongside the block it
filled and `FallbackSampler` composes the halves by the chain rule, so
evaluating a rejection candidate's `dnu/dmu` needs **no forward pass at all** --
`mu` arrives with the candidate and `nu` is a lookup into the block's cached
marginals. Only *drawing* candidates costs anything.

This is the agenda, written down so the choices get argued against the actual
recursion rather than against a summary of it. The
transcription of the algorithm lives in `autospec/autospeculation.py`.

Each decision below changes the shape of the code, not a constant inside it,
which is why the sampler is a stub until they are settled.

---

## D1. What distribution is this exact *with respect to*?  — RESOLVED: T = 1.0, no top-p

Algorithm 3's selling point is exactness: Proposition 18 says the output is
distributed as `mu`, with no tolerance parameter. Two things complicate that
here.

**The oracle is approximate and inconsistent.** The paper assumes a genuine
conditional-marginal oracle. A masked diffusion LM is not one: the joint
implied by conditioning in one order differs from the joint implied by another.
That inconsistency is exactly the quantity `adaptive:delta` thresholds. So the
honest claim is narrower:

> Algorithm 3 returns a sample from the distribution that **plain
> one-token-at-a-time ancestral sampling in the order sigma** would have
> produced — using fewer sequential rounds.

That is still a real and *testable* claim, and it gives the correctness test:
autospec's output distribution must match sequential ancestral sampling under
the same sigma, on a small enough canvas to estimate both.

**Temperature.** The coin `min(1, dmu/dnu)` is meaningful only if the proposer
and the verifier score the *same* distribution. The previous experiment
deliberately broke this (README divergence 3: propose under nucleus, score
unfiltered), which is fine for a threshold heuristic and fatal for an exact
coin. That is the one hard constraint, and any fixed `T > 0` satisfies it.

`T = 1` is **not** required for correctness. Algorithm 3 stays exact under
tempered conditionals: take `mu_S` to be the tempered chain-rule joint and
`nu_S` the product of tempered per-position conditionals, and `dmu_S/dnu_S` is
still the true likelihood ratio between what is sampled in parallel and what
`FallbackSampler` produces. Prop. 18 needs the ratio to be genuine; it never
needs `nu` to be the marginals of `mu`.

What `T = 1` actually buys, in three separable pieces:

1. **It is the only T at which the target can be named.** Tempering the
   conditionals does not temper the joint. With `beta = 1/T`,

       mu_sigma^beta(x) = mu(x)^beta / prod_s Z_s(x_sigma(<s)),
       Z_s(c) = sum_y p(y | c)^beta

   The numerator is order-free; the denominator is path-dependent, so the
   target depends on `sigma`. This is not model error -- on a 2-variable oracle
   whose conditionals are exactly consistent, `beta = 2` already makes the two
   orders disagree, and both differ from `mu^beta/Z`. At `beta = 1` all three
   coincide exactly. So at `T != 1`, "exact" means exact with respect to an
   object with no characterization independent of `sigma`. The correctness test
   survives either way: match sequential ancestral sampling at the same T and
   the same sigma.

2. **It is what makes the speculation simultaneously cheap and good.** At
   `T = 1` a single forward pass on the masked canvas returns, at every
   position at once, the true marginal of `mu_S`. That coincidence is the
   "auto" in autospeculation: one oracle call yields the whole product
   speculation, and that speculation is the best product approximation to
   `mu_S` -- which is exactly what the pinning-lemma bounds (Lemma 13) control
   to reach `O(sqrt(n))` rounds. Tempering breaks the identity, so `nu` is
   neither optimal nor covered by any bound. Correctness survives; the round
   bound does not, and rejection rates then have no theory to be compared to.

3. **`T -> 0` is a different algorithm, not a limiting case.** `nu_S` becomes a
   point mass at the per-position argmaxes (what `fixed:|S|` commits) and
   `mu_S` a point mass at the sequential greedy chain in order `sigma`. The
   coin degenerates to 0/1 -- accept iff they agree -- and in the fallback
   branch every `Y(i)` is the same deterministic chain, so `p(i) = 1` and the
   geometric batch loop collapses to one call. What remains is recursive
   verify-and-split that provably reproduces sequential greedy: a lossless
   greedy accelerator, genuinely interesting, but with the speculative
   rejection sampling -- the paper's contribution -- deleted.

Nucleus filtering is separate and worse: the conditional and marginal nuclei
truncate to different supports, so `nu` and `mu` disagree about what is
possible. Correctness still survives (nothing divides by zero -- `X` is drawn
from `nu`, `Y` from `mu`), but the speculation quality is wrecked.

*Recommendation: run autospec at `T = 1.0`, no top-p -- because that is where
it is the paper's algorithm and the analysis applies, not because lower T is
invalid. A T sweep is a legitimate ablation, just one that measures a different
target at each T. Keep a greedy `fixed:1` arm as a reference point to the
previous experiment.*

## D2. The permutation `sigma` vs confidence ordering  — RESOLVED: uniform random sigma

The algorithm draws `sigma` uniformly at random **up front, independent of the
sample**. Both the exactness argument and the `O(sqrt(n))` round bound need
that independence.

Every scheduler in the previous experiment instead commits the *most confident*
positions — a data-dependent order, which is most of why Dream works at all.
The two cannot be combined: a confidence-chosen order is a function of `Z`, and
conditioning on it invalidates the coin.

- (a) Faithful: uniform random `sigma`, fixed before generation. Theory holds.
  Quality will be materially worse than confidence ordering, and some of that
  gap is the *order*, not the sampler.
- (b) Confidence-ordered `sigma`: keeps quality, abandons the guarantee.
  Algorithm 3 becomes another heuristic and the experiment loses its point.
- (c) Faithful `sigma`, and add a **`fixed:1` under the same random `sigma`** as
  the quality reference. Then autospec-vs-that isolates the sampler, and
  that-vs-confidence-`fixed:1` prices the ordering separately.

**Decided: uniform random `sigma`, and it belongs to Algorithm 3 alone.**

Fixed-k and adaptive are *not* re-run in sigma order. They keep the scheduler,
the confidence ordering and the recipe of the previous experiment, unchanged
and untouched -- `autospec/scheduler.py` is byte-identical to the port. Only
Algorithm 3 consumes a permutation.

Implementation consequences, all live:

- `sigma_seed` is `sha256(sigma:<seed>:<example_id>)`, while `task_seed` is
  `sha256(<seed>:<method>:<example_id>)`. The permutation deliberately omits
  the method label so that two configurations of Algorithm 3 -- a `rho` sweep,
  say -- walk the same permutation on the same task and remain a paired
  comparison rather than a comparison of two random orders.
- Drawn on a CPU `torch.Generator`, so it is identical across devices and does
  not consume from the global RNG stream the token draws use.
- Recorded per record as `sigma_seed`, and per method as `position_order`.

**The correctness reference is a unit test, not a benchmark arm.** Algorithm 3
targets the sigma-ordered chain-rule joint by construction -- that is what the
numerator of `dmu_S/dnu_S` is. Checking it does not need a GPU arm: on a small
synthetic oracle with a tiny canvas the exact sigma-ordered joint can be
enumerated and compared against the sampler's empirical distribution directly,
which is a far stronger check than agreement with another sampler's outputs.
It lives in `tests/test_autospeculation.py`.

**Known, accepted confound.** Autospec walks a random order while fixed-k and
adaptive rank by confidence, so a benchmark quality gap between them mixes the
sampler with the position order. This is not engineered around; it is stated.
See D7.

## D3. What maps onto the GPU batch dimension  — RESOLVED: (a) lockstep batching

The inner loop is unbounded (`r = 0, 1, 2, ...`) and each batch member is an
independent recursive `FallbackSampler` call. On real parallel hardware that is
one round; on one GPU it is only one round if the members genuinely run as rows
of a single forward pass.

They can be batched — each member is an independent canvas — but their control
flow diverges immediately, since each makes its own accept/reject decisions.
So:

- (a) Lockstep batching with masking: run `B` fallback samplers as `B` rows,
  masking out members that have finished. Real parallelism, significant
  complexity, and wasted rows as members diverge.
- (b) Serialize batch members. Simple and obviously correct, but then the
  "parallel" batch is sequential and wall-clock will be far worse than the
  sequential sampler while round-complexity looks great.
- (c) Serialize for now, instrument round-complexity properly (D4), and treat
  lockstep batching as a follow-up if the round counts justify it.

The rejection loop has no bound. A cap `r_max` with a deterministic fallback
reintroduces bias, so the faithful implementation must continue until a
residual candidate is accepted.

**Decided: (a), lockstep batching with masking.** The inner loop's `B` fallback
samplers run as `B` rows of one forward, with finished members masked out. This
is the expensive path to implement and the only one that keeps the experiment
honest: under (b) or (c) the "parallel" batch is sequential, so a physical round
count (D4) would measure the implementation rather than the algorithm.

What it commits us to:

- Members diverge immediately -- each makes its own accept/reject decisions --
  so the batch carries wasted rows once some members finish. The waste is real
  cost and gets reported, not hidden.
- **The rejection batch and the block width compete for the same memory.** A
  block of size `m` already needs `m` verifier rows (D5); `B` concurrent
  fallback samplers multiply that. The contention is worst at the root, where
  the block is widest, and eases as the recursion halves blocks -- which is
  convenient, since `rho`'s geometric growth also happens later.
- The rejection loop is unbounded, as in Algorithm 3. Operational safeguards
  may abort a task, but must never substitute a non-residual sample into the
  results.

## D4. What counts as a round  — RESOLVED: physical rounds are the headline

Wall-clock on one GPU will make Algorithm 3 look bad for reasons that have
nothing to do with the algorithm (D3b). The previous experiment's metrics
(`denoising_steps`, `verifier_rows`, `wall_time_seconds`) do not distinguish
sequential depth from total volume, which is the entire distinction the paper
turns on.

Needed: **two** counters, `oracle_rounds` (the depth of the dependency graph —
the paper's metric, hardware-independent) and `oracle_queries` (total rows
forwarded). Both have to be defined for fixed-k and adaptive too, or the arms
cannot be compared on either.

Open: does chunking a block across verifier microbatches count as one round or
several? Physically several; theoretically one. *Recommendation: count both,
report the theoretical one as the headline and the physical one as batching-
dependent — the split the previous README already uses.*

## D5. Computing `dmu/dnu` for a long block  — RESOLVED: canvas sized to one round

For a block of size `m`, the numerator needs `pi(X_s | Z, X_<s)` for every `s`.
Those are `m` rows of one batched forward -- exactly what
`_score_all_prefixes` already builds, so this part is free to write.

It is not free to run. The `m` probe rows are `m` *different* sequences, so
scoring a block is a genuine batch-`m` forward over the full `S ~ 640`
prompt+canvas. What that costs, at `m = 512`, `S = 640`:

| | Dream-7B | ~2B (H=2048, I=5632) |
| --- | ---: | ---: |
| weights | 14.19 GiB | 3.73 GiB |
| `[B,S,H]` hidden output | 2.19 | 1.25 |
| per-layer transient (MLP gate+up dominate) | **30.31** | **11.25** |
| logits `[B,V]` fp32 log-softmax | 0.44 | 0.44 |
| max verifier rows on a 16 GiB A4000 | **8** | **434** |

The blocker is activations, not logits -- the vocabulary projection is already
lean (one position per row) and chunkable. The model predicts 8 rows for
Dream-7B, which is the microbatch the previous experiment settled on, so it is
calibrated rather than assumed.

**Model size decides whether the root block is one round or several.** At ~2B
the whole canvas nearly fits in one batch (434 of 512 at `S = 640`; the root is
the only block that wide, since one level of recursion halves it). At 7B it
takes 64 chunks, and the reported round counts would be simulated rather than
executed.

**But fitting is not winning.** The previous experiment's own timings show the
card is compute-bound at batch 1: MBPP+ latency is exactly linear in step count
(133.8/16.7 = 8.01 for an 8x step reduction) and 133.8 s / 512 steps = 261 ms
per forward = 41 TFLOP/s, about 54% MFU. A masked diffusion forward already
processes the whole canvas every step, so a 512-row batch costs the same
wall-clock as 512 sequential forwards.

This is the structural difference from autoregressive speculative decoding,
where verification is nearly free because decode is memory-bound at batch 1 and
the batch dimension rides in the bandwidth shadow. There is no such shadow here.

So in its **best** case -- root block accepted on the first coin -- Algorithm 3
spends 1 proposer pass + 512 verifier rows against sequential's 512 forwards.
It cannot beat sequential in single-GPU wall-clock. It buys 2 rounds instead of
512, which is exactly the paper's claim: query complexity stays `~O(n)`, round
complexity drops to `~O(sqrt(n))`. Whether rounds convert to seconds is a
hardware question -- see D3 and D4.

One caveat on the table: the transient estimate assumes SDPA/flash attention
with no `[B, heads, S, S]` materialization. At `B = 434`, `S = 640` an eager
attention implementation needs ~5.7 GiB *per layer* and none of this holds.
Check the chosen checkpoint's remote code for that specifically.

Secondary interaction with early exit: the adaptive scheduler can stop at the
first violating microbatch, whereas the exact coin needs the **whole** product
before it can flip. Autospec therefore pays full block cost where adaptive paid
partial. That is a real and reportable cost difference, not a bug.

### Decided: size the canvas so the root block is one physical round

Given D4, chunking the root block is a measured cost, so the canvas is chosen
to avoid paying it where the benchmark allows. Largest canvas whose root block
fits in a single batch at ~2B on a 16 GiB A4000:

| canvas | prompt | S | rows needed | rows that fit | |
| ---: | ---: | ---: | ---: | ---: | --- |
| 256 | 130 | 386 | 256 | 705 | one round |
| 384 | 130 | 514 | 384 | 537 | one round |
| 512 | 130 | 642 | 512 | 433 | chunks into 2 |

At 7B the same table reads 13, 10 and 8 rows that fit -- the root block chunks
into 20 to 64 pieces at every canvas length. **7B cannot execute a single-round
root block at any usable canvas.** That is the hard argument for the smaller
model; memory headroom was only the soft one.

But the canvas also has to fit the answer, and measured reference lengths
(Dream tokenizer) say 256 is not universally safe:

| benchmark | mean | p90 | p95 | max | > 256 | > 384 | > 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K reference answer | 124 | 197 | 221 | 413 | 2.1% | 0.2% | -- |
| MATH-500 reference solution | 219 | 468 | 594 | 1389 | **28.6%** | 13.6% | 8.2% |

So:

- **GSM8K 256.** One round, 2.1% of references exceed it, unchanged from the
  previous experiment.
- **MBPP+ / HumanEval+ 256.** One round. Measured completions at canvas 512 ran
  18-45 tokens under the greedy recipe and up to a 260-token mean under the
  sampled one, with `maximum_length_rate` at 0.00-0.01 throughout; 256 should
  hold at T=1 but is not verified, and `maximum_length_rate` is the check.
- **MATH-500 384.** The largest canvas that is still one round. A 256 canvas
  would truncate the reference solution for 28.6% of problems, which scores the
  canvas rather than the sampler; 384 halves that to 13.6%. Going to 512 buys
  8.2% at the cost of a second physical round at the root, on the one benchmark
  where the root block matters most.

## D6. `rho` and the base case

`rho = 1/log2(n)` is 1/9 at `n = 512`, so batches grow slowly: rounds 0-6 are
single members. Worth exposing as `autospec:<rho>` and sweeping, since it is
the one knob the paper leaves free. The recursion bottoms out at `|S| = 1`,
where `u = 1` always and `RS2` degenerates to ancestral sampling — depth
`log2(512) = 9`.

## D7. What autospec is compared against

Given D1 and D2, and with fixed-k and adaptive left exactly as they were:

| Arm | Answers | Order | T |
| --- | --- | --- | --- |
| `fixed:1` | best-quality, slowest reference | confidence | 0.0 |
| `fixed:k`, k>1 | the speed/quality curve autospec must beat | confidence | 0.0 |
| `adaptive:delta` | the same speculation, accepted by threshold not coin | confidence | 0.0 |
| `autospec:rho` | this paper | random sigma | 1.0 |

`adaptive` is the interesting row: it builds the *same* `nu` from the *same*
ratio, and differs only in accepting a prefix heuristically rather than a block
exactly. That comparison is the experiment.

Two confounds are now known and unresolved, and both are recipe rather than
scheduler:

1. **Order.** Autospec is random-sigma; the baselines are confidence-ordered.
2. **Temperature.** Autospec is forced to T=1.0 (D1); the baselines default to
   the previous experiment's greedy recipe.

Either can be closed by a second baseline pass -- `TEMPERATURE=1.0` into its
own `OUTPUT_ROOT` -- at the cost of re-running the baselines. Nothing about
the design forces that choice now, and it can be made after the first results
show whether the gap is large enough to care about. What must not happen is
the two roots being read as one, which is what `IMPORTED_FROM.txt` and the
`position_order` / `temperature` fields in each record exist to prevent.



## D8. The model  — OPEN: SDAR-1.7B-Chat-b64, and a block-loop harness change

Proposed: SDAR-1.7B-Chat (arXiv:2510.06303). Config: hidden 2048, 28 layers,
intermediate 6144, 16 heads / 8 KV heads, vocab 151936, bf16, Qwen3 derived --
almost exactly the "~2B" column D5 sized its memory table on.

**Block diffusion is usable; it caps `n` at the block size.** SDAR is
autoregressive *across* blocks and a masked diffusion model *within* a block, so
within a block it provides exactly the any-order conditional-marginal oracle
Algorithm 3 assumes. Algorithm 3 applies per block with `n = block_length`, and
the canvas becomes a loop over blocks. The default `block_length = 4` leaves
nothing to win (best case 2 rounds against 4). SDAR ships per-block-size
checkpoints `-b8 -b16 -b32 -b64`.

### Why b64: the block size must not censor the adaptive sweep

**The selection criterion is not speed.** Choosing the block size to maximize
Algorithm 3's speedup would tune the setup to flatter the method under test, and
it would specifically destroy D4 -- the point of measuring physical against
theoretical rounds is to find the theory-practice gap, so picking B to minimize
that gap is choosing the setup to make the answer come out well. The criterion
is: **what makes all three arms measurable and comparable to the previous
experiment.**

Under SDAR, the block size B replaces `max_candidates`: no scheduler can commit
more than B tokens, because a block is all there is. The previous experiment
raised `max_candidates` to 64 precisely because smaller caps censored the
adaptive sweep -- at a cap of 8 it bound 20.5% of GSM8K steps and 68.9% of
schema JSON steps at delta 1.5, and at 32 it still bound 14%. Its measured
commitment widths say directly which block sizes would repeat that mistake:

| tokens committed per step (Dream, cap 64) | GSM8K | MBPP+ | schema JSON |
| --- | ---: | ---: | ---: |
| delta 0.10 | 2.71 | 4.06 | 4.48 |
| delta 0.50 | 6.28 | 9.11 | 10.93 |
| delta 1.50 | 10.55 | 14.81 | 16.49 |

| block size | adaptive arms committing >= B/2 per step |
| ---: | --- |
| 8 | 21 of 24 -- the block is smaller than the mean commitment; fully censored |
| 16 | 11 of 24, and the observed max (16.5) exceeds B outright |
| 32 | 1 of 24 -- borderline; MBPP+ at delta 1.5 sits at 46% of the block |
| 64 | 0 of 24 -- uncensored, and reproduces the prior cap exactly |

At B=16, `adaptive:1.50` would commit the whole block nearly every step and
become indistinguishable from `fixed:16`: the top of the delta sweep collapses
and the arm stops measuring the delta criterion at all. **b64 is the only SDAR
block size that keeps the adaptive sweep uncensored**, and it is also the
largest SDAR offers, so it is the closest approach to the unbounded-canvas dLLM
regime the Dream experiment ran in. That is the bridge.

Caveat: those widths are Dream-7B's. SDAR-1.7B will commit differently, so the
censoring check has to be re-run on a pilot -- `at_cap_rate` and
`mean_committed_tokens_per_step` are already in the summary for exactly this.

### What b64 is predicted to cost, as a result rather than a criterion

A forward costs `max(weight read, compute)`. Weight read is fixed at 7.6 ms for
1.7B on a 16 GiB A4000; compute scales with rows x tokens, so a bigger block
means the free ride on extra rows ends sooner:

| B | rows that ride free | rounds | sequential | Algorithm 3 best case | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 11.2 | 2 vs 8 | 60.7 ms | 15.2 ms | 4.00x |
| 16 | 5.6 | 2 vs 16 | 121.4 ms | 29.3 ms | 4.14x |
| 32 | 2.8 | 2 vs 32 | 242.9 ms | 94.6 ms | 2.57x |
| 64 | 1.4 | 2 vs 64 | 485.7 ms | 355.7 ms | 1.37x |

So at b64 the round saving is largest (2 instead of 64) and the wall-clock
saving is smallest (1.37x). **That is the finding, not a reason to move.** It is
a precise, quantitative instance of the question D4 exists to ask: an
`O(sqrt(n))` round bound does not convert to wall clock once the batch it
depends on is compute-bound. Reporting it at b64, where the divergence is
widest, is more informative than hiding it at b16.

### Risk, and the fallback

The SDAR paper (5.2.2) finds "smaller models (1.7B, 4B) exhibit significant
sensitivity to increases in block size, with performance degradation becoming
pronounced for B > 4." This shifts every arm equally, so the *comparison* stays
valid; the risk is **flooring** -- if 1.7B at b64 scores near zero on MATH-500
or MBPP+, the quality axis carries no information and only the cost axis
survives.

Mitigation, in order: pilot b64 on a small sample and check absolute scores are
non-trivial; if floored, either drop to b32 and accept mild censoring at the top
of the delta sweep, or move to SDAR-4B-Chat-b64 (4B bf16 is ~8 GB, comfortable
on a 16 GiB card) which the same analysis places in a less sensitive regime.

### Contrast with Dream, and why D5's conclusion was model-specific

Dream is a full masked diffusion LM with bidirectional attention over
prompt+canvas and **no KV cache** -- there cannot be one, because every step
rewrites tokens at arbitrary canvas positions and under bidirectional attention
that changes the keys and values every other position attends to, earlier ones
included. So the whole ~640-token sequence is re-encoded every step, 512 times.
That is measured, not assumed: 133.8 s / 512 steps = 261 ms per forward, with
latency exactly linear in step count.

SDAR's attention is causal across blocks, so previous blocks' K/V are fixed and
cacheable, and a forward re-encodes only the current block. 16 tokens instead of
640 is what moves it from compute-bound to memory-bound at batch 1, and that is
the whole reason batching a verifier is cheap here and was not on Dream.

### Cost: the harness has to change — now fully specified

Resolved against SDAR's own reference implementation
(`generate.py` in JetAstra/SDAR), so none of this is inferred:

**1. Loader.** `AutoModelForCausalLM`, not `AutoModel` (which resolves to the
backbone `SDARModel`, no `lm_head`). `Conventions` needs a loader field.

**2. Attention mask — block lower-triangular.** Verbatim from the reference:

```python
block_mask = torch.tril(torch.ones(num_blocks, num_blocks))
attn = block_mask.repeat_interleave(block_length, 0) \
                 .repeat_interleave(block_length, 1).unsqueeze(0)
```

Block i attends to every block j <= i, fully bidirectionally within each. The
prompt is **not** special-cased: `num_blocks = ceil((prompt_len + gen_len)/B)`
and the prompt simply occupies the first `prompt_len // B` blocks under the
same mask. A prompt that is not a multiple of B leaves its remainder inside the
first *decode* block, where only the still-masked positions get filled -- that
falls out of `mask_index = (cur_x == mask_id)` and needs no special handling.

**3. `position_ids` are passed explicitly** -- `torch.arange(total_length)`
sliced per block. They are not inferred from the cache.

**4. KV cache, and the `store_kv` flag.** The checkpoint's `modeling_sdar.py`
accepts a custom `store_kv` kwarg (confirmed present in the HF copy, not only
the training copy). The reference runs every denoising step with
`store_kv=False` so masked-token K/V never enters the cache, then issues **one
extra forward with `store_kv=True`** once the block is complete, to commit it.
So the cost of a block is `steps + 1` forwards, not `steps`.

**5. Logits are position-aligned** -- the reference samples `x0` from
`model(...).logits` and writes it back at the same indices, with no shift. So
SDAR is `logits_alignment="none"`, unlike Dream.

**6. The remasking strategies map onto ours exactly.** `low_confidence_static`
selects the top `block_length // steps` masked positions by confidence: that is
`fixed:k` with `confidence="logp"`, since the reference's confidence `x0_p` is
the probability of the sampled token. `low_confidence_dynamic` is the threshold
rule; `entropy_bounded` is a cumulative-entropy budget. Both are schedulers we
do not have.

**7. Stopping is at block granularity** -- the reference breaks out of the block
loop when a stop token appears anywhere in the generated span. D9 keeps the
fixed-canvas policy for comparability, so we do not use this, but it is what
makes the derived early-stopping number computable after the fact.

What remains to build is a block loop: `generate` currently treats the whole
canvas as one field, and under SDAR every sampler -- fixed, adaptive and
autospec -- runs inside a block, with the canvas a loop over blocks and a KV
cache carrying the committed prefix.

### Porting policy: what to copy, and two things not to

Copy everything in items 1-6 verbatim -- they are requirements of the
checkpoint, not design choices, and imitating them is what makes the comparison
high-fidelity. Note the source is **JetAstra/SDAR `generate.py`**, the canonical
SDAR inference. *SimSD is not a source for this*: it is draft/target speculative
decoding, i.e. the rival approach to autospeculation, useful as a comparison
point and not as code to port.

Two places where copying would be wrong:

**Their stopping rule (item 7).** Their block loop breaks when a stop token
appears. D9 keeps the fixed canvas instead, because Dream structurally cannot
stop early and an SDAR arm that stops would have latency partly determined by
where EOS lands rather than by its scheduler. Derive it afterwards instead.

**Their sampling function.** `sample_with_temperature_topk_topp` has no greedy
branch: `if temperature != 1.0: logits = logits / temperature`, so
`temperature=0` divides by zero. Greedy must therefore be spelled `top_k=1` --
and under `top_k=1` the returned `token_prob` is identically **1.0**, because
softmax after top-1 filtering puts all mass on the survivor. Every masked
position would report confidence 1.0, `torch.topk` would break ties by index,
and `low_confidence_static` would silently degenerate into left-to-right
sequential decoding.

That cannot be what produced their published numbers, and it contradicts the
paper's own description of confidence as "the highest probability assigned to
the most likely token". Our `_select_tokens` computes confidence from the
**unfiltered** distribution at temperature 0, which is that description taken
literally and matches Dream's `maskgit_plus`. Keep ours.

### Their four remasking strategies, and what ours measures instead

| strategy | rule | our analogue |
| --- | --- | --- |
| `sequential` | commit the next `k` positions from the first mask, left to right; confidence unused | none -- `fixed:k` in *index* order rather than confidence order |
| `low_confidence_static` | top-`k` by confidence, `k = B // steps` | **exactly our `fixed:k`** with `confidence="logp"` |
| `low_confidence_dynamic` | commit every position with confidence > tau, with a top-`k` floor for progress (tau = 0.85 in the script, 0.9 in the paper) | none -- worth adding as `threshold:tau` |
| `entropy_bounded` | sort by entropy ascending, take the longest prefix whose cumulative entropy is under 0.35 | closest in spirit to `adaptive:delta`, but see below |

**A remasking strategy IS a scheduler** -- the same slot our arms occupy. It
answers "which masked positions do I commit this step", which is exactly what
`fixed:k`, `adaptive:delta` and `autospec` answer. Their four are competitors to
ours, not a separate layer underneath them. What separates them is the
*information* each is allowed to use:

| | forwards per decision | information used |
| --- | --- | --- |
| their four | 1 | per-position marginals only |
| `adaptive:delta` | proposer + verifier | joint vs marginal factorization error |
| `autospec` | recursive, may reject and resample | the same, corrected exactly |

**All four decide from a single forward pass, which means all four see only
marginals.** `sequential` ignores the model entirely; the other three rank
positions by how confident the model is about each one *on its own*. None of
them can detect that two positions are individually confident but jointly
dependent -- which is the failure mode that makes parallel commitment lose
quality in the first place.

That is the cleanest statement of what this experiment adds. Their strategies
ask *"how confident is each position by itself?"*. `adaptive:delta` asks *"are
these positions actually independent?"*, by measuring the joint-vs-marginal
factorization error directly, and `autospec` asks the same question and then
corrects for the answer exactly rather than thresholding it.

**`entropy_bounded` as written does not compute an entropy.** `x0_p` is the
probability of the *selected* token, shape `[batch, block]`, so
`-(x0_p * x0_p.log()).sum(dim=-1)` sums over the block and yields one number
per batch, not per position -- and `-p log p` on a single scalar is not the
entropy of a distribution regardless. At batch 1 it then broadcasts that scalar
across every position, so the subsequent sort is arbitrary. Do not port it. Our
`confidence="entropy"` already computes the real thing,
`(probs * log probs).sum(-1)` over the full vocabulary.

### What SimSD actually uses, and what the SDAR paper calls its ceiling

SimSD's own SDAR configuration (`configs/experiments/sdar.yaml`) settles which
strategy pairs with argmax:

| method | sampling | remasking strategy |
| --- | --- | --- |
| `vanilla` (reproduces `generate.py`) | multinomial, T=1.0 | `low_confidence_dynamic`, tau 0.9 |
| `vanilla_cg` (its argmax twin) | argmax | **`low_confidence_static`** |
| `ours` (SimSD's method) | **argmax** | **`low_confidence_static`** |

Both argmax methods use static. That is our `fixed:k`, so the phase-2 baseline
matches what the most recent SDAR-based work runs. It also confirms the greedy
question independently: SimSD spells greedy `draft_sampling: argmax`, not
`top_k=1`, which is exactly the reading we adopted.

And the SDAR paper is explicit about which is the reference point: static
decoding "maximizes output quality, thereby establishing the **performance
ceiling** for the given architecture. All dynamic, early-exit strategies are
thus evaluated as trade-offs against this quality benchmark."

So the two are not competing choices, they are the two ends of the comparison:
`fixed:k` (static) gives the quality ceiling and the k-sweep frontier directly
comparable to Dream's, and `threshold:tau` (dynamic) would give SDAR's own
accelerator frontier. Phase 2 runs the first; the second is the natural arm to
add when autospec needs a native SDAR accelerator to beat.

### What does not exist there to port

Their four remasking strategies all decide from a **single** forward per step.
`adaptive:delta` and `autospec` instead need the *conditional* distributions --
several probe rows differing in which in-block positions are already filled --
so the verifier batch has no counterpart in their code and is ours to build.

Its cost is the KV cache, which must be replicated across the batch: at 28
layers, 8 KV heads and head_dim 128 in bf16 that is **112 KiB per token per
row**, so a 64-row verifier batch over a 704-token prefix holds **4.8 GiB** of
cache. With 3.4 GiB of weights it fits a 16 GiB card, but the replicated cache,
not the weights, is the dominant memory term -- the opposite of Dream.

### Environment: two things SDAR needs that this box cannot give it

Both found by trying to load the checkpoint, not by reading.

**1. `flash_attn` is an unguarded top-level import** in `modeling_sdar.py`
(`from flash_attn.ops.triton.layer_norm import rms_norm_fn`), so
`trust_remote_code` refuses to load the module at all without it. It cannot be
built here: no nvcc, and torch 2.13+cu130 has no prebuilt wheel.

**2. It declares transformers 4.52.4**, against the 4.46.2 that Dream declares
and that phase 1 is running on. The gap is not cosmetic -- ten API differences,
including a whole missing module (`transformers.modeling_layers`).

Neither is patched around by hand, because ten compatibility patches would be a
reimplementation rather than a port. Instead:

* `scripts/prepare_sdar_local.py` builds `models/<repo>-local`, symlinking the
  weights and applying **two** patches to `modeling_sdar.py`. Both fall back to
  the authors' own code: the pure-PyTorch RMSNorm is already in their file,
  commented out directly beneath the flash call with their note that it is
  equivalent; and the SDPA attention path is the branch they already use for
  prefill, computing the same exact attention. The script refuses to run if
  either patch fails to match, so a changed checkpoint cannot be silently
  mis-patched.
* A 20 MB `--no-deps` transformers 4.52.4 overlay in `vendor_env/tf452` goes on
  `PYTHONPATH` for SDAR workers only. **The base environment stays at 4.46.2**,
  so Dream and the running phase 1 are untouched. `ExpandedCache` subclasses
  `Cache`, which is an `nn.Module` in 4.46 and not in 4.52, so it bypasses
  `Module.__setattr__` and the test suite passes under both versions.

### Validated: the block loop reproduces the reference token-for-token

An independent transcription of `generate.py`'s block loop -- its mask, its
cache discipline, its `low_confidence_static` at k=1 -- was run against the real
SDAR-1.7B-Chat-b64 weights and compared with `generate_blockwise`:

| prompt | match | decoded |
| --- | --- | --- |
| "What is 2+2?" | token-for-token | `2 + 2 = 4` |
| "Name a colour." | token-for-token | `Blue.` |

Coherent output matters as much as the match: a wrong attention mask degrades
generation rather than breaking it, so clean arithmetic is evidence the
block-causal mask and `store_kv` discipline are right.

**Validation gate before trusting any SDAR number:** run `fixed:1` on the
**b4** checkpoint and check GSM8K lands near its published **80.1**. That is the
same check that validated the Dream path (measured 0.820 against published
81.0), and it costs about 0.1 GPU-h.

## D9. Shots, canvas policy, and SDAR's own decoding recipe

### Zero-shot, both new benchmarks — settled by both papers

Dream's Table 1 (base) carries shot counts: GSM8K (8), MATH (4), HumanEval (0),
MBPP (4). Its Table 2 (**instruct**, the checkpoint we use) carries none, which
is the usual convention for 0-shot. The empirical check settles it: our
zero-shot GSM8K measured **0.820** against Dream-Instruct's published **81.0**.
An 8-shot published number would not be reproduced to 0.01 by a 0-shot run.

SDAR's protocol is explicit and matches: "Mathematics: GSM8K (0-shot, CoT),
MATH-500 (0-shot, CoT)" and "Code Generation: HumanEval ... all evaluated in a
zero-shot setting."

So **0-shot for HumanEval+ and MATH-500**. It also keeps our four benchmarks
internally consistent (GSM8K is already 0-shot via lm-eval's `gsm8k_cot`), and
avoids the cost blowup from 4-shot MATH exemplars, which are long worked
solutions and would inflate prompt+canvas superlinearly.

### The canvas problem is real and known; here is what the field does

| family | needs a canvas? | what it does |
| --- | --- | --- |
| AR speculative decoding | **no** | generates until EOS; length is dynamic |
| full-canvas dLLM (Dream, LLaDA) | **yes** | fixed `gen_length` passed in, truncate at EOS afterwards |
| block diffusion (SDAR, BD3-LM) | **partly** | generate block by block, stop at a block boundary when a stop string appears |

Fast-dLLM's `var_generate` is the block-wise pattern in code: decode after each
block and `if any(stop_token in text): break`. That is the one real escape from
a fixed canvas, and **SDAR can do it while Dream structurally cannot** -- under
bidirectional full-canvas attention there is no "boundary" to stop at.

This is a decision, not a detail: under early stopping an arm that commits EOS
early becomes genuinely faster, so the ranking changes, and `early_eos_rate`
already ranges 0.00-0.68 across arms in this project.

**Recommendation: keep the fixed-canvas policy for the primary numbers** -- it
is what makes latency attributable to the scheduler rather than to output
length, and it is the only policy under which Dream and SDAR are comparable at
all. Then **derive** the early-stopping latency analytically rather than
running it twice: the block structure is deterministic and EOS position is
recorded, so if EOS lands in block j the early-stopping cost is the recorded
per-block cost summed over blocks 0..j. That needs per-block step counts in the
record, and yields both numbers from one run at zero extra GPU cost.

### SDAR's published recipe IS our `fixed:k`

From the paper, its two remasking strategies:

- **Static Low Confidence Remasking** -- "select the ceil(B/T) masked positions
  with the highest prediction confidence (i.e. the highest probability assigned
  to the most likely token)". That is exactly `fixed:k` with
  `confidence="logp"`, k = ceil(B/T). Their published protocol is "greedy
  static decoding ... block length and denoising steps set to 4", i.e.
  **`fixed:1`, greedy, logp**.
- **Dynamic Low Confidence Remasking** -- "select any masked position where the
  prediction confidence exceeds a predefined threshold tau", with a minimum of
  one position to guarantee progress, tau in {0.80, 0.85, 0.90, 0.95}.

So the fixed arm needs no adaptation: it is SDAR's own decoding strategy, and
running it makes our harness directly checkable against SDAR's published
numbers. The dynamic strategy is a **third scheduler family** we do not have --
a per-position confidence threshold, unrelated to `adaptive:delta`'s joint
factorization error. It is SDAR's native accelerator and therefore the natural
thing for autospec to beat on that model, the way the entropy recipe is Dream's.
Worth adding as `threshold:tau`.

SDAR recipe for our runs: **temperature 0.0, no top-p, confidence `logp`** --
which is already the harness default.

### Published numbers, and what they do to the framing

| | Dream-7B-Instruct | SDAR-1.7B-Chat (b4, greedy static) |
| --- | ---: | ---: |
| GSM8K | 81.0 | 80.1 |
| MATH-500 | 39.2 (full MATH) | **63.2** |
| HumanEval | 55.5 | **61.6** |
| MBPP | 58.8 | **61.1** |

**SDAR-1.7B is not a model downgrade.** It is built on Qwen3-1.7B, a far newer
base than Dream's Qwen2.5-7B, and it matches or beats Dream-7B on all four
while being 4.5x smaller. So "how much quality do we lose going 7B to 1.7B" may
have the answer "none" -- and the Dream frontier may be beatable on both axes
for reasons that have nothing to do with any sampler. That makes the
within-SDAR comparison the only load-bearing one, and makes it more important,
not less, to run `fixed:k` on SDAR.

Caveat: those numbers are at **b4**, and we run **b64**, where the paper warns
1.7B degrades. Our `fixed:k` run is the only way to learn where the model
actually sits at our configuration. A cheap validation exists: run `fixed:1` on
the **b4** checkpoint and check GSM8K lands near 0.80. ~0.1 GPU-h.
