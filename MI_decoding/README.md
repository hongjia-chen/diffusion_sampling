# MI_decoding: the MI commitment rule for SDAR, without remasking

The question: can the MI rule commit several tokens per SDAR step and keep
the best quality SDAR offers, at lower latency than SDAR's own fast decoders?

The decoding code is the shared [`../harness/`](../harness/README.md). This
folder holds the launchers, the ported exp06 references and the analysis.
Read [`../harness/AUDIT.md`](../harness/AUDIT.md) before comparing any number
here with SDAR's paper.

## The rule (`xadaptive:δ` in the runner)

At each step, one model proposes a token for every masked position in the
current block and ranks them by confidence. The verifier scores each candidate
`y_j` alone (`log p_V(y_j | c)`) and given the candidates before it
(`log p_V(y_j | c, y_<j)`). The step commits the longest prefix whose
cumulative log-ratio stays within `log(1 + δ)` in absolute value, and always at
least one token. There is no veto and no remasking. δ is the knob: the number
committed per step is not set, it emerges, and at most the rest of the block
can commit. In exp06 the 64-token cap bound in under 0.5% of steps.

## Arms

| arm | runner method | launcher |
|---|---|---|
| SDAR static, 1 token/step (B=4 is the published Table 2 setting) | `fixed:1` | `launch_baselines.sh` |
| SDAR static, k tokens/step | `fixed:k`, k ≤ block | `launch_baselines.sh` |
| SDAR dynamic (`low_confidence_dynamic`): every token with p > θ, at least 1 | `dynamic:θ` | `launch_baselines.sh` |
| **MI rule, 8B proposes / 1.7B verifies** | `xadaptive:δ` | `launch_mi.sh` (default) |
| MI rule, 1.7B proposes / 8B verifies | `xadaptive:δ` | `ROLES=1p7b_proposes launch_mi.sh` |
| **Dynamic first, then the MI rule** (2026-10-18): the candidates above 0.95 commit untested; the MI rule runs on the rest from the canvas with them filled (its first candidate commits untested too, as always) | `xadaptive-dyn95:δ` | `VARIANTS=xadaptive-dyn95 launch_mi.sh` |
| **Anchored band** (2026-10-20): the candidates above 0.95 all commit, all but the least confident one untested; that one is the MI rule's first candidate (the anchor), and the rule tests only the candidates above the floor (here 0.80), from the canvas with the untested ones filled. δ = 0 is dynamic 0.95, a δ that passes everything is dynamic 0.80 | `xadaptive-dyn95-band80:δ` | `VARIANTS=xadaptive-dyn95-band80 launch_mi.sh` |

**Block length is a property of the checkpoint.** SDAR ships b4 (the default,
`SDAR-*-Chat`), b8, b16, b32 and b64, and the config files do not record which
is which. The harness reads the block from the checkpoint name and refuses to
decode at any other length (`--allow-block-mismatch` overrides it, for
ablations only). It follows that `fixed:k` needs k ≤ block. At B=4 the static
frontier is k ∈ {1, 2, 4}: k=4 fills a whole block in one step, and k=8 cannot
exist, because the next block only ever attends to this one once it is fully
committed. Settings above 4 tokens per step need a b8+ checkpoint.
`launch_baselines.sh` drops k > block on its own.

Cost per block: `steps + 1` forwards, because a finished block needs one
forward to write its KV cache. At B=4 that is 5 forwards per 4 tokens for
`fixed:1` and 2 for `fixed:4`. At B=64 it is about 1/64 per token.

## Running

```bash
bash launch_b4_quality.sh        # B=4 anchors: SDAR-1.7B-Chat + SDAR-8B-Chat, fixed:1,2,4 + dynamic 0.95/0.90, full sets
bash launch_baselines.sh                                          # SDAR-1.7B-b64, card 0
MODEL=../models/SDAR-1.7B-Chat-local CARDS=1 bash launch_baselines.sh   # SDAR-1.7B b4
MODEL=../models/SDAR-8B-Chat-b64-local CARDS=0,1 bash launch_baselines.sh
bash launch_mi.sh                                                 # 8B proposes, δ 0.1/0.3/1/3
DRY_RUN=1 bash launch_mi.sh
```

Outputs go to `outputs/opencompass/<checkpoint>/` and `outputs/mi/<roles>/`,
each with `comparison/summary.csv`. All launchers check GPUs for other users'
processes and refuse to share a card.

**Protocol** (both launchers, since 2026-10-03): SDAR's own OpenCompass eval
config, as `../harness/AUDIT.md` details. That means GSM8K (1,319), MATH-500,
HumanEval (164) and sanitized MBPP (257) from `../data/opencompass/`, with
SDAR's prompts and graders (MATHEvaluator v2, math_verify, human-eval and MBPP
base tests). It also means a 4096-token canvas, fp16 and the whole vocabulary.
On top of that, ours: greedy T=0 (= its top_k=1), logp confidence,
max_candidates = block, and early stop at the first settled stop token, which
gives the same output. Every SDAR-family peer grades on base tests and runs fp16
or FP32 (`PEER_GROUP.md`). `TASKS=N` samples N per benchmark with seed 271828
for screening. The legacy benchmarks (`mbpp_plus`, `humaneval_plus`, the old
prompts, `TASKS=100` = exp04–06's samples) still run when named; their records
have no `protocol` field and are graded as before.

Answers that end inside the canvas do not depend on its length (blocks are
causal), which `../harness/scripts/extend_canvas.py` uses to re-decode only
truncated tasks after a canvas change.

Grids:
- static `fixed:k` uses k ≤ block;
- dynamic θ ∈ {0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3}, where SDAR itself uses 0.9 and 0.95;
- MI δ ∈ {0.03, 0.1, 0.3, 1, 3, 10}.

Together they cover about 1.3 to 7 tokens per step, so the curves overlap.

**Timing.**
- Batch 1, 3 untimed warmups per method, then one timed run per task. CUDA is
  synchronized on every card around each task.
- `wall_time_seconds` is end to end, prefill included.
- `prefill_seconds` is the part spent committing the prompt's whole blocks,
  measured with CUDA events inside the timed run. A prompt shorter than one
  block has none, because it is decoded with the first block.
- The summary reports `tokens_per_second` (output tokens / wall time,
  corpus-level) and `decode_tokens_per_second` (prefill excluded; SimSD's
  metric), plus forwards per task.
- Run metadata records the GPU model, each model's load time
  (`load_seconds`), and each method's warmup times (`warmup_seconds`).
- **Per-stage latency** (`launch_mi.sh`, `PROFILE_STAGES=10` by default). The
  first N tasks of each arm are decoded once more with a timer around each
  stage. The stages are prefill, `propose_forward` / `propose_rank`,
  `verify_forward` / `verify_test`, `decide`, `stop_check` and `commit`;
  `../harness/sdar_harness/timing.py` defines them. Each timer synchronizes
  every card, so this pass runs slower than the timed one. The headline
  latency is never taken from it. Records carry `stage_seconds` and
  `stage_counts`. `python analyze_stages.py --output-root <root>` prints each
  arm's shares, ms per call, calls per step, the untimed residual (the Python
  loop), and the profiling slowdown.
- Compare speedups only between runs on the same card set, and see
  [PEER_GROUP.md](PEER_GROUP.md) for how eager-mode tok/s relates to published
  numbers.

**Checkpoints available:** `../models/SDAR-1.7B-Chat-local` (b4),
`SDAR-1.7B-Chat-b64-local`, `SDAR-8B-Chat-b64-local`. The 8B b4 checkpoint
(`JetLM/SDAR-8B-Chat`, 16 GB) has not been downloaded.

## Deferred: decided 2026-10-09, revisit later

- **Dynamic θ ∈ {0.8, 0.7, 0.5} for SDAR-8B-b64**, all four benchmarks (about
  10 h on a card pair). MI at δ ≥ 0.3 commits more tokens per step than
  dynamic:0.90, so without these there is no dynamic arm at matching
  tokens/step. Add `PROFILE_STAGES` to `launch_baselines.sh` first, so the
  baselines get the same stage breakdown as the MI run.
- **Acceptance-test ablation.** The rule tests |cumulative log-ratio|. Most
  stops are positive excursions, and the code losses at δ ≥ 0.3 are syntax
  slips (a dropped `if`, an unclosed paren). Variants to compare on the
  20-task probe sample: a one-sided test (reject only a negative cumulative
  ratio) and a per-token or drawdown test.
- **Verifier rows per call: kept at 4.** Each row is the full 64-token block
  through the 1.7B, so a 4-row call is 256 tokens. In HF eager mode on an A4000,
  a 1.7B call costs about the same at 4, 64 or 256 tokens:
  - 39–41 ms for one 4-token row (B=4 fixed:1);
  - 38–46 ms for one 64-token row (b64 fixed:1);
  - 42–45 ms for the 4-row MI call.

  So the 4 rows ride free on the fixed per-call launch overhead. (Corrected
  2026-10-12: an earlier note here split the 43 ms as about 10 ms fixed plus
  8 ms per row.) Compute starts to show between 4 and 16 rows: exp06 measured
  about 8 ms/row at 16 rows. Early exit makes a bigger group score more rows
  that are thrown away. The probe already scores about 2 such rows
  per step: 4.2–10.9 rows scored per step for 1.8–8.8 tokens committed. The
  net gain would be about 10% of verify time at best. Group size 4 was chosen
  for exp05's cacheless Dream verifier (400–660-token rows) and never tuned
  for the SDAR 1.7B. It also changes fp16 numerics, so it must match across
  arms. What would cut verify time instead: shorter rows (b16 puts all 16 rows
  in one 256-token call), CUDA graphs (the fixed per-call cost), or a faster
  card. A 15-minute microbenchmark (rows 1–16 × prefix length, one card) would
  settle the per-row curve.

  **Measured 2026-10-15** (`bench_verifier_rows.py`, card 1, 1.7B b32 and b64,
  fp16; `outputs/verifier_rows_bench/card1_20261015.json`). The flat cost holds
  only at short context. Past ~500 prefix tokens the cost per call grows with
  rows × context, about 9 ms per row per 1000 prefix tokens. The reason is that
  SDAR's attention `torch.cat`s the cached prefix onto every row in every
  layer, and then runs SDPA with an explicit mask. Median ms per call:

  | prefix | 1×32 | 4×32 | 8×32 | 16×32 | 4×64 |
  |---|---|---|---|---|---|
  | 128 | 39 | 40 | 49 | 76 | 43 |
  | 512 | 39 | 42 | 68 | 128 | 54 |
  | 1024 | 48 | 61 | 104 | 201 | 76 |
  | 2048 | 41 | 100 | 180 | 352 | 122 |
  | 4096 | 65 | 182 | 337 | 665 | 216 |

  The 4×64 column reproduces the B=64 MI run's in-run `verify_forward`: 52 ms
  at ~0.5k, 118 ms at ~2.1k. So 256 tokens per call does not cost the same as
  8×32 and 4×64; rows are what cost. Projected onto the B=64 run's per-step
  commits and contexts at B=32, G=8 vs G=4 changes verify time by +20% to −22%.
  G=4 is better at δ ≤ 0.1 and on MATH; G=8 is better at δ ≥ 0.3. G=1, G=2 and
  G=16 are worse everywhere. A context-dependent G (8 below 384 prefix
  tokens, else 4) gains at most ~3% over the better fixed choice.

  **B=16, measured 2026-10-17 on EC2** (A10G card 0, 1.7B-b16 alone, same
  rounds and reps; `outputs/verifier_rows_bench/ec2_card0_b16_20261017.json`).
  The floor is ~36 ms. Past it, each row costs ~6.1 ms per 1000 prefix tokens,
  about 0.73× the A4000. Median ms per call (16-token rows):

  | prefix | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 16 |
  |---|---|---|---|---|---|---|---|---|
  | 128 | 36 | 36 | 37 | 37 | 36 | 36 | 36 | 39 |
  | 512 | 36 | 36 | 37 | 38 | 42 | 52 | 59 | 74 |
  | 1024 | 36 | 36 | 41 | 55 | 66 | 82 | 95 | 121 |
  | 2048 | 36 | 41 | 66 | 93 | 115 | 143 | 167 | 218 |
  | 4096 | 41 | 66 | 116 | 168 | 215 | 269 | 317 | 419 |

  The projection uses per-rank acceptance from the B=32 traces, every task's
  prompt, and b16 fixed:1 lengths. On these costs it gives **G=8 for B=16**:
  - G=8 is the best fixed G in total wall, and at most 3.4% behind the best G
    in any cell.
  - Totals for the other sizes: G=6 +1.3%, G=10 +1.2%, G=4 +6.2%, G=16 +4.3%.
  - A context-dependent G would gain about 2%.

## The η study (designed 2026-10-18)

[ETA_STUDY.md](ETA_STUDY.md) measures how well the 1.7B's in-block dependence
stands in for the 8B's: η_j = ½‖(p_j^P − q_j^P) − (p_j^V − q_j^V)‖₁ at the
greedy candidates, on held-out prompts (`../data/heldout/`). `measure_eta.py`
decodes at fixed:1 and probes one step per block; `launch_eta.sh` runs it;
`analyze_eta.py` reports κ candidates, false accepts and prefix replays.

## Peer group

[PEER_GROUP.md](PEER_GROUP.md) tracks the published methods to compare with or
cite: SimSD (the same SDAR-8B/1.7B pair, with roles swapped), S2D2, PUNT,
APD, Spiffy and others. It also holds the tokens/s sanity check against them.

## Ported from exp06 (`reference/exp06/`, gitignored)

Copied verbatim from exp06 outputs, same tasks and recipe:

| folder | source | arms |
|---|---|---|
| `sdar8b_b64/` | `exp06_vgb/outputs/sdar8b` | `fixed:1,2,4,8`, 4 benchmarks |
| `sdar1p7b_b64/` | `exp06_vgb/reference/sdar_1p7b_chat_b64` (+ `_math500`) | `fixed:1,2,4,8`; MATH timings not comparable (no early stop) |
| `mi_8b_proposes/` | `exp06_vgb/outputs/cross8b_swap` | `xadaptive` δ 0.1, 1 |
| `mi_1p7b_proposes/` | `exp06_vgb/outputs/cross8b` | `xadaptive` δ 0.1, 1 |

Only the `comparison/` summaries were regenerated, over the copied arms.
