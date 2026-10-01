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
bash launch_baselines.sh                                          # SDAR-1.7B-b64, card 0
MODEL=../models/SDAR-1.7B-Chat-local CARDS=1 bash launch_baselines.sh   # SDAR-1.7B b4
MODEL=../models/SDAR-8B-Chat-b64-local CARDS=0,1 bash launch_baselines.sh
bash launch_mi.sh                                                 # 8B proposes, δ 0.1/0.3/1/3
DRY_RUN=1 bash launch_mi.sh
```

Outputs go to `outputs/baselines/<checkpoint>/` and `outputs/mi/<roles>/`,
each with `comparison/summary.csv`. All launchers check GPUs for other users'
processes and refuse to share a card.

**Recipe** (both launchers): greedy T=0, logp confidence, max_candidates =
block, early stop at the first settled stop token, logits over IDs < 151665,
bf16. Benchmarks are GSM8K (the full 1,319-problem set, `../data/gsm8k_full`),
MATH-500, MBPP+ (378) and HumanEval+ (164), on full sets by default. `TASKS=N`
instead samples N per benchmark with seed 271828, for screening.

Canvases are GSM8K 512, MATH-500 2048, MBPP+ 512 and HumanEval+ 768. exp05/06
used 256/1024/256/512, and 13–16% of MATH answers hit the limit. Blocks are
causal, so answers that ended earlier are unchanged and the exp06 references
still pair task for task.

This is the in-house protocol, not SDAR's OpenCompass one. Paired comparisons
are valid; absolute scores are not comparable to the paper (AUDIT §2).

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
- Run metadata records the GPU model.
- Compare speedups only between runs on the same card set, and see
  [PEER_GROUP.md](PEER_GROUP.md) for how eager-mode tok/s relates to published
  numbers.

**Checkpoints available:** `../models/SDAR-1.7B-Chat-local` (b4),
`SDAR-1.7B-Chat-b64-local`, `SDAR-8B-Chat-b64-local`. The 8B b4 checkpoint
(`JetLM/SDAR-8B-Chat`, 16 GB) has not been downloaded.

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
