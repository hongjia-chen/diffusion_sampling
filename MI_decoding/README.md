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

**Recipe** (both launchers, the exp05/06 recipe): greedy T=0, logp confidence,
max_candidates = block, 100 tasks per benchmark sampled with seed 271828,
canvases GSM8K 256 / HumanEval+ 512 / MBPP+ 256 (MATH-500 1024, opt-in through
`BENCHMARKS`), early stop at the first settled stop token, logits over IDs
< 151665, bf16. This is the in-house protocol. It is not SDAR's OpenCompass
protocol, so paired comparisons are valid and absolute scores are not
comparable to the paper (AUDIT §2).

**Checkpoints available:** `../models/SDAR-1.7B-Chat-local` (b4),
`SDAR-1.7B-Chat-b64-local`, `SDAR-8B-Chat-b64-local`. The 8B b4 checkpoint
(`JetLM/SDAR-8B-Chat`, 16 GB) has not been downloaded.

## Ported from exp06 (`reference/exp06/`, gitignored)

Copied verbatim from exp06 outputs, same tasks and recipe:

| folder | source | arms |
|---|---|---|
| `sdar8b_b64/` | `exp06_vgb/outputs/sdar8b` | `fixed:1,2,4,8`, 4 benchmarks |
| `sdar1p7b_b64/` | `exp06_vgb/reference/sdar_1p7b_chat_b64` (+ `_math500`) | `fixed:1,2,4,8`; MATH timings not comparable (no early stop) |
| `mi_8b_proposes/` | `exp06_vgb/outputs/cross8b_swap` | `xadaptive` δ 0.1, 1 |
| `mi_1p7b_proposes/` | `exp06_vgb/outputs/cross8b` | `xadaptive` δ 0.1, 1 |

Only the `comparison/` summaries were regenerated, over the copied arms.
