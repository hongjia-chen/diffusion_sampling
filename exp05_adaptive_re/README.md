# exp05: adaptive delta, remasking, cross-model verification

SDAR-1.7B-Chat-b64 proposes, Dream-v0-Instruct-7B verifies and detects
corrections, on three of exp04's benchmarks (GSM8K, HumanEval+, MBPP+) and the same
100-task samples.
The design, and the measurements behind it, are in [DESIGN.md](DESIGN.md).

**Status:** implemented. B1 (SDAR fixed:k) is done and reproduces exp04. The
Dream-verified sweep is launched. The SDAR-verified adaptive baseline (B2) was
dropped. The rules and the reasoning
behind them are in [DESIGN.md](DESIGN.md).

| method | what it is |
| --- | --- |
| `fixed:k` | SDAR commits its k most confident tokens per step (B1) |
| `adaptive:delta` | SDAR proposes and SDAR verifies (B2); dropped, not part of exp05 |
| `xadaptive-remask:delta` | **the exp05 arm:** SDAR proposes, Dream verifies (Dream-only ratio, no veto), one Dream remasking pass per block (delta 0.1, 0.3, 1, 3, 10) |
| `xadaptive:delta` | the same without remasking; an ablation, not run by default (`--veto` adds the veto, also an ablation) |

Rules shared by every arm: early stop at the first stop token
(`<|im_end|>` or `<|endoftext|>`) once everything before it is committed;
logits read over token IDs < 151665, the range SDAR and Dream share;
canvases GSM8K 256, HumanEval+ 512, MBPP+ 256. MATH-500 is deferred.

Dream-7B on its own is **not** re-run. exp04's Dream `fixed:k` numbers are the
quality reference (see DESIGN.md "Baselines to re-run" for why their timings
aren't comparable).

## Layout

```
adaptive_re/          ported from exp04's autospec/ (package renamed)
  scheduler.py        fixed-k and adaptive commitment, early stop, stop set
  blockwise.py        block-diffusion loop (SDAR), Algorithm 3 branch removed
  cross.py            SDAR proposes / Dream verifies, remasking (veto optional)
  model.py            checkpoint-agnostic provider (dream / sdar families)
  scoring.py          gsm8k / math_500 / mbpp_plus / humaneval_plus
  io.py, timing.py
scripts/              run_benchmark, score_evalplus, summarize_results,
                      compare_exp04 (B1/B2 must reproduce exp04's text),
                      preflight, pack_work, check_run.sh, queue_status.sh
launch_sdar_baselines.sh   PHASE=b1 | b2, one process per GPU
launch_cross.sh            xadaptive arms; LAYOUT=exclusive (2 workers) | shared (3)
tests/                everything from exp04 except the Algorithm 3 and sigma tests
reference/            exp04's Dream-7B frontier and SDAR cost tables;
                      exp04_sdar_costs.json balances B1/B2 across GPUs
vendor_env -> ../exp04_autospeculation/vendor_env   (transformers 4.52.4 overlay)
```

Shared, at the repository root:

- `../models/`: patched SDAR checkpoints (moved from exp04; exp04 keeps a symlink).
- `../data/`: the frozen benchmark files (moved from exp04; exp04 keeps a symlink).
  These are the **full** frozen sets (GSM8K 500, MATH-500 500, HumanEval+ 164,
  MBPP+ 378). The 100-task samples are chosen at run time with
  `--sample-size 100 --seed 271828`, which reproduces exp04's tasks exactly.

## Checks

```bash
PYTHONPATH=vendor_env/tf452:. python -m pytest -q
```
