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

The sampler, runner, scoring scripts and tests now live in the shared
[`../harness/`](../harness/README.md) (package `sdar_harness`, formerly this
experiment's `adaptive_re/`; exp06's copy, a strict superset of this one, was
kept). The code exactly as it produced exp05's results is in git commit
`b98dad7`. DESIGN.md still names files by their old `adaptive_re/` paths.

```
scripts/              compare_exp04 (B1/B2 must reproduce exp04's text),
                      check_run.sh, queue_status.sh
launch_sdar_baselines.sh   PHASE=b1 | b2, one process per GPU
launch_cross.sh            xadaptive arms; LAYOUT=exclusive (2 workers) | shared (3)
reference/            exp04's Dream-7B frontier and SDAR cost tables;
                      exp04_sdar_costs.json balances B1/B2 across GPUs
vendor_env -> ../harness/vendor_env   (transformers 4.52.4 overlay)
```

Shared, at the repository root:

- `../models/`: patched SDAR checkpoints (moved from exp04; exp04 keeps a symlink).
- `../data/`: the frozen benchmark files (moved from exp04; exp04 keeps a symlink).
  These are the **full** frozen sets (GSM8K 500, MATH-500 500, HumanEval+ 164,
  MBPP+ 378). The 100-task samples are chosen at run time with
  `--sample-size 100 --seed 271828`, which reproduces exp04's tasks exactly.

## Checks

The tests moved with the code:

```bash
cd ../harness && PYTHONPATH=vendor_env/tf452:. python -m pytest -q
```
