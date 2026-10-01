# harness: shared SDAR block-diffusion decoding harness

This is the code that every SDAR experiment from exp05 on runs through. It
covers model loading, the block-decoding loop, the samplers, scoring, the
benchmark runner and the tests. Like `../models/` and `../data/`, it lives at
the repository root, and experiments call into it rather than carry copies.

It was lifted unchanged from `exp06_vgb/adaptive_re/` on 2026-09-30. That copy
was a strict superset of exp05's: it adds multi-GPU sharding, block verifiers
and VGB, and the Dream path is untouched. The only changes in the move are the
package rename (`adaptive_re` → `sdar_harness`), a `sys.path` line so the
scripts run from any directory, and a path fix in one test. The code exactly as
it produced exp05 and exp06's results is in git commit `b98dad7`.

**Before trusting a number from it, read [AUDIT.md](AUDIT.md).** The decoder
matches SDAR's reference. The benchmark protocol does not match the one behind
SDAR's published table.

## Layout

```
sdar_harness/
  model.py        checkpoint provider (sdar / dream families), pipeline sharding over cards
  blockwise.py    SDAR block loop: prefill, per-block denoise, store_kv commit, early stop
  scheduler.py    fixed-k (SDAR static), dynamic θ (SDAR low_confidence_dynamic) and
                  adaptive (same-model MI ratio) commitment, stop set, early stop
  cross.py        cross-model MI decoding: one model proposes, the other verifies (+ remask)
  vgb.py          density-ratio VGB walk (exp06 Track A/B)
  scoring.py      gsm8k / math_500 answer extraction; code sanitizing for evalplus
  io.py, timing.py
scripts/
  run_benchmark.py      the runner: --methods fixed:k | dynamic:θ | adaptive:δ | xadaptive[-remask]:δ | vgb:…
  score_evalplus.py     executes MBPP+ / HumanEval+ records with EvalPlus
  summarize_results.py  per-arm summary.csv / summary.json
  pack_work.py, preflight.py, prepare_sdar_local.py (builds ../models/*-local)
tests/            147 tests, CPU only except one 2-card shard test (skipped without 2 GPUs)
vendor_env/tf452  transformers 4.52.4 overlay SDAR needs (gitignored; exp04/05 symlink here)
```

## Block length

An SDAR checkpoint decodes at the block length it was trained with. b4 is
`JetLM/SDAR-*-Chat`; b8 to b64 are the `-bN` variants. Their config.json files
are identical, so `ModelOptions` reads the block from the checkpoint name
(`trained_block_length`) and refuses any other `--block-length`. The family
default is 64, so a b4 checkpoint needs `--block-length 4`.
`--allow-block-mismatch` is for out-of-distribution ablations only. Changes
after the move (2026-10-01): the `dynamic:θ` arm and this guard.

## Using it from an experiment

Launchers set `HARNESS_DIR="$PROJECT_DIR/../harness"`. They put it on
`PYTHONPATH` and call `"$HARNESS_DIR/scripts/run_benchmark.py"` etc.; see
`../exp06_vgb/launch_cross8b.sh`. Run models with the overlay first on the path:

```bash
PYTHONPATH=vendor_env/tf452:. python scripts/run_benchmark.py --help
```

Experiment-specific analysis scripts stay in the experiment and add
`../harness` to `sys.path`.

## Checks

```bash
cd harness && PYTHONPATH=vendor_env/tf452:. python -m pytest -q   # 147 passed, ~5 min
```
