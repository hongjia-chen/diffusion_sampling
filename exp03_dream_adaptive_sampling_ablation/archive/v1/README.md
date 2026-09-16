# Archived: first-generation run (v1)

Everything here predates the v2 configuration and **must not be mixed with
`outputs/v2`**. Kept rather than deleted because `outputs/core` represents
roughly 28 GPU-hours and is the input `report/` was built from; neither is in
git, since the repository ignores `outputs/` and `logs/`.

## Why it is superseded

| | v1 (here) | v2 (`run_all.sh`) |
| --- | --- | --- |
| GSM8K prompt | custom, demanded a `#### <number>` line | lm-eval `gsm8k_cot`, `Q: {q}\nA:` |
| GSM8K scoring | last `####` match | flexible-extract (last number) + `until` |
| Candidate cap | 8 | 64 |
| Verifier | scored all candidates | early exit at first violation |
| Excluded tokens | mask, pad, bos | mask only |
| Fixed-arm sampling | greedy argmax, max-log-prob | entropy, T=0.1, top_p=0.9 |
| Seeding | none | per (seed, method, task) |

The `####` prompt is the reason these results are not merely stale but
misleading: the confidence-ordered proposer commits `####` at canvas position 0
and ends the response before any reasoning is written, which collapsed `fixed:1`
to bare answers and inverted the quality-versus-parallelism curve. The cap of 8
separately censored the adaptive sweep, binding on 20.5% of GSM8K steps and
68.9% of schema-JSON steps at delta 1.5.

## Contents

- `outputs/core/` — the 100-task, 12-method run behind the report.
- `report/` — figures, LaTeX, and aggregated CSV built from it.
- `logs/core/`, `logs/core-ext/` — worker logs for that run.
- `outputs/scratch/` — one-off probes: a `TASKS=1` end-to-end check and two
  attention-mask investigations.
- `run.sh`, `run_delta_ext.sh` — the two-phase launcher `run_all.sh` replaces.
- `*.orig` — editor leftovers from before the v2 changes.
