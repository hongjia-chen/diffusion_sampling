# Report: Dream fixed vs. adaptive scheduling

## Build

```bash
tectonic report.tex     # or: latexmk -pdf report.tex
```

## Contents

| File | What it is |
| --- | --- |
| `report.tex` / `report.pdf` | The combined report over both setups. 10 pages, 3 tables, 6 figures. |
| `analyze.py` | Reads both roots, re-scores Schema JSON with the shipped scorer, computes paired bootstrap CIs. |
| `figures.py` | Builds the six figures (PDF for LaTeX, PNG for viewing). |
| `make_tables.py` | Emits `table_main_matched.tex`, `table_main_v2.tex`, `table_sampling.tex`. |
| `derived/main_table.csv` | Per-method summary for both roots. |
| `derived/per_example_scores.csv` | Per-task paired scores, both roots. |
| `derived/sampling_effect.csv` | Fixed arm, v2 recipe vs matched recipe, paired. |
| `derived/diagnostics.json` | Machine-readable red flags. |

Regenerate everything:

```bash
python report/analyze.py && python report/figures.py && python report/make_tables.py && (cd report && tectonic report.tex)
```

## Structure

Three groups of measurements, not four: the adaptive arm was run once and never
regenerated, so its rows are bit-identical between the two roots and appear once.
`make_tables.py` asserts that rather than assuming it.

| Group | Arm and recipe | Root |
| --- | --- | --- |
| 1 | fixed $k$, greedy `logp` | `outputs/v3_matched` |
| 2 | fixed $k$, `entropy` $T{=}0.1$ top-$p{=}0.9$ | `outputs/v2` |
| 3 | adaptive $\delta$, greedy `logp` | both roots, identical |

Groups 1 and 3 are the scheduler comparison; group 2 is the fixed arm's
reference-recipe curve.

| Section | Covers |
| --- | --- |
| 1 | All three groups in one figure and one table |
| 2 | The fixed arm under two recipes: the JSON scoring bug, and the paired effect of the recipe change |
| 3 | Early EOS and accuracy — why section 1's ordering does not hold |
| 4 | Verifier cost |
| 5–6 | Run hygiene; what is settled and the next run |

Figure conventions: accuracy on $y$, latency on $x$, best is top-left. The fixed
series are joined because they are ordered and monotone in $k$; the adaptive points
are left unjoined, because accuracy shows no monotone trend in $\delta$ on any
benchmark and a line would imply one.

## Status of the report's recommendations

The report documents `outputs/v2` as the run left it. Three of its five recommendations
have since been applied:

| # | Recommendation | Status |
| --- | --- | --- |
| 1 | Fix the JSON scorer and re-score offline | **Done** — `extract_json_object` in `dream_ablation/scoring.py`, applied with `scripts/rescore_schema_json.py` |
| 2 | Re-issue `summary.csv` corrected, arms marked | **Done** — `outputs/v2/comparison/` regenerated; the arms are documented as separate sampling regimes |
| 3 | Run the fixed arm greedy so the arms are comparable | **Done** — `run_matched_fixed.sh` → `outputs/v3_matched`; this report covers it |
| 4 | EOS-suppression variant | **Now the critical one** — see below |
| 5 | State whether step count or latency is the headline metric | Open for the writeup |

`analyze.py` reads **both** roots and emits paired per-task data, so every comparison
in the report — across methods within a run, and across runs within a method — is a
paired bootstrap. For schema_json it reads the original verdict from `records.jsonl`
and recomputes the corrected one with the shipped scorer, which keeps the
scoring-bug comparison reproducible now that the fix has landed.

## What the matched run left open

Matching the recipe removed the sampling confound and introduced a sharper one.
Greedy decoding makes the fixed arm commit EOS at or near canvas position 0 — 68% of
GSM8K completions carry no reasoning at `k=1`, and 19 of 100 MBPP+ completions are
empty. The unconditional ordering (adaptive > fixed on GSM8K and MBPP+) reverses once
that is conditioned away, so it cannot be quoted as a scheduler result.

Quotable today: the Schema JSON comparison (no early EOS at any setting; fixed strictly
dominates) and the verifier-cost analysis (recipe-independent). Not quotable: the GSM8K
or MBPP+ fixed-vs-adaptive ordering, in either direction.

The unblocking experiment is EOS suppression until a minimum committed-token count,
on GSM8K and MBPP+ only — roughly 7 GPU-hours, ~2.5h on three cards.
