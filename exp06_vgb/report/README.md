# report/

`report.html`: exp05's remasking page rebuilt for the exp06 runs (DESIGN.md
§10–10b), built only from finished runs. Rebuild with:

```bash
PYTHONPATH=../harness/vendor_env/tf452:. python report/make_report.py
```

| file | what it does |
|---|---|
| `make_report.py` | builds `report.html` from `report.template.html` |
| `trajectories.py` | the nine examples (and why each was picked), their traced re-runs, and `verify` |
| `data/traced/` | traced re-runs of the examples: subsets, `launch.sh`, logs, and outputs per arm |

## What the page shows

- **Accuracy against throughput**: SDAR-8B and SDAR-1.7B fixed:k, cross8b and
  swap with remasking on (filled) and off (hollow), and VGB Track A.
- **What remasking does to accuracy**: every arm with remasking off and on,
  paired on the same tasks, for exp05 (Dream verifies), cross8b and the swap.
- **What kind of edit it makes**: the first surviving edit of every edited run.
  exp05 keeps no edit log, so this is the one like-for-like view (the same rule
  as `scripts/analyze_cross8b.py`, with whitespace split into line endings and
  the rest).
- **Edits that changed the score**: every fix and break in cross8b and the
  swap, with the logged edits. "Edit alone" means the two answers otherwise
  align (difflib token similarity >= 0.9). DESIGN.md §10b compared tokens
  position by position, which counts an edit that shifts later tokens as a
  re-generation, so its direct counts are lower (11/12 against 16/7 for the
  cross8b fixes).
- **What kind of error it makes**: every failed task of the fixed (8B, 1.7B
  k=1) and adaptive (cross8b, swap; remasking off and on) arms, one error kind
  each, from `outputs/error_taxonomy.json`. Build that first with
  `python scripts/error_taxonomy.py build` (it re-runs the failing code's base
  tests). Code kinds come from rules, then hand labels for what the rules leave
  as a wrong plan (`data/error_labels_code.json`); GSM8K kinds are all hand
  labels (`data/error_labels_gsm8k.json`). The paired view shows what
  remasking does to each kind, and the score-flip table names each flip's kind
  with remasking off and on.
- **Decoding player**: SDAR-8B fixed k=1, SDAR-1.7B fixed k=1, and cross8b at
  δ=0.1 and δ=1, stepped in lockstep. A toggle switches the δ panels to their
  no-remask counterfactual. Below them are VGB Track A's three arms (forward
  only, λ=0.5, λ=1), which step one chain move at a time. Stays and switches
  change nothing, so they are counted onto the next move. A remasked token
  stays struck through until it is refilled, and the inspector gives each
  move's reason: the shortlist and its resampling weights for an unmask, and ρ
  and the runners-up for a remask.

## Examples

Three per benchmark (`trajectories.py` lists them with the reason for each):
every panel wrong; a baseline right and remasking wrong; remasking fixed it.
cross8b broke no MBPP+ task (11 fixes, 0 breaks), so the MBPP+ example of the
second kind has both remask arms wrong without the edit being the cause.

The benchmark runs traced only their first five tasks, and the baselines traced
none, so each example was re-run with tracing on and its arm's recorded flags:

```bash
python report/trajectories.py plan      # subsets + data/traced/launch.sh
bash report/data/traced/launch.sh       # ~20 min on cards 0+1
python report/trajectories.py plan vgb  # data/traced/launch_vgb.sh
bash report/data/traced/launch_vgb.sh   # ~2.7 h on cards 0+1
python report/trajectories.py verify    # completion, EOS, steps, remask edits, VGB remask lists
```

Track A ran with `--trace-limit 0`, and before 2026-09-30 the runner never
turned VGB tracing on even when asked. `run_benchmark.py` now does, and the
VGB trace records each move's reason (`log_rho` and the top remask
candidates; the shortlist with counts and weights). Tracing draws no random
numbers (`tests/test_vgb.py::test_tracing_does_not_change_the_walk`), so the
sampled arms re-run exactly: `verify` checks each one's answer, step count and
per-block remask list against Track A.

Seeds are per (seed, method, task), so a subset decodes as the full run did.
The 1.7B baseline re-runs with exp05's code, which produced it. The builder
refuses to write the page if a re-run stops matching its recorded run, or if
the canvas it reconstructs from the traces does not decode to the recorded
answer.
