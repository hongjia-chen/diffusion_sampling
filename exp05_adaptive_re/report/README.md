# report/

`remask_report.html`: what the remasking pass does, built only from finished
runs. Rebuild with:

```bash
PYTHONPATH=vendor_env/tf452:. python report/make_remask_report.py
```

| file | what it does |
|---|---|
| `remask_diagnosis.py` | plans, verifies and reads the paired no-remask counterfactual; groups tasks by what remasking did; launches and checks the traced re-runs |
| `make_remask_report.py` | builds `remask_report.html` from `remask_report.template.html` |
| `examples.json` | the nine hand-picked examples, with the reason each was chosen |
| `data/no_remask/` | the counterfactual: subsets, launcher, logs and its `out/` records |
| `data/traced/` | traced re-runs of the examples, both arms |

The HTML also reports the same four task IDs per benchmark used in exp04's
decoding-path analysis (12 tasks across GSM8K, HumanEval+, and MBPP+). For each
task it shows all five exp05 delta settings, remasking on versus off, and the
corresponding exp05 completions.

## The counterfactual

The page's central claim is causal, so it needs the same task decoded twice:
with remasking (`outputs/cross`) and without it (`xadaptive`, same δ, no veto).

Decoding is deterministic — greedy SDAR, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, a
per-task seed — and the remasking pass draws randomness only when it edits a
token. So a task whose record says `tokens_changed_by_remask == 0` decodes
identically without remasking, and only the 1,106 task runs with an edit were
re-run. 26 unchanged tasks were re-run as controls and every one reproduced its
answer exactly, which is what makes the shortcut legitimate.

```bash
python report/remask_diagnosis.py plan       # subsets + data/no_remask/launch.sh
bash report/data/no_remask/launch.sh         # ~3 h, 3 workers (shared layout)
python report/remask_diagnosis.py verify     # completeness + the control check
# then score the code arms (evalplus) before building
```

The counterfactual ran in the shared layout, three SDAR copies on one card, so
**its timings are contended and are never reported**. It exists for accuracy and
answer text only; every latency number on the page comes from the uncontended
runs (`outputs/cross`, `outputs/baseline/sdar`).

## Examples

`examples.json` holds nine tasks, three per effect (remasking made the answer
right, made it wrong, moved where it ends), one per benchmark in each. Every one
was chosen to have **exactly one remask edit**, so the panel diff has a single
cause. Selection is by hand from `remask_diagnosis.py candidates`, and the
reason is recorded per example.

```bash
python report/remask_diagnosis.py trace         # traced re-runs, both arms
python report/remask_diagnosis.py check-traces  # each reproduces its recorded answer
```

The builder refuses to write the page if a traced re-run's completion differs
from the recorded run, or if the canvas it reconstructs from the traces does not
decode to that completion.

## Reading the page

- **Charts:** Dream-7B is exp04's run, unchanged; its scores are comparable and
  its tokens/s is not (different recipe, no early stop, MBPP+ at canvas 512).
- **Panels:** the two arms are identical until the first remask edit, which is
  outlined in violet. "Next remask" steps between remask passes.
- **Remask inspector:** every token Dream re-scored with that token masked, its
  probability then and at commit time, Dream's preferred token, which of the
  bottom 4 Dream would change, and the 1/p draw weights over those.
