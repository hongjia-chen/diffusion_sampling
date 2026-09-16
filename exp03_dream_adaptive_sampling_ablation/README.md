# Dream scheduler comparison

This repository runs one focused experiment: compare fixed and adaptive
parallel decoding with `Dream-org/Dream-v0-Instruct-7B`.

Two output roots:

| Root | Fixed arm sampling | Adaptive arm sampling | Read it for |
| --- | --- | --- | --- |
| `outputs/v2` | `entropy`, T=0.1, top_p=0.9 | greedy, `logp` | each arm's own quality/speed curve |
| `outputs/v3_matched` | greedy, `logp` | greedy, `logp` (imported) | **fixed vs adaptive** |

`outputs/v2` mixes a scheduler effect with a decoding-recipe effect and cannot
settle whether adaptive scheduling beats fixed-k. `outputs/v3_matched` puts both
arms on the same recipe.

**Read `outputs/v3_matched` with one caveat.** Matching the recipe moved the
fixed arm onto greedy decoding, where it commits EOS at or near canvas position
0: at `fixed:1`, 68% of GSM8K completions carry no reasoning and 19 of 100 MBPP+
completions are literally empty. The unconditional result says adaptive beats
fixed on GSM8K and MBPP+; restricted to completions that contain work, `fixed:1`
is best on all three benchmarks. Only Schema JSON, which has no early EOS at any
setting, can be read off directly — and there fixed strictly dominates. See
`report/report.tex` §3. Suppressing EOS until a minimum token count is the
outstanding experiment.

`outputs/diagnostics/` holds three small side-runs backing the timing and
reproducibility claims in that report; they are not experiment results.

## What is tested

Every method receives the same deterministic, seed-fixed sample of 100 frozen
examples from each benchmark:

| Benchmark | Tasks | Maximum completion | Verifier microbatch |
| --- | ---: | ---: | ---: |
| MBPP+ | 100 | 512 tokens | 4 |
| GSM8K | 100 | 256 tokens | 8 |
| Schema JSON | 100 | 256 tokens | 4 |

GSM8K uses lm-eval's `gsm8k_cot` prompt (`Q: {question}\nA:`) zero-shot, scored
with flexible-extract (last number) after the `until` list `["Q:", "</s>",
"<|im_end|>"]`, matching ADJUST's `configs/lm_evals/gsm8k_instruct.yaml`. The
canvas length of 256 also matches.

Microbatch is bounded by measured GPU memory, and what bounds it is
prompt+canvas rather than canvas alone: measured p95 prompt lengths are GSM8K
126, Schema JSON 381 (it embeds the whole JSON Schema in every prompt) and
MBPP+ 117, for sequence lengths of ~413 / ~639 / ~704. Only GSM8K has room for
8; Schema JSON OOMs there despite sharing GSM8K's 256-token canvas.

The twelve scheduler settings are:

| Method | Purpose |
| --- | --- |
| `fixed:1` | sequential quality reference |
| `fixed:2` | conservative fixed parallelism |
| `fixed:4` | moderate fixed parallelism |
| `fixed:8` | maximum-parallelism speed reference |
| `adaptive:0.10` | tightest adaptive setting |
| `adaptive:0.15` | cautious adaptive scheduler |
| `adaptive:0.20` | lower-middle adaptive setting |
| `adaptive:0.30` | lower-middle adaptive setting |
| `adaptive:0.50` | upper-middle adaptive setting |
| `adaptive:0.75` | aggressive adaptive scheduler |
| `adaptive:1.00` | very aggressive adaptive setting |
| `adaptive:1.50` | loosest adaptive setting |

The adaptive value is the relative tolerance `delta`; the scheduler uses
`tau_nats = log(1 + delta)`. It always accepts the first candidate and accepts
longer consecutive prefixes while the joint-vs-marginal factorization error
stays within that threshold. Both proposer and verifier are the same Dream
checkpoint.

EOS is treated as an ordinary committed token during denoising for both fixed
and adaptive scheduling. Every method completes the full fixed-size canvas;
only then is the returned and scored completion truncated at the first EOS.
Consequently, an early EOS does not reduce denoising steps or latency. Canvas
length is benchmark-specific but fixed across every example and scheduler
setting within a benchmark, using the limits in the table above.

This matches the reference implementations. Dream's `generation_utils.py` has
no early exit and no EOS special-casing, and its lm-eval wrapper truncates with
`decode(...).split(eos_token)[0]`. ADJUST runs every downstream eval with
`sampling.mask_eos: false` and asserts on it in `evaluate_lm`. Only the mask
token is excluded from candidates, as in ADJUST's `mask_eos: false` branch.

## Deliberate divergences from ADJUST

Three, all intentional, none of them silent:

1. **Sampling differs between arms in `outputs/v2`.** There, the fixed arm
   follows the Dream/ADJUST instruct recipe (`alg="entropy"`, temperature,
   top_p) while the adaptive arm stays on greedy argmax with max-log-prob
   confidence, which is why `run_all.sh` exposes them as separate flags.
   **The two arms of `outputs/v2` are therefore not comparable to each other**,
   and every fixed-vs-adaptive difference in that run mixes a scheduler effect
   with a decoding-recipe effect.

   `run_matched_fixed.sh` resolves this by regenerating the fixed arm under the
   adaptive recipe into `outputs/v3_matched`. Read fixed-vs-adaptive there, and
   read `outputs/v2` only for the fixed arm's own quality/speed curve under the
   reference recipe. See [Matched-sampling run](#matched-sampling-run).
2. **Adaptive commits a confidence-ranked prefix**, not a contiguous one.
   Row *j* of the verifier holds the *j* most confident positions anywhere in
   the canvas. ADJUST instead commits a contiguous left-to-right span from the
   first masked position. Both are valid chain-rule factorizations, but they
   measure different questions: "can I commit the *k* most confident positions
   jointly" versus "can I commit the next *k* positions jointly".
3. **The verifier scores under the unfiltered distribution.** `_propose` draws
   from the temperature-scaled, nucleus-filtered distribution, while
   `_score_all_prefixes` scores those tokens under the plain log-softmax. The
   delta criterion is a statement about the model's factorization error, not
   about the sampler's truncated distribution; renormalising over the nucleus
   would make `tau_nats` mean something different at every position.

## Reading the cost metrics

`mean_verifier_rows` and the task latencies are batching-invariant and are the
numbers to compare schedulers on. `mean_verifier_forward_passes` and
`peak_gpu_memory_bytes` move with `--verifier-microbatch-size` for identical
work, so they live under `batching_dependent` in the summary and are
diagnostics only.

`at_cap_rate` is the share of steps that committed `max_candidates` tokens. A
non-trivial value means the cap, not the delta criterion, decided those steps
and the tokens-per-step axis is censored there. It is what drove the cap up: at
a cap of 8 it reached 20.5% on GSM8K and 68.9% on schema JSON at delta 1.5, and
at 32 it still bound 14% of steps at delta 1.5. Under the cap of 64 used by
`outputs/v2` the observed maximum is 10.5%, so the sweep is no longer censored.

`early_eos_rate` is the share of completions whose first EOS lands before token
16 — short enough that no worked solution can be present. It is positional, not
score-derived, and it is k-dependent, which is why it is reported.

Under the matched (greedy) recipe it stops being a diagnostic and becomes the
dominant effect: on GSM8K it runs 0.68 / 0.31 / 0.05 / 0.01 across `fixed:1/2/4/8`
and correlates with accuracy at r = -0.56 across all twelve methods. Any
quality comparison drawn from `outputs/v3_matched` on GSM8K or MBPP+ has to
account for it. Schema JSON is 0.00 throughout.

The run records accuracy, task latency, denoising steps, tokens committed per
step, proposer passes, and verifier work. It intentionally uses one timed
generation and no decision-trace replay. Across 100 paired tasks, this is enough
for an exploratory scheduler comparison without paying for repeated inference.

Generation is bit-reproducible **only when `CUBLAS_WORKSPACE_CONFIG=:4096:8` is
set**, which both launchers export. `torch.use_deterministic_algorithms` is never
called, so nothing in the code enforces it: a method re-run by hand without that
variable produces different completions and step counts. Set it, or go through
the launchers. Evidence: `outputs/diagnostics/`.

## Scoring

Each benchmark extracts an answer before grading it, because a diffusion model
fills a fixed canvas and cannot be prompted to stop cleanly:

| Benchmark | Extraction |
| --- | --- |
| GSM8K | lm-eval `until` truncation, then flexible-extract (last number) |
| MBPP+ | evalplus `sanitize`, which unwraps ` ```python ` fences |
| Schema JSON | `extract_json_object`: raw, then fence-stripped, then outermost `{...}` |

`extract_json_object` tries the least aggressive repair first and records which
one worked in `json_repair` (`null` when the raw completion already parsed), so
the repair is auditable rather than silent. A method whose scores lean heavily
on `brace_span` is a method that is not really emitting clean JSON.

Schema JSON originally had no extraction step at all — it called `json.loads`
on the raw completion. The instruct checkpoint fences its JSON at some sampling
settings and not others, so this measured fence rate rather than JSON quality.
In `outputs/v2` it scored all four fixed methods **0.00** on content that was
byte-identical to adaptive completions scoring 1.00, and it depressed the
adaptive arm by an amount that grew with delta, manufacturing a
quality/speed tradeoff that does not exist:

| Schema JSON exact match | k=1 | k=2 | k=4 | k=8 | δ=0.10 | δ=0.50 | δ=1.00 | δ=1.50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| as originally scored | 0.00 | 0.00 | 0.00 | 0.00 | 0.91 | 0.71 | 0.62 | 0.45 |
| corrected | 1.00 | 0.94 | 0.13 | 0.00 | 1.00 | 0.91 | 0.94 | 0.95 |

`k=4` and `k=8` stay near zero after the fix; those are real generation
failures (a dropped opening brace, then token-repetition collapse), not
formatting. The corrected adaptive curve is flat, so **the Schema JSON
quality/speed tradeoff reported from the original `summary.csv` should be
withdrawn.**

Re-scoring needs no GPU — the raw completions are on disk:

```bash
python scripts/rescore_schema_json.py outputs/v2          # --dry-run to preview
python scripts/summarize_results.py --output-root outputs/v2 --output-dir outputs/v2/comparison
```

This writes `records.scored.jsonl`, which `summarize_results.py` prefers over
`records.jsonl`. The original records are never modified.

## Run

On the three-GPU host:

```bash
cd /home/hc33998/projects/diffusion_sampling/exp03_dream_adaptive_sampling_ablation
mkdir -p logs/v2 && nohup bash run_all.sh > logs/v2/launcher.out 2>&1 &
```

The defaults are `TASKS=100`, `GPU_IDS="0 1 2"`, and output under
`outputs/v2`. The run resumes completed records after interruption, and
refuses to resume into records written under a different candidate cap.

For a one-task end-to-end check in a separate output directory:

```bash
TASKS=1 OUTPUT_ROOT="$PWD/outputs/check" LOG_ROOT="$PWD/logs/check" bash run_all.sh
```

Monitor without touching CUDA or NVML:

```bash
bash scripts/check_run.sh v2
```

The final compact result is `outputs/v2/comparison/summary.json`. MBPP+
execution scoring runs after GPU generation; GSM8K and schema JSON are scored
during generation.

## Matched-sampling run

`run_all.sh` gives the two arms different decoding recipes (divergence 1 above),
so `outputs/v2` cannot answer "does adaptive scheduling beat fixed-k".
`run_matched_fixed.sh` regenerates only the fixed arm under the adaptive arm's
recipe — greedy argmax, `confidence=logp`, no top_p — holding seed, task set,
candidate cap, canvas length, microbatch and timing repeats identical:

```bash
mkdir -p logs/v3_matched
nohup bash run_matched_fixed.sh > logs/v3_matched/launcher.out 2>&1 &
```

It picks idle GPUs itself (anything using less than `FREE_MEMORY_MIB`, default
1024 MiB), since this host is shared; set `GPU_IDS="0 1"` to pin. Work is packed
across whatever it finds by longest-processing-time over measured per-method
costs — `scripts/pack_work.py --workers 3 --plan` prints the plan and makespan
without running anything. On three GPUs at 100 tasks that is ~4.1h, an upper
bound, since greedy skips the nucleus sort the reference recipe pays for.

The adaptive arm is **not** regenerated. It was already greedy, so re-running it
would spend ~17 GPU-hours reproducing identical records; instead
`scripts/assemble_matched_root.py` copies those method directories from
`outputs/v2`, and refuses to unless the source metadata agrees with the freshly
generated fixed arm on sampling, seed, candidate cap, canvas length, microbatch,
model, dtype, prompt template and EOS policy. The result, `outputs/v3_matched`,
is a self-contained experiment root, and `IMPORTED_FROM.txt` records which half
came from where.

For a cheap end-to-end check of the whole pipeline:

```bash
KS=8 TASKS=2 GPU_IDS=0 OUTPUT_ROOT="$PWD/outputs/check_matched" \
    LOG_ROOT="$PWD/logs/check_matched" bash run_matched_fixed.sh
```

Read fixed-vs-adaptive from `outputs/v3_matched`. Read `outputs/v2` for the
fixed arm's own curve under the Dream/ADJUST reference recipe, and for the
difference between the two recipes.

## Integrity checks

```bash
/home/hc33998/miniconda3/envs/diffusion/bin/python -m pytest -q
/home/hc33998/miniconda3/envs/diffusion/bin/python -m py_compile dream_ablation/*.py scripts/*.py
bash -n run_all.sh run_matched_fixed.sh scripts/*.sh
```

The core implementation is intentionally small:

- `run_all.sh` fixes the experiment matrix and launches the three workers,
  running all twelve methods in one balanced pass (~20.6h at 100 tasks).
- `run_matched_fixed.sh` regenerates the fixed arm under the adaptive arm's
  sampling so the two are comparable (~4.1h at 100 tasks).
- `dream_ablation/scheduler.py` implements fixed and adaptive commitment.
- `dream_ablation/model.py` loads Dream and projects only needed logits.
- `dream_ablation/scoring.py` extracts and grades answers; `extract_json_object`
  is the Schema JSON extraction step.
- `scripts/run_benchmark.py` generates resumable task records.
- `scripts/pack_work.py` packs methods across however many GPUs are idle.
- `scripts/assemble_matched_root.py` imports the adaptive arm into a matched
  root, refusing when the settings disagree.
- `scripts/rescore_schema_json.py` re-grades existing records without a GPU.
- `scripts/score_mbpp_plus.py` and `scripts/summarize_results.py` score and summarize.
- `report/` builds the LaTeX benchmark report from a finished root.
- `tests/` checks scheduling, scoring, JSON extraction, EOS behavior,
  tracing-off resume, and the memory-lean logits path.

## Archived work

`archive/v1/` holds the previous run, its report, and the two-phase launcher it
used. That configuration differs from this one in the GSM8K prompt and scoring,
the candidate cap, the verifier's early exit, the excluded-token set, and the
fixed-arm sampling, so its records must not be mixed with `outputs/v2`. See
`archive/v1/README.md` for the full comparison.
