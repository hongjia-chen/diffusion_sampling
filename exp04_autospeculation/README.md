# Autospeculative parallel sampling

This experiment adds a third sampler to the fixed-k / adaptive comparison of
`../exp03_dream_adaptive_sampling_ablation`: **Algorithm 3 of "Parallel Sampling via
Autospeculation"** (Anari, Baronio, Chen, Haqi, Koehler, Li, Vuong;
[arXiv:2511.07869](https://arxiv.org/abs/2511.07869)).

It differs from the previous experiment in three ways: a smaller model, four
benchmarks instead of three, and the new sampler.

**Status: Algorithm 3 is implemented and verified exact; no benchmark run yet.**
`autospec/autospeculation.py` holds the sampler, `tests/test_autospeculation.py`
checks it against an enumerated joint. What remains before a run is choosing
the checkpoint and settling D6/D7 in [DESIGN.md](DESIGN.md).

Verified against an exactly-consistent oracle (a coordinate denoiser backed by
an explicit joint, so no model error is in the way), 20k draws each:

| | TV vs enumerated joint | sampling noise floor |
| --- | ---: | ---: |
| n=4, sigma identity | 0.0088 | 0.032 |
| n=4, sigma reversed | 0.0057 | 0.032 |
| n=4, sigma scrambled | 0.0069 | 0.032 |
| n=6, sigma scrambled | 0.0149 | 0.061 |

And the round saving behaves as the theory says it should -- it is governed
entirely by how far the product of marginals sits from the joint. At n=8,
against 8 rounds for sequential ancestral sampling:

| coupling | rounds | speculation accept rate |
| ---: | ---: | ---: |
| independent | 2.00 | 1.00 |
| weak | 8.13 | 0.45 |
| strong | 13.11 | 0.33 |

With independent coordinates the root block is accepted on the first coin: two
rounds, whatever `n` is. With strong dependence it costs *more* than sequential.
Which regime a real checkpoint sits in, at a random sigma, is the experiment.

## Why this algorithm is worth implementing here

The previous experiment's `adaptive:delta` scheduler and Algorithm 3 use the
**same speculation**: the product of per-position marginals, `nu`, scored
against the joint chain-rule conditionals, `mu`. `_score_all_prefixes` in
`autospec/scheduler.py` already computes precisely the log-ratio that appears
in the paper's `dmu/dnu`.

What differs is the acceptance rule, and what a rejection costs:

| | acceptance | on rejection | guarantee |
| --- | --- | --- | --- |
| `adaptive:delta` | longest prefix with \|log dmu/dnu\| <= log(1+delta) | commit the prefix anyway | none; bias unbounded |
| Algorithm 3 | the whole block, by a coin at `min(1, dmu/dnu)` | recurse into two half-blocks | exact (Prop. 18) |

So `adaptive:delta` is a heuristic approximation of a sampler that now has an
exact form. The headline question of this experiment is what exactness costs,
measured in rounds rather than in wall-clock. See [DESIGN.md](DESIGN.md) D1 for
the ways in which "exact" is narrower than it sounds when the oracle is a
neural network.

## Benchmarks

| Benchmark | Tasks | Canvas | Primary metric | Extraction |
| --- | ---: | ---: | --- | --- |
| GSM8K | 500 | 256 | exact match | lm-eval `until`, then flexible-extract |
| MATH-500 | 500 | 384 | accuracy | last `\boxed{...}`, then normalized comparison |
| HumanEval+ | 164 | 256 | pass@1 | evalplus `sanitize`, then execution |
| MBPP+ | 378 | 256 | pass@1 | evalplus `sanitize`, then execution |

`--sample-size` draws a reproducible subset (100 by default in `run_all.sh`);
`select_sample` keeps the frozen dataset order rather than biasing toward the
start of the file the way `--limit` does.

Two changes from the previous experiment's benchmark set:

- **Added** HumanEval+ and MATH-500. HumanEval+ shares all of MBPP+'s
  machinery, so it costs one prepare script. MATH-500 is the one benchmark
  whose answers are expressions rather than numbers, which is why it needs
  `\boxed{}` extraction and a LaTeX normalizer rather than flexible-extract.
- **Dropped** Schema JSON. It was the benchmark where the previous experiment's
  reported quality/speed tradeoff turned out to be a scoring artifact, and its
  prompts embed a whole JSON Schema, making it the memory-tightest benchmark
  for no corresponding scientific return.

Canvas lengths are set so Algorithm 3's root block -- the whole canvas -- fits
in one physical verifier batch at ~2B (DESIGN.md D5), while still fitting the
answer: 28.6% of MATH-500 reference solutions exceed 256 tokens, which is why
it gets 384 rather than 256.

MATH-500's prompt names `\boxed{}` explicitly. That is not decoration: scoring
reads the last `\boxed{...}` span, and a correct answer written as prose scores
zero. GSM8K can afford a format-free prompt because flexible-extract takes the
last number; there is no equivalent fallback for an expression.

## Model

**Not chosen yet, and deliberately not defaulted.** `DEFAULT_MODEL` is `None`,
`--model` is required, and `run_all.sh` refuses to start without `MODEL_ID`.
The previous experiment hardcoded `Dream-org/Dream-v0-Instruct-7B`; a stale
default here is how a twenty-hour run of the wrong thing gets launched.

`autospec/model.py` is therefore checkpoint-agnostic, with the two conventions
that actually differ between masked diffusion LMs pulled out into
`Conventions`:

- `logits_alignment` — Dream's wrapper shifts model output one position right;
  most masked-diffusion heads are already position-aligned.
- `attention_mask` — Dream's remote code takes the string `"full"` to mean
  bidirectional; others want a real tensor, or nothing.

`FAMILIES["dream"]` is verified against Dream's `generation_utils.py`.
`FAMILIES["generic"]` is a starting point and **is not verified against any
specific checkpoint**. Check whichever model is chosen against its own
reference generation code before spending GPU hours: a wrong convention still
produces fluent text, and `verify_lean_path` will not catch it, because both
read paths apply the same convention and so agree with each other.

Why a smaller model matters beyond cost: Algorithm 3's inner loop wants many
independent samples running *simultaneously*. On the four 16 GB A4000s here, a
7B bf16 checkpoint leaves room for a verifier microbatch of about 4 — which is
not enough parallelism for the algorithm to be doing what it claims. See
DESIGN.md D3.

## Layout

```
autospec/
  scheduler.py        fixed-k and adaptive commitment, ported unchanged
  autospeculation.py  Algorithm 3 (generator-based lockstep batching)
  model.py            checkpoint-agnostic provider, no default model
  scoring.py          gsm8k / math_500 / mbpp_plus / humaneval_plus
  io.py               atomic writes, jsonl streaming, percentile
scripts/
  prepare_all.sh      freeze all four benchmarks
  run_benchmark.py    resumable per-task record generation
  score_evalplus.py   execution scoring for both code benchmarks
  summarize_results.py
  preflight.py        will the real run fit, and did the lean path engage
  pack_work.py        LPT packing over MEASURED costs (no default table)
tests/                Algorithm 3 exactness, scheduler, scoring, conventions
report/               empty until there are results
```

Everything except `autospeculation.py`, `scoring.py`, `model.py` and the two
new prepare scripts is a port of the previous experiment, with the package
renamed. The pieces that were carried over unchanged were carried over because
they are settled, not because they were convenient.

## Run

```bash
bash scripts/prepare_all.sh
```

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/preflight.py --model "$MODEL_ID" --dataset data/mbpp_plus/test.jsonl --continuation-length 512
```

```bash
MODEL_ID=... TASKS=100 GPU_IDS="0 1 2 3" nohup bash run_all.sh > logs/v1/launcher.out 2>&1 &
```

Integrity checks, none of which need a GPU:

```bash
python3 -m pytest -q && python3 -m py_compile autospec/*.py scripts/*.py && bash -n run_all.sh scripts/*.sh
```

## The Dream-7B reference frontier

`reference/dream7b_frontier.json`, built by `scripts/reference_frontier.py`,
restricted to the **Dream/ADJUST instruct recipe** (`alg="entropy"`, T=0.1,
top_p 0.9) -- the same recipe the new HumanEval+ and MATH-500 Dream runs use, so
all four benchmarks are one curve.

| GSM8K (canvas 256) | s/task | score | | MBPP+ (canvas 512) | s/task | score |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `fixed:8` | 4.9 | 0.150 | | `fixed:8` | 16.7 | 0.050 |
| `fixed:4` | 9.9 | 0.510 | | `fixed:4` | 34.0 | 0.240 |
| `fixed:2` | 20.0 | 0.770 | | `fixed:2` | 69.2 | 0.380 |
| `fixed:1` | 38.8 | **0.820** | | `fixed:1` | 133.8 | **0.560** |

**The harness agrees with the model's own paper.** Dream-v0-Instruct-7B
publishes GSM8K 81.0, MATH 39.2, HumanEval 55.5, MBPP 58.8 (arXiv:2508.15487).
Our `fixed:1` measures GSM8K **0.820** against 81.0 published, and MBPP+ 0.560
against MBPP 58.8 (MBPP+ is the harder plus-test variant, so slightly lower is
expected). That agreement is the evidence that the prompt, extraction and
scoring path are right -- and it gives sanity targets for the two new
benchmarks: expect HumanEval+ around 0.47-0.50 (HumanEval+ typically lands
5-8 points under HumanEval) and MATH-500 around 0.35-0.40.

**It is a reference curve, not a control.** Dream-7B has 4.5x the parameters of
SDAR-1.7B, so "beats Dream's latency" is close to guaranteed by parameter count
and "matches Dream's quality" close to impossible; neither is evidence about a
sampler. The controlled comparison -- fixed-k vs adaptive vs autospec -- lives
entirely within the SDAR arms.

Note the frontier is entirely `fixed:k`. That is the bar.

**Task pairing is locked, not hoped for.** `data/gsm8k` and `data/mbpp_plus` are
copied verbatim from the Dream run with their sha256 verified against that run's
recorded `dataset_sha256`, so `--sample-size 100 --seed 271828` selects the same
100 tasks the frontier was measured on. `PORTED_FROM.txt` records provenance in
each. HumanEval+ and MATH-500 are sampled the same way, 100 tasks at seed
271828, frozen once and reused by every arm and both models.

## Throughput

Three rates, all corpus-level -- `sum(tokens) / sum(time)`, never the mean of
per-task rates. vLLM and Fast-dLLM both aggregate this way, and it matters
here: averaging per-task ratios over-weights short tasks, which on GSM8K means
over-weighting exactly the collapsed early-EOS completions whose speed is the
least meaningful number in the run.

| metric | definition | use it for |
| --- | --- | --- |
| `tokens_per_second` | output tokens (to first EOS) / wall time | the industry-standard number, comparable to published TPS |
| `canvas_tokens_per_second` | canvas length / wall time | **comparing schedulers** |
| `tokens_per_forward` | canvas length / model forward calls | what the parallelism bought; 1 for an AR model by construction |

`tokens_per_second` matches vLLM's `total_output_tokens / elapsed_time` and
Fast-dLLM's `(generated_ids != eos).sum() / elapsed`. It is the right number to
quote against other papers, and the wrong one to rank schedulers by: a masked
diffusion model does the same work filling its canvas whether it emits 20
tokens or 250, and this project's `early_eos_rate` ranges from 0.00 to 0.68
across methods. An arm that commits EOS at position 3 has not run 12x slower --
it has produced 12x less text for the same work. `canvas_tokens_per_second`
divides the work actually done by the time it took, so it isolates the
scheduler.

`tokens_per_forward` is the diffusion-LM field's NFE metric inverted. It is
batching-dependent for the arms that batch a verifier, so read it alongside
`mean_verifier_rows`, which is not.

Wall time is `time.perf_counter()` around a `torch.cuda.synchronize()` on
either side, which is what makes the measurement honest on an async device.

## Carried-over conventions worth knowing

These were decided in the previous experiment and are not revisited here:

- **EOS is an ordinary committed token** during denoising. Every method fills
  the whole fixed canvas; only the returned and scored completion is truncated
  at the first EOS. An early EOS therefore does not reduce latency.
- **Only the mask token is excluded** from candidates, matching ADJUST's
  `mask_eos: false`.
- **Seeds are per `(seed, method, task)`**, not per arrival order, so resume,
  timing repeats and trace replay all reproduce the same stream.
- **Reproducibility needs `CUBLAS_WORKSPACE_CONFIG=:4096:8`**, which
  `run_all.sh` exports. `torch.use_deterministic_algorithms` is never called,
  so nothing in the code enforces it.
- **`mean_verifier_rows` and latency are the batching-invariant costs.**
  Forward-pass counts and peak memory move with `--verifier-microbatch-size`
  for identical work and live under `batching_dependent` in the summary.
