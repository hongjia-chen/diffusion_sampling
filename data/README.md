# Benchmark data

| folder | what it is | GSM8K | MATH-500 | HumanEval | sanitized MBPP | MBPP+ | HumanEval+ |
|---|---|---|---|---|---|---|---|
| **`opencompass/`** | **SDAR's protocol (the default since 2026-10-03)**: its four benchmarks with its prompts | 1,319 | 500 | 164 | 257 | — | — |
| `full/` | legacy exp04–06 prompts, complete sets | 1,319 | 500 | — | — | 378 | 164 |
| `sample100/` | the legacy 100-task samples exp04–06 ran on | 100 | 100 | — | — | 100 | 100 |
| `sample250/` | development subsets of `opencompass/` (2026-10-18, `TASKS=250`) | 250 | 250 | — | — | — | — |
| `heldout/` | calibration prompts from **non-test** splits (2026-10-18, η study) | 150 train | 150 train | — | 320 non-test | — | — |

Records are in the harness format (`example_id`, `prompt`, `target` / `entry_point`).
EvalPlus's own test cases, which score MBPP+ and HumanEval+, are cached
separately in `~/.cache/evalplus` (HumanEvalPlus-v0.1.10, MbppPlus-v0.2.0).

## opencompass/

Built 2026-10-03 with `harness/scripts/prepare_opencompass.py --output-dir data/opencompass`.
Every record has `"protocol": "opencompass"`, and its `prompt` is the HUMAN turn of
SDAR's OpenCompass config (`harness/sdar_harness/opencompass.py::PROMPTS`). Example
IDs match the legacy files, so tasks pair across protocols.

| file | source | gold / grading |
|---|---|---|
| `gsm8k.jsonl` | `openai/gsm8k` main test, 1,319 | `gsm8k_dataset_postprocess(answer)`; MATHEvaluator v2 on the last `\boxed{}` |
| `math_500.jsonl` | `HuggingFaceH4/MATH-500`, 500 (= OpenCompass `test_prm800k_500`) | `extract_boxed_answer(solution)` (equal to `answer` on all 500); math_verify |
| `humaneval.jsonl` | `openai/openai_humaneval`, 164, original prompts and tests | first fenced block + base tests |
| `mbpp_sanitized.jsonl` | `google-research-datasets/mbpp` "sanitized" test, 257 (ids 11–479; OpenCompass rows [7:264]) | its 3 asserts. 224 of the 257 are also in MBPP+ |

## full/

| file | source | `<name>.metadata.json` |
|---|---|---|
| `gsm8k.jsonl` | `openai/gsm8k` "main" test split, all 1,319 problems in source order; built 2026-10-01 with `harness/scripts/prepare_gsm8k.py --test-size 0`. Prompt is lm-eval's `gsm8k_cot` 0-shot `Q: …\nA:` | sha256 `d5e6ec59…`, source indices |
| `math_500.jsonl` | `HuggingFaceH4/MATH-500`, all 500 | sha256 `2143694f…` |
| `mbpp_plus.jsonl` | EvalPlus MBPP+ v0.2.0, all 378 (ported from exp03) | sha256 `5ea7b4c2…`, evalplus hash |
| `humaneval_plus.jsonl` | EvalPlus HumanEval+ v0.1.10, all 164 | sha256 `2857f0f7…`, evalplus hash |

The MATH-500, MBPP+ and HumanEval+ files are byte-identical to the
`data/<bench>/test.jsonl` files that exp04–06 runs record in their metadata.

## sample100/

Each file is the exact task list, in order, that `--sample-size 100 --seed 271828`
picked in exp04–06. On 2026-10-01 each one was checked against
`exp06_vgb/outputs/sdar8b` task for task, and every record is identical to its
`full/` record.

- **GSM8K's sample was drawn from a frozen 500-task subset (seed 12345, from
  exp03), not from the full 1,319.** Re-sampling 100 from `full/gsm8k.jsonl`
  gives different tasks. To stay paired with old runs, use `sample100/`.
- The other three were drawn from the full sets, the same files as in `full/`.

## sample250/

Development subsets of `opencompass/`, made 2026-10-18 for B=16 algorithm
work: with all of HumanEval (164) and MBPP (257), each benchmark has about 250
tasks. `TASKS=250` in the MI_decoding launchers runs these two plus the full
HumanEval and MBPP. Records are byte-identical to their `opencompass/` lines;
`<name>.metadata.json` lists the IDs, the selection and hashes.

- **GSM8K:** `select_sample(rows, 250, seed=271828)`, the same tasks
  `--sample-size 250 --seed 271828` picks from `opencompass/gsm8k.jsonl`.
- **MATH-500:** stratified by level, half of each (22/45/52/64/67), drawn with
  `random.Random(271828)`. A plain random 250 drew 14 of the 43 Level-1
  problems and scored 2 points below the full set at SDAR-8B-b16 fixed:1; the
  stratified one scores 76.0 against 75.6.

Every full-set run already holds these tasks, so a subset baseline is a filter
by `example_id`, not a new run.

## heldout/

Calibration prompts that no evaluation set uses, for thresholds fitted on
model behaviour (the η study, `MI_decoding/ETA_STUDY.md`). Built 2026-10-18 with
`harness/scripts/prepare_heldout.py --output-dir data/heldout`, seed 271828,
OpenCompass templates: 150 GSM8K and MATH, and MBPP with `--only mbpp_heldout --size 320`; `<name>.metadata.json` lists IDs, selection and hash.

- `gsm8k_train.jsonl`: `openai/gsm8k` main train, seeded sample.
- `math_train.jsonl`: `EleutherAI/hendrycks_math` train (all 7 subjects; MATH-500
  is from the test split), stratified by level in MATH-500's proportions
  (13/27/32/38/40), MATH-500's template.
- `mbpp_heldout.jsonl`: full MBPP prompt/validation/train splits (IDs 1–10,
  511–974), minus every MBPP+ ID (EvalPlus draws on all splits): all 320 left;
  sanitized MBPP's template.

Each is checked disjoint from `opencompass/` (IDs and problem text).

## Removed 2026-10-01

- the per-benchmark folders (`gsm8k/`, `gsm8k_full/`, `math_500/`,
  `mbpp_plus/`, `humaneval_plus/`), whose test files now live in `full/`;
- the frozen GSM8K-500 file (a subset of `full/gsm8k.jsonl`);
- the 20–100-task `dev.jsonl` pilot sets, which no launcher used.

exp04's launchers still point at the old paths. exp03 keeps its own
`exp03_dream_adaptive_sampling_ablation/data/`.
