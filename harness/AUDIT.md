# Harness audit against SDAR's published evaluation

## 2026-10-03: the harness now runs SDAR's own protocol

Every MI_decoding launcher defaults to SDAR's OpenCompass eval config
(`eval_sdar_hf.py` and the four dataset configs it imports). The prompts and
graders are ported verbatim into `sdar_harness/opencompass.py`, and the data is
in `../data/opencompass/` (built by `scripts/prepare_opencompass.py`).

| | SDAR (OpenCompass) | harness now |
|---|---|---|
| checkpoint / block | `SDAR-*-Chat` (b4), B=4, 1 token per step (`threshold=1.0`) | same (`fixed:1`); `dynamic:θ` is its `threshold=θ` |
| prompts | one HUMAN turn per config, through the chat template | same text (`opencompass.PROMPTS`), same template call |
| GSM8K | 1,319; `math_postprocess_v2` + `MATHEvaluator(version='v2')` | same, ported |
| MATH-500 | prm800k 500; gold `extract_boxed_answer(solution)`; `MATHVerifyEvaluator` | same problems, gold identical on all 500, same `math_verify` call |
| HumanEval | openai 164; first fenced block; human-eval base tests, 3 s | same data and grading (`score_opencompass.py`) |
| MBPP | sanitized test, 257; `_process_answer`; its 3 asserts, 9 s | same data and grading |
| generation | `gen_length=4096`; output cut at the stop tokens | 4096; early stop, identical output |
| dtype | float16 | float16 |
| vocabulary | whole (151,936 logits) | whole; only the mask token is barred |

What still differs, all expected to be rounding-level:
- Kernels: SDPA and the authors' PyTorch RMSNorm in place of flash-attn (§1).
- `math_verify` 0.9.0. OpenCompass's version is not pinned. 0.9.0 ignores the
  config's `equations=True`, which is deprecated there.
- Code runs in a subprocess (`exec(program, {})`, stdin closed), not in
  OpenCompass's multiprocessing worker. The semantics are the same, and only
  the timeout path is load-sensitive.
- The mask token can never be committed. The reference could commit it, in
  principle, and then strips `<|MASK|>` from the text.

## 2026-10-06: block 64 (`SDAR-*-Chat-b64`)

Before the b64 baseline run (`outputs/opencompass/SDAR-{8B,1.7B}-Chat-b64`),
the harness was checked against the B=4 run and against SDAR's decoder.

- **Checkpoints.** The b64 and b4 copies of each size share their tokenizer,
  chat template, `generation_config` (stop ids 151645, 151643), `config.json`
  and patched `modeling_sdar.py` byte for byte. Only the weights differ, and
  they come from `JetLM/SDAR-{8B,1.7B}-Chat-b64`. The name guard reads 64.
- **Recipe.** It is the same launcher (`launch_baselines.sh`), protocol, data
  hashes, fp16, 4096 canvas, warmups and seed as the B=4 run, with
  `max_candidates` = 64. The 8B runs on the B=4 run's cards (2,3).
- **The reference at B=64.** `eval_sdar_hf.py` sets `denoising_steps =
  block_length` for any block. So `threshold=1.0` is `fixed:1`, and
  `threshold=θ` is `dynamic:θ` with a floor of one token per step.
  `denoising_steps = 64/k` is `fixed:k`.
- **Token-for-token check** (`../MI_decoding/logs/opencompass/b64_provenance/equiv_b64.py`).
  SDAR's `block_diffusion_generate` was copied verbatim from `JetAstra/SDAR`
  and run on the same loaded 8B-b64. Its logits were moved back to the first
  card, since the reference assumes one GPU. It was compared with the harness
  on 2 GSM8K, 1 HumanEval, 1 MBPP and 1 MATH prompt:
  - **fixed:1, fixed:2, dynamic:0.95 and dynamic:0.90:** identical on 20 of 20.
  - **fixed:8:** 2 of 5 differ (HumanEval/0 at token 125, MATH
    precalculus/807 at token 151). Both become identical once the reference's
    softmax runs in float32. The reference ranks by fp16 probabilities and the
    harness by float32 log-probabilities. At 8 tokens per step, the 8th pick
    falls among close candidates, where fp16 rounding reorders them. The
    harness is kept as is so that b4 and b64 stay consistent.
  - **The canvas tail:** a third fixed:8 case (gsm8k-test-1) loops to the
    limit in both and is identical through 4096 tokens. The reference also
    decodes the up to B−1 positions that round the canvas up to a whole block.
    The harness leaves them masked. This only affects answers that reach the
    4096 limit.

Legacy runs (exp04–06 and the 2026-10-01 B=4 run) used the prompts in
`../data/full/` and `../data/sample100/`. Their records have no `protocol`
field and keep being graded as before. Everything below is the original
2026-09-30 audit of that legacy protocol.

---

## 2026-10-11: re-timed SDAR-8B-b64 baseline tasks

An outside load slowed cards 2,3 during parts of the 8B-b64 baseline run on
2026-10-06, by 2–9× per forward. Scores were unaffected. The timings of
the following cells were not:

| cell | tasks | ms/forward | cell tok/s |
|---|---|---|---|
| MATH fixed:1 | 9–62 and 292–302 | 548 → 63 | 7.5 → 12.3 |
| MBPP dynamic:0.95 | 116–197 | 111 → 58 | 20.6 → 26.1 |
| MBPP dynamic:0.90 | 114–175 | 256 → 58 | 18.3 → 31.7 |
| MBPP fixed:4 | 155–213 | 255 → 59 | 18.0 → 31.4 |

One more GSM8K task in each of dynamic:0.95 and fixed:8 also ran slow; its
prefill took about 0.4 s too long.

**Detection.** Each task's decode time per forward was compared against a cost
model fitted across all cells (51.9 ms plus 25.8 µs per context token). Each
prefill was compared against the same prompt's prefill in the other arms. The
windows are contiguous in time with sharp edges. The tasks just outside them
are at 0.94–1.04× the model, and no other 8B-b64 cell has a window.

**Re-time.** The run was on 2026-10-11, on the same cards, with the 10-06 code
rebuilt from `b64_provenance` and hash-checked. 270 tasks were re-timed, plus
52 clean neighbouring tasks as controls. All of it is in
`../MI_decoding/logs/opencompass/retime_20261011/`.

`splice.py` replaced only the timing and memory fields, and only after two checks:
- Every re-timed record matched its original in every other field, completion
  included.
- The controls' median new/original wall time fell in 0.95–1.05. It came in at
  1.012–1.029.

Each cell keeps its originals in `records.contaminated_20261006.jsonl` and logs
the replacement in `retime_log.json`. The pre-splice summary is
`comparison/summary.pre_retime_20261011.*`. A regenerated summary changes only
those cells' timing columns.

**Residual.** Host-load bursts on 10-11 (load average up to 516) slowed five
re-timed MATH tasks and one MBPP fixed:4 task by 10–36%. That is about 23 s
in a 28,000 s cell. They are listed under `residual_slow_tasks` in the
`retime_log.json` files.

**Not re-timed (user's call):** three SDAR-1.7B-b64 cells ran 8–10% slow
throughout from host CPU load on 10-08 and 10-09: MATH fixed:1, GSM8K fixed:1 and
MATH fixed:8.

## 2026-10-17: B=32 MI run, host-load timing corrected by model

The B=32 MI run (`MI_decoding/outputs/mi/8b_proposes_b32`, G=8) finished
2026-10-17 in one attempt. The audit found it complete and consistent:
- Counts, IDs and dataset hashes match. The code matches provenance except a
  mid-run `scoring.py` MemoryError guard, which never triggered.
- The GPU guard saw no foreign processes.
- Allocator OOM warnings came only on 4 runaway tasks, and all were retried.

The guard does not watch host CPU load. Three cells ran partly at load average
100–318, which slowed tasks by up to ~1.4×. Scores are unaffected. The user
chose a model correction over a re-time:

| cell | flagged tasks | cell tok/s |
|---|---|---|
| HumanEval δ0.10 | 130 / 164 | 21.2 → 24.5 |
| GSM8K δ3 | 437 / 1319 | 37.2 → 38.9 |
| MATH δ0.03 | 377 / 500 | 18.9 → 19.5 |

Every other cell is within 0.2% of its clean-task cost model.

`MI_decoding/logs/mi/8b_proposes_b32/load_correction.py` fits each cell on its
tasks below load 100 and caps flagged tasks at the model. Its output,
`load_correction.json`, lists the flagged IDs and the corrected wall and tok/s.
`summary.csv` keeps the measured values, so take these three cells' speed from
the JSON.

# Original audit (2026-09-30), legacy protocol

The reference is SDAR's own evaluation code: `JetAstra/SDAR`,
`evaluation/opencompass/configs/eval_sdar_hf.py`. Its decoder is
`opencompass/models/huggingface_bd3.py::block_diffusion_generate`, and its
dataset configs are the four it imports. The published numbers are Table 2 of
arXiv:2510.06303.

## Bottom line

- **The decoder is faithful.** Block grid, masks, KV-cache handling, confidence
  ordering and stop tokens all match the reference line for line (§1).
- **The benchmark protocol is not SDAR's.** Checkpoint, block size, prompts,
  generation length, scorers, task sets and dtype all differ (§2).
- **GSM8K lands on the published number, 80 vs 80.1 (1.7B) and 91 vs 91.3 (8B).**
  The decoder and model loading being right is consistent with that.
- **Code and MATH land 9–28 points below the published numbers** (§3). Nothing
  in the decoder explains the gap. It has to come from the protocol
  differences, and the run that would separate them has not been made (§5).
- **What this means for results so far:** the in-house paired comparisons
  (method A vs B on the same tasks, prompts and scorer) are internally valid.
  The absolute scores are not comparable to SDAR's table or to other dLLM
  papers. For a paper, the harness first needs a mode that reproduces SDAR's
  numbers.

## 1. What matches the reference (verified by reading both)

| | reference | ours (`blockwise.py`, `model.py`) |
|---|---|---|
| canvas | `ceil((prompt+gen)/B)*B`, prompt included | same |
| first block | prompt tail shares the first decode block, only masked slots filled | same |
| attention | block lower-triangular, `repeat_interleave` | same (`block_causal_mask`) |
| KV cache | prefill whole prompt blocks; denoise with `store_kv=False`; one extra `store_kv=True` pass per finished block | same, and the commit pass is counted |
| logits | position-aligned (no shift) | same |
| confidence | prob of the greedy token under the untruncated softmax | log p of the same token (same order) |
| stop | `<\|im_end\|>` 151645, `<\|endoftext\|>` 151643; stop after the block containing one | same set; stops as soon as the answer before the stop is settled (identical output, since blocks are causal) |
| chat template | `apply_chat_template(..., add_generation_prompt=True)`, single user turn | same |
| model code | upstream `modeling_sdar.py` with flash-attn | 3 patches: guarded flash import, the authors' own PyTorch RMSNorm, and SDPA where they call flash. All three compute the same function (bf16 rounding aside) |

## 2. Protocol differences from the published table

| | SDAR Table 2 | ours (exp05/06) | likely effect |
|---|---|---|---|
| checkpoint | `SDAR-*-Chat` (block 4) | `SDAR-*-Chat-b64` | unknown; §5.2 of the paper says small models are the least robust to block size |
| block / steps | B=4, 4 steps (1 token/step), static | B=64, `fixed_k_1` (1 token/step by confidence within 64) | as above; same tokens per step, but a far wider choice of which position goes next |
| generation length | `gen_length=4096` for every task | **matched 2026-10-03:** 4096 everywhere (MI_decoding launchers). exp05/06 used 256 GSM8K / 256 MBPP+ / 512 HE+ / 1024 MATH; the 10-01 B=4 runs used 512/2048/512/768 and re-decoded only their truncated tasks (`scripts/extend_canvas.py`) | exp05/06: MATH 13% (8B) and 16% (1.7B) of answers hit the canvas. B=4 at 2048: 1.8% / 3.8%. HumanEval at 768: 0 and 1 of 164, so canvas cannot explain the HumanEval gap |
| GSM8K prompt | `{q}\nPlease reason step by step, and put your final answer within \boxed{}.` | `Q: {q}\nA:` (lm-eval gsm8k_cot, 0-shot) | small (we match the published score) |
| GSM8K scorer | boxed answer, MATHEvaluator v2 | last number in the text (flexible-extract) | checked: every disagreement is an answer the model didn't box; ours is fair under our prompt |
| MATH prompt | `{problem}\nPlease reason step by step, and put your final answer within \boxed{}.` | `Solve the following math problem. Put your final answer inside \boxed{}.\n\nProblem: …\n\nSolution:` | unknown |
| MATH scorer | `math_verify` (symbolic) | **matched 2026-10-03:** `math_verify`, OpenCompass `MATHVerifyEvaluator` call for call. Before that a string normalizer, which under-credited by 3.8–4.0 pts on B=4 SDAR (19–20 answers per arm, none over-credited); old verdicts kept as `string_match` | none now |
| HumanEval | 164 tasks, original human-eval tests, instruction prefix "Read the following function signature and docstring, and fully implement the function…"; first ``` block (`humaneval_postprocess_v2`), prompt + completion + test, 3 s timeout | evalplus HumanEval+ (same 164 tasks), raw signature+docstring with no instruction; we report **plus** tests, and base | scoring checked 2026-10-03: SDAR's procedure on our B=4 completions gives 54.3 / 78.7 (1.7B / 8B fixed:1) vs our evalplus base 56.1 / 78.7, so the scorer is not the gap; the prompt is the main remaining difference |
| MBPP | sanitized MBPP test split = task ids 11–510, **257 tasks**; prompt "You are an expert Python programmer… Your code should pass these tests: {3 asserts}"; graded on those same 3 asserts (9 s timeout). The paper reports no MBPP+ or HumanEval+ | evalplus MBPP+ (378 tasks, 224 of them in SDAR's 257), docstring with **1** assert, **plus** tests | different task set and prompt. SDAR's 3-assert grading on our B=4 completions: 64.8 / 79.1 (1.7B / 8B fixed:1), 63.8 / 79.9 on the 224-task overlap, vs published 61.1 / 72.0 |
| task set | full test sets | fixed 100-task samples (GSM8K from a frozen 500 of 1319) | ±5 pt standard error per benchmark |
| dtype | float16 | bfloat16 | small |
| vocabulary | full | logits restricted to IDs < 151665 (shared with Dream; drops `<think>` etc.) | small for SDAR alone; needed only for the Dream verifier |

## 3. Scores: published (B=4) vs ours (b64, B=64, `fixed_k_1`, 100 tasks)

| | GSM8K | MATH-500 | HumanEval (base / plus) | MBPP (base / plus) |
|---|---|---|---|---|
| SDAR-1.7B published | 80.1 | 63.2 | 61.6 | 61.1 |
| SDAR-1.7B ours | 80 | 43 | 41 / 35 | 33 / 27 |
| SDAR-8B published | 91.3 | 78.6 | 78.7 | 72.0 |
| SDAR-8B ours | 91 | 62 | 70 / 60 | 48 / 40 |

The 1.7B MATH-500 number is exp04's run (canvas 1024, no early stop). Our base
pass rates come from evalplus's base tests. The comparison is only
approximate, since sanitized MBPP and MBPP+ are different task sets.

## 4. Smaller findings

- Two 8B MBPP+ answers are empty: the model commits `<|im_end|>` at position 0
  (Mbpp/64, Mbpp/769). This is model behaviour under B=64 confidence ordering,
  not a stop-logic bug: `finished_at` requires everything before the stop to
  be committed.
- Code syntax errors come almost entirely from the model. Re-extracting with a
  looser fence strip rescues at most 1 per 100 (8B MBPP+: 1 of 8). The extractor
  is not the bottleneck.
- `prompt_imports` repair (adds `from typing import List` etc. that the
  solution omits) fires on 11–14% of HumanEval+ answers. SDAR's
  `humaneval_postprocess_v2` scorer does not do this, but OpenAI's HumanEval
  harness runs prompt + completion, so imports come from the prompt there too.
  It's fine, but the paper should say so.
- Timing: one warmup per arm. Per-task CUDA sync on every card in the replica.
  Batch 1. HF eager kernels, so per-forward cost is overhead-bound (1.7B 34 ms
  vs 8B 57 ms per block forward). The 8B baseline and the cross8b runs used
  different card pairs (2,3 vs 0,1). Re-time on the same pair before quoting
  speedups.

## 5. To reach paper-grade integrity

1. **Add an `sdar_opencompass` protocol** to the runner. It copies the four
   OpenCompass prompts, uses `gen_length` up to 4096 with early stop, and
   scores with `math_verify` (GSM8K/MATH), sanitized MBPP with 3 asserts and
   base tests, and HumanEval base tests (also report evalplus plus). Keep the
   current protocol as a named alternative; don't edit it in place.
2. **Reproduce the paper:** `SDAR-1.7B-Chat` (b4, `../models/SDAR-1.7B-Chat-local`),
   B=4, static 1 token/step, full test sets. Target: within ~2 pts of
   80.1 / 63.2 / 61.6 / 61.1. This validates the harness end to end.
3. **Isolate block size:** the same protocol with `SDAR-1.7B-Chat-b64` at B=64.
   The difference from step 2 is the cost of b64, which every exp05/06 number
   carries.
4. Then the 8B, the same two ways (the b4 8B checkpoint is a 16 GB download).
5. From then on, report full test sets, base and plus for code, and paired
   McNemar tests for method comparisons.
