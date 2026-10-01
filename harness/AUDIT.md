# Harness audit against SDAR's published evaluation (2026-09-30)

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
| generation length | `gen_length=4096` for every task | 256 GSM8K / 256 MBPP+ / 512 HE+ / 1024 MATH | **MATH: 13% (8B) and 16% (1.7B) of answers hit the canvas**; GSM8K 1–6%; code 0–2% |
| GSM8K prompt | `{q}\nPlease reason step by step, and put your final answer within \boxed{}.` | `Q: {q}\nA:` (lm-eval gsm8k_cot, 0-shot) | small (we match the published score) |
| GSM8K scorer | boxed answer, MATHEvaluator v2 | last number in the text (flexible-extract) | checked: every disagreement is an answer the model didn't box; ours is fair under our prompt |
| MATH prompt | `{problem}\nPlease reason step by step, and put your final answer within \boxed{}.` | `Solve the following math problem. Put your final answer inside \boxed{}.\n\nProblem: …\n\nSolution:` | unknown |
| MATH scorer | `math_verify` (symbolic) | string normalizer plus numeric compare | **under-credits** equivalent forms (e.g. `\frac{\pi}{2}` vs `\pi/2`, tuples, intervals) |
| HumanEval | 164 tasks, base tests, instruction prefix "Read the following function signature and docstring, and fully implement the function…" | evalplus HumanEval+ (same 164 tasks), raw signature+docstring with no instruction; we report **plus** tests | plus is strictly harder (8B: base 70 vs plus 60) |
| MBPP | sanitized MBPP (test split), prompt "You are an expert Python programmer… Your code should pass these tests: {3 asserts}", base tests | evalplus MBPP+ (378 tasks), docstring with **1** assert, **plus** tests | fewer tests shown in the prompt and stricter tests at scoring (8B: base 48 vs plus 40) |
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
