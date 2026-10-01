# Peer group: parallel / speculative decoding for diffusion LMs

A tracking list of the methods MI_decoding will be compared with or cited
against. Last updated 2026-10-01. "Checked" means the numbers were read from
the paper's HTML/PDF; "abstract" means only the abstract or search snippet was
read, so verify those before citing.

## Closest neighbours (compare directly)

| method | arXiv | models | what it does | relation to the MI rule | checked |
|---|---|---|---|---|---|
| **SimSD** | [2606.02544](https://arxiv.org/abs/2606.02544) | **SDAR-1.7B drafts, SDAR-8B verifies**, B=4 and B=8 | Token-level speculative decoding for dLLMs: a mask adds draft "reference tokens" with temporally valid context, and the target verifies them | Same model pair with the roles the other way round (small proposes, big verifies). Our default is 8B proposes / 1.7B gates. The most direct speed comparison | checked |
| **S2D2** | [2603.25702](https://arxiv.org/abs/2603.25702) | SDAR 1.7B/4B/8B, Fast-dLLM v2, LLaDA2.1-mini | Training-free self-speculation: the same block model at block size 1 (≈ AR) verifies its own block-diffusion drafts | Same family; uses a **tuned dynamic decoding** baseline, the one our `dynamic:θ` arm mirrors | checked (main table) |
| **PUNT** | [2510.21961](https://arxiv.org/abs/2510.21961) | LLaDA-1.5, Dream-v0-Instruct-7B | Approximate conditional-independence test: KL(p(·\|anchor tokens) ‖ p(·)) below ε certifies a token for parallel unmasking; divide and conquer, O(log\|M\|) forwards per step | **Closest algorithm.** Same idea, the same model's conditional vs marginal. We use the pointwise log-ratio of the chosen token along a sequential prefix, scored by a different, smaller verifier | checked (setup) |
| **APD** | [2506.00413](https://arxiv.org/abs/2506.00413) | Dream-7B + Qwen2.5-0.5B (AR helper) | Number of tokens per step from a multiplicative mixture of dLLM marginals and a small AR model's joint; knobs R (mixture), W (KV window), M (masked lookahead) | **Closest in spirit:** a small auxiliary model decides how many tokens commit. Theirs is AR, ours a diffusion verifier | checked |
| **Spiffy** | [2509.18085](https://arxiv.org/abs/2509.18085) | LLaDA, Dream, SDAR | Lossless speculative decoding; drafts are sampled from the target itself and arranged as a draft graph | Lossless reference point on SDAR; composes with KV cache and threshold unmasking. The arXiv listing's title is "Structuring the Future: Diffusion LLM Speculative Decoding via Calibrated Draft Graphs" | abstract |

## Commitment rules and lossless schemes (baselines / citations)

| method | arXiv | models | one line | checked |
|---|---|---|---|---|
| Fast-dLLM | [2505.22618](https://arxiv.org/abs/2505.22618) | LLaDA, Dream | Confidence-threshold parallel decoding (= SDAR's `low_confidence_dynamic`) + approximate block KV cache; up to 27.6× throughput | abstract |
| EB-Sampler | [2505.24857](https://arxiv.org/abs/2505.24857) | LLaDA, Dream | Commit the longest confidence-ordered prefix with Σ H − max H ≤ γ (Eq. 2). Not run as an arm (decided 2026-10-01) | checked |
| SSD (Self Speculative Decoding) | [2510.04147](https://arxiv.org/abs/2510.04147) | LLaDA, Dream | Model drafts and verifies itself with hierarchical trees in one forward; lossless vs stepwise; up to 3.46× | abstract |
| FreeDave | [2510.00294](https://arxiv.org/abs/2510.00294) | several dLLMs | Lossless draft-and-verify that reproduces static sampling exactly; up to 2.83× | abstract |
| PSD | [2605.15609](https://arxiv.org/abs/2605.15609) | — | Training-free spatial (adaptive unmasking) + temporal (multi-depth speculative drafts) parallelism; up to 5.5 tokens/forward | abstract |

To triage (not yet read): Prophet / "dLLMs know the answer before decoding"
[2508.19982](https://arxiv.org/abs/2508.19982), DAWN [2602.06953](https://arxiv.org/abs/2602.06953),
Longest Stable Prefixes [2603.05454](https://arxiv.org/abs/2603.05454),
Locally Coherent Parallel Decoding [2603.20216](https://arxiv.org/abs/2603.20216),
Mean-Field Parallel Decoding [2606.15805](https://arxiv.org/abs/2606.15805),
From Bits to Rounds [2511.21103](https://arxiv.org/abs/2511.21103),
Block Verification for speculative diffusions [2606.13426](https://arxiv.org/abs/2606.13426),
CForce [2608.13925](https://arxiv.org/abs/2608.13925). Learned (trained) unmasking policies,
for context only: Learning to Parallel [2509.25188](https://arxiv.org/abs/2509.25188),
dParallel [2509.26488](https://arxiv.org/abs/2509.26488),
Learning Unmasking Policies [2512.09106](https://arxiv.org/abs/2512.09106).

## Context

- Block-diffusion models: SDAR [2510.06303](https://arxiv.org/abs/2510.06303),
  Fast-dLLM v2 [2509.26328](https://arxiv.org/abs/2509.26328), D2F [2508.09192](https://arxiv.org/abs/2508.09192).
- dLLM as the *drafter* for an AR target (the opposite direction; cite only):
  SpecDiff-2 [2511.00606](https://arxiv.org/abs/2511.00606), Bastion [2605.29727](https://arxiv.org/abs/2605.29727),
  PRESTO [2607.22634](https://arxiv.org/abs/2607.22634), Cost-Aware Diffusion Draft Trees [2606.01813](https://arxiv.org/abs/2606.01813).
- Benchmarks and theory: ParallelBench [2510.04767](https://arxiv.org/abs/2510.04767),
  error bounds and optimal schedules for factorized MDM sampling [2510.25544](https://arxiv.org/abs/2510.25544).

## Speed sanity check: tokens/s on SDAR, batch 1

| source | model / setting | hardware, engine | tok/s (GSM8K) |
|---|---|---|---|
| ours (exp06, e2e incl. prefill) | SDAR-8B-b64 `fixed:1` | 2× RTX A4000 (pipeline split), bf16, HF eager | 14.6 |
| ours | SDAR-8B-b64 `fixed:2` / `fixed:4` | same | 28.0 / 54.5 |
| ours | MI, 8B proposes / 1.7B verifies, δ 0.1 / δ 1 | same | 22.2 / 32.4 |
| ours | SDAR-1.7B-b64 `fixed:1` | 1× A4000, bf16, HF eager | 23.9 |
| ours, **SimSD's Vanilla protocol** (2026-10-01; 20 GSM8K tasks, 1 token/step, max 512, 3 warmups, decode tok/s) | SDAR-8B, B=4 / B=8 / B=64 (b64 weights at B=4/8: timing only, out of distribution) | 2× A4000 pipeline, bf16, HF eager | **15.1 / 16.5 / 14.3** (52 / 53 / 58 ms per forward) |
| SimSD Table 1 (decode only) | SDAR-8B, B=4, Vanilla | 2× RTX PRO 6000 Blackwell, FP32, TP=2, KV cache (engine not stated) | 10.0 |
| SimSD | same, Vanilla + CUDA Graph | same | 50.1 |
| SimSD | SimSD (1.7B drafts) | same | 72.6 |
| APD (context, not SDAR) | Qwen2.5-7B AR | 1× A5000 | 38.6 |
| S2D2 | SDAR-8B: dynamic 2.6×, S2D2 3.8× over AR (block size 1) | 1× H100 | speedups only |

Reading:
- Under SimSD's own protocol our harness gives 15.1 tok/s at B=4 and 16.5 at
  B=8, against their Vanilla's 10.0 and 15.3. Ours scales with forward count
  (1.27 → 1.16 forwards per token, 1.09×); theirs gains 1.53×, which points to
  extra per-block overhead in their Vanilla. At B=64, about 17% of forwards
  decode positions after the stop token inside the last block, which output
  tok/s does not count.
- Our eager numbers sit in the same range as SimSD's eager "Vanilla", so the
  harness is not unusually slow.
- Every strong published baseline uses CUDA graphs (SimSD's +CG is 5.4× its
  eager Vanilla). Absolute tok/s against those needs CUDA graphs here too.
  Until then, compare speedups measured on the same engine and hardware.
- For the 8B, per-forward cost is near the memory-bandwidth floor: 57 ms
  measured vs about 36 ms for 16 GB at 448 GB/s.
- The 1.7B verifier is overhead-bound: 34 ms measured vs about 8 ms. Eager
  mode therefore understates the MI rule's speedup; CUDA graphs would cut the
  verifier's share most.
