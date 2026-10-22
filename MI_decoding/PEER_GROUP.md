# Peer group: parallel / speculative decoding for diffusion LMs

A tracking list of the methods MI_decoding will be compared with or cited
against. Last updated 2026-10-12. "Checked" means the numbers were read from
the paper's HTML/PDF; "abstract" means only the abstract or search snippet was
read, so verify those before citing. Reported quality and speed for the
speculative decoders and other SDAR decoders are in the two sections at the
end, "Speculative decoders against dynamic:θ" and "Other decoders built on SDAR".

## Closest neighbours (compare directly)

| method | arXiv | models | what it does | relation to the MI rule | checked |
|---|---|---|---|---|---|
| **SimSD** | [2606.02544](https://arxiv.org/abs/2606.02544) | **SDAR-1.7B drafts, SDAR-8B verifies**, B=4 and B=8 | Token-level speculative decoding for dLLMs: a mask adds draft "reference tokens" with temporally valid context, and the target verifies them | Same model pair with the roles the other way round (small proposes, big verifies). Our default is 8B proposes / 1.7B gates. The most direct speed comparison | checked |
| **S2D2** | [2603.25702](https://arxiv.org/abs/2603.25702) | SDAR 1.7B/4B/8B, Fast-dLLM v2, LLaDA2.1-mini | Training-free self-speculation: the same block model at block size 1 (≈ AR) verifies its own block-diffusion drafts | Same family; uses a **tuned dynamic decoding** baseline, the one our `dynamic:θ` arm mirrors | checked (main table) |
| **PUNT** | [2510.21961](https://arxiv.org/abs/2510.21961) | LLaDA-1.5, Dream-v0-Instruct-7B | Approximate conditional-independence test: KL(p(·\|anchor tokens) ‖ p(·)) below ε certifies a token for parallel unmasking; divide and conquer, O(log\|M\|) forwards per step | **Closest algorithm.** Same idea, the same model's conditional vs marginal. We use the pointwise log-ratio of the chosen token along a sequential prefix, scored by a different, smaller verifier | checked (setup) |
| **APD** | [2506.00413](https://arxiv.org/abs/2506.00413) | Dream-7B + Qwen2.5-0.5B (AR helper) | Number of tokens per step from a multiplicative mixture of dLLM marginals and a small AR model's joint; knobs R (mixture), W (KV window), M (masked lookahead) | **Closest in spirit:** a small auxiliary model decides how many tokens commit. Theirs is AR, ours a diffusion verifier | checked |
| **Spiffy** | [2509.18085](https://arxiv.org/abs/2509.18085) | LLaDA, Dream, SDAR | Lossless speculative decoding; drafts are sampled from the target itself and arranged as a draft graph | Lossless reference point on SDAR; composes with KV cache and threshold unmasking. The arXiv listing's title is "Structuring the Future: Diffusion LLM Speculative Decoding via Calibrated Draft Graphs" | checked (SDAR table) |

## Commitment rules and lossless schemes (baselines / citations)

| method | arXiv | models | one line | checked |
|---|---|---|---|---|
| Fast-dLLM | [2505.22618](https://arxiv.org/abs/2505.22618) | LLaDA, Dream | Confidence-threshold parallel decoding (= SDAR's `low_confidence_dynamic`) + approximate block KV cache; up to 27.6× throughput | abstract |
| EB-Sampler | [2505.24857](https://arxiv.org/abs/2505.24857) | LLaDA, Dream | Commit the longest confidence-ordered prefix with Σ H − max H ≤ γ (Eq. 2). Not run as an arm (decided 2026-10-01) | checked |
| SSD (Self Speculative Decoding) | [2510.04147](https://arxiv.org/abs/2510.04147) | LLaDA, Dream | Model drafts and verifies itself with hierarchical trees in one forward; lossless vs stepwise; up to 3.46× | checked (speed table) |
| FreeDave | [2510.00294](https://arxiv.org/abs/2510.00294) | TraDo-4B/8B (SDAR-based), Dream-7B | Lossless draft-and-verify that reproduces static sampling exactly; up to 2.83× | checked |
| PSD | [2605.15609](https://arxiv.org/abs/2605.15609) | Dream-7B, LLaDA-1.5, openPangu-7B-Diffusion | Training-free spatial (adaptive unmasking) + temporal (multi-depth speculative drafts) parallelism; up to 5.5 tokens/forward | summary |
| Trajectory-level SD | [2608.27514](https://arxiv.org/abs/2608.27514) | LLaDA-7B, Dream-7B | Tree-structured draft trajectories verified blockwise, plus inter-block lookahead; on top of Fast-dLLM dual cache | checked (main tables) |

To triage (not yet read): Prophet / "dLLMs know the answer before decoding"
[2508.19982](https://arxiv.org/abs/2508.19982), DAWN [2602.06953](https://arxiv.org/abs/2602.06953),
Longest Stable Prefixes [2603.05454](https://arxiv.org/abs/2603.05454),
Locally Coherent Parallel Decoding [2603.20216](https://arxiv.org/abs/2603.20216),
Mean-Field Parallel Decoding [2606.15805](https://arxiv.org/abs/2606.15805),
From Bits to Rounds [2511.21103](https://arxiv.org/abs/2511.21103),
CForce [2608.13925](https://arxiv.org/abs/2608.13925). Learned (trained) unmasking policies,
for context only: Learning to Parallel [2509.25188](https://arxiv.org/abs/2509.25188),
dParallel [2509.26488](https://arxiv.org/abs/2509.26488),
Learning Unmasking Policies [2512.09106](https://arxiv.org/abs/2512.09106).
Not relevant after reading: Block Verification [2606.13426](https://arxiv.org/abs/2606.13426)
is speculative sampling for continuous (image) diffusions.

## Context

- Block-diffusion models: SDAR [2510.06303](https://arxiv.org/abs/2510.06303),
  Fast-dLLM v2 [2509.26328](https://arxiv.org/abs/2509.26328), D2F [2508.09192](https://arxiv.org/abs/2508.09192).
- dLLM as the *drafter* for an AR target (the opposite direction; cite only):
  SpecDiff-2 [2511.00606](https://arxiv.org/abs/2511.00606), Bastion [2605.29727](https://arxiv.org/abs/2605.29727),
  PRESTO [2607.22634](https://arxiv.org/abs/2607.22634), Cost-Aware Diffusion Draft Trees [2606.01813](https://arxiv.org/abs/2606.01813).
- Benchmarks and theory: ParallelBench [2510.04767](https://arxiv.org/abs/2510.04767),
  error bounds and optimal schedules for factorized MDM sampling [2510.25544](https://arxiv.org/abs/2510.25544).

## How the peers evaluate code (read 2026-10-03)

Every paper grades on **base tests**; only Fast-dLLM v2 also reports EvalPlus plus. MBPP's task set depends on the harness.

| paper | harness | HumanEval | MBPP |
|---|---|---|---|
| SDAR (2510.06303) | OpenCompass (`eval_sdar_hf.py`), gen 4096, **fp16** | 164, base, instruction prompt, first ``` block | **sanitized, 257** (ids 11–479), 3 asserts in the prompt, graded on them |
| SimSD (2606.02544) | OpenCompass default prompts and chat templates, 0-shot, max length 512, **FP32** | not run | pass@1; vanilla SDAR-8B B=4 66.9 (most likely SDAR's sanitized config; not stated) |
| S2D2 (2603.25702) | lm-eval-harness; own scripts for SDAR code; **fp16** for SDAR (code default in `phymhan/S2D2`, not stated in the paper) ("You are an expert Python programmer. Write only the function code, no explanations.") | base | base (lm-eval's default MBPP is the 500-task test split, 3-shot) |
| Spiffy (2509.18085) | not stated; SDAR-8B-b32, gen 512; "reduced precision" (fp16 or bf16, unstated) | base (no + variant) | base (no + variant) |
| Fast-dLLM, D2F, SSD, APD, PUNT | lm-eval-harness | 0-shot, base | 3-shot (Fast-dLLM, D2F) or 0-shot (SSD), base |
| Fast-dLLM v2 (2509.26328) | EvalPlus | base **and** plus | base **and** plus |
| FreeDave (2510.00294) | TraDo templates, 0-shot | base | base |

Precision: APD states BF16. The other lm-eval papers do not say. No SDAR-family paper uses bf16.

Ours since 2026-10-03: SDAR's OpenCompass protocol in full (prompts, graders, sanitized MBPP, gen 4096, fp16, whole vocabulary). FP32, SimSD's choice, does not fit: the 8B alone would be ~32 GB.

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

## Speculative decoders against dynamic:θ (read 2026-10-12)

The numbers in this section and the next were read from each paper's arXiv
HTML on 2026-10-12, and ASRD's from its PDF. They came through a fetch tool
that summarizes pages, and ambiguous tables were queried a second time. Check a
number against the paper before citing it.

Each paper uses its own harness, prompts, generation length, precision and
sampling temperature. So compare speedups within a paper, not scores or tok/s
across papers. Two things matter most when reading these tables:

- **Temperature.** Spiffy, SimSD, FreeDave and LightningRL's training sample at
  temperature 1.0. Threshold decoding loses accuracy when sampling: FreeDave's
  τ = 0.9 baseline drops 1.4–10 points against static decoding. At greedy, which
  is our setting, dynamic θ = 0.95 at B=4 loses nothing (below).
- **The baseline.** Several papers measure speed against one token per step,
  some without CUDA graphs, rather than against a tuned threshold rule.

### Our reference: SDAR-8B, B=4, greedy, OpenCompass protocol

This is report v7. Speed is over the loop-free tasks: tokens per second, 2×
A4000, HF eager, fp16, batch 1. The speedup is against fixed:1 on the same
checkpoint.

| | GSM8K | MATH-500 | HumanEval | MBPP |
|---|---|---|---|---|
| fixed:1: score, tok/s | 90.8, 15.1 | 75.0, 14.5 | 78.0, 14.5 | 72.4, 14.2 |
| dynamic θ=0.95: score, tok/s (speedup) | 90.9, 25.7 (1.70×) | 75.0, 26.8 (1.85×) | 78.0, 23.9 (1.65×) | 72.4, 18.9 (1.33×) |
| dynamic θ=0.90: score, tok/s (speedup) | 90.5, 27.1 (1.79×) | 75.2, 28.0 (1.93×) | 76.8, 25.6 (1.77×) | 71.2, 20.5 (1.44×) |
| tokens/step at θ=0.95 (before the stop token) | 1.96 (1.93) | 2.33 (2.31) | 2.01 (1.94) | 1.48 (1.41) |

SDAR's published B=4 scores for the 8B are 91.3 / 78.6 / 78.7 / 72.0. Its README
claims ">2× speedup over static inference with negligible accuracy loss", and
recommends threshold 0.9.

### Verdict

No paper shows a speculative decoder that beats a tuned dynamic threshold rule
on both quality and speed, on all four benchmarks, on SDAR at greedy.

- **The lossless methods** match their reference's quality.
  - **Spiffy** is 1.23–1.33× faster than dynamic τ=0.9, but only when stacked
    on top of it.
  - **FreeDave** runs at the same speed as τ=0.9, with better quality. The
    gap is most likely a temperature-1.0 effect: at greedy, our dynamic
    θ=0.95 loses nothing against fixed:1.
- **S2D2** is a trade-off, not a dominance.
  - Config-A matches or beats dynamic's quality but is slower or equal.
  - Config-B is faster but loses up to 3.6 points on HumanEval.
- **SimSD** never runs a dynamic baseline. Against CUDA-graph vanilla it is
  1.45× faster on GSM8K at −2.6 points, and 1.42× on MBPP at −0.7. In our
  harness, dynamic θ=0.95 is 1.70× and 1.33× faster than fixed:1 on those two
  benchmarks, at no loss. The engines differ, so take this only as
  orientation.
- **On LLaDA/Dream**, Trajectory-level SD is 1.14–1.49× faster than Fast-dLLM's
  threshold rule (dual cache), at −0.9 to +3.1 points. This is the clearest
  "beats threshold on both" result, but it is not on SDAR.
- **MATH-500 on SDAR** appears in only one of these papers, Spiffy.

### S2D2 (2603.25702), SDAR-1.7B/4B/8B-Chat

Setup:
- One H100, batch 1.
- Speedup is wall-clock against the same model at block size 1 ("AR"), with
  the KV cache.
- The diffusion baseline is static/dynamic confidence decoding at B=4, with
  "default or recommended" thresholds (value not stated).
- The paper's own scripts grade GSM8K and HumanEval.
- Temperature is not stated in what was read.
- No MATH.

SDAR-8B-Chat: score (speedup over AR)

| | GSM8K | MBPP | HumanEval |
|---|---|---|---|
| AR (B=1) | 89.3 (1.0×) | 64.4 (1.0×) | 75.6 (1.0×) |
| diffusion dynamic, B=4 | 89.3 (2.6×) | 60.6 (2.1×) | 78.0 (2.9×) |
| S2D2 config-A (B=4, score-threshold routing) | 89.6 (2.0×) | 62.0 (2.1×) | 78.7 (2.2×) |
| S2D2 config-B (B=16, always-on verification) | 88.3 (3.8×) | 61.4 (2.8×) | 74.4 (3.2×) |

On the 1.7B, config-B is faster than dynamic on all three, at −3.4 / +22.8 /
+0.6 points on GSM8K / MBPP / HumanEval. Dynamic collapses on the 1.7B's MBPP
(21.6 against AR's 48.4).

Absolute speed: Table 5 gives batched throughput. One value read at batch 1
was "BD3 dynamic 38 tok/s (GSM8K@128)", but the setting is unclear.

### Spiffy (2509.18085), SDAR-8B-Chat-b32

Setup:
- One A100, gen 512, B=32, temperature 1.0.
- Speedup is TPS against one token per step with prefix KV cache.
- Dynamic is threshold τ=0.9.
- Spiffy runs on top of dynamic.
- No absolute tok/s.

| | GSM8K | HumanEval | MATH-500 | MBPP |
|---|---|---|---|---|
| baseline (1 token/step): accuracy | 0.90 | 0.68 | 0.49 | 0.26 |
| + dynamic τ=0.9: accuracy, TPS speedup | 0.88, 6.71× | 0.71, 5.54× | 0.47, 4.38× | 0.24, 3.99× |
| + Spiffy: accuracy, TPS speedup | 0.88, 8.25× | 0.71, 6.96× | 0.47, 5.46× | 0.25, 5.29× |

Spiffy over dynamic alone: 1.23× / 1.26× / 1.25× / 1.33× at the same accuracy.
These b32 scores are far below SDAR's B=4 numbers, at gen 512 with sampling.

### SimSD (2606.02544), SDAR-1.7B drafts and SDAR-8B verifies

Setup:
- Two RTX PRO 6000 Blackwell, FP32, KV cache, gen 512, temperature 1.0, batch 1.
- Decode tokens/s.
- No MATH, no HumanEval, no dynamic baseline.

| B | method | GSM8K tok/s | MBPP tok/s | GSM8K acc | MBPP pass@1 |
|---|---|---|---|---|---|
| 4 | Vanilla (1 token/step) | 10.0 | 11.4 | 92.2 | 66.9 |
| 4 | Vanilla + CUDA graph | 50.1 | 50.7 | — | — |
| 4 | S2D2 | 54.8 | 47.7 | — | — |
| 4 | SimSD | 72.6 | 72.2 | 89.6 | 66.2 |
| 8 | Vanilla / + CUDA graph | 15.3 / 30.9 | 18.7 / 30.4 | 90.6 | 64.2 |
| 8 | S2D2 | 57.6 | 50.2 | — | — |
| 8 | SimSD | 74.1 | 80.4 | 89.4 | 63.4 |

The headline 7.46× is against eager Vanilla. Against CUDA-graph Vanilla, SimSD
at B=4 is 1.45× on GSM8K and 1.42× on MBPP.

### FreeDave (2510.00294), TraDo-8B-Instruct (SDAR-8B after TraceRL RL)

Setup:
- One L40S, B=4, gen 2048, temperature 1.0.
- "Parallel" is threshold 0.9.
- FreeDave+ is lossless against static decoding.
- Score, then TPS (TPF).

| | MATH-500 | GSM8K | HumanEval | MBPP |
|---|---|---|---|---|
| static | 78.6, 23.7 (0.80) | 93.0, 23.0 (0.80) | 79.3, 23.9 (0.80) | 63.2, 22.6 (0.80) |
| parallel τ=0.9 | 74.0, 46.7 (1.73) | 91.6, 40.0 (1.53) | 76.2, 50.0 (1.68) | 53.2, 51.6 (1.74) |
| FreeDave+ | 79.0, 48.3 (1.89) | 93.0, 42.2 (1.85) | 78.7, 48.0 (1.90) | 63.2, 51.6 (2.00) |

Against τ=0.9, FreeDave+ runs at 0.96–1.05× the TPS and scores +1.4 to +10
points. Note that static TPF is 0.80, not 1: static spends an extra forward per
block.

### Not on SDAR (for context)

| method | models | quality and speed against a threshold rule |
|---|---|---|
| Trajectory-level SD (2608.27514) | LLaDA-7B, Dream-7B, B=32, gen 512, A800 | LLaDA TPS (Fast-dLLM dual cache → ours): GSM8K 48.8 → 55.7 (75.6 → 75.5), HumanEval 70.6 → 94.4 (45.7 → 46.3), MBPP 48.5 → 56.4 (13.6 → 13.4), MATH 60.7 → 73.5 (35.7 → 36.0). Tokens/step 2.6 → 4.3 |
| SSD (2510.04147) | LLaDA, Dream, B=8, gen 256, A100 | Lossless against stepwise decoding; 2.03–2.37× TPS (e.g. Dream-Instruct 16.3 → 38.5 tok/s), up to 3.46× on MBPP. No threshold baseline |
| PSD (2605.15609) | Dream, LLaDA-1.5, openPangu, B=32, V100 | Claims greedy-level accuracy at 2.7–4.0× on GSM8K, where threshold decoding loses accuracy (summary only) |
| DFlash, DFlow, SpecDiff-2 | dLLM drafts, AR target | The opposite direction; cite only |

## Other decoders built on SDAR (read 2026-10-12)

Best reported quality against speed, training-free and trained. TPF is tokens
per forward pass.

| method | type | model, block | protocol, hardware | GSM8K | MATH-500 | HumanEval | MBPP | speed |
|---|---|---|---|---|---|---|---|---|
| SDAR dynamic (LightningRL's baseline) | — | SDAR-8B, b32, τ=0.9 | greedy eval, max length 8192 | 88.9 | 63.6 | 73.5 | 58.0 | TPF 2.85 / 4.81 / 2.39 / 2.44; **105.6 tok/s** GSM8K (H100, SGLang, TP1, batch 1) |
| **LightningRL** (2603.13319) | RL fine-tune for accuracy + TPF | SDAR-8B, b32 | same | 90.3 | 63.0 | 72.6 | 58.3 | TPF 5.58 / 6.28 / 6.30 / 11.10; **336.0 tok/s** GSM8K. At b4: SDAR 91.1 @ 2.35 TPF vs LightningRL 91.0 @ 3.21 (GSM8K) |
| **MBD-SDAR** (2606.29215) | post-training (multi-block teacher forcing) | SDAR-8B, b4 | H100 TP=2; EvalPlus "+" for code | 91.8 (vs 91.1) | 72.4 (vs 72.8) | 72.6+ (vs 73.7+) | 64.3+ (vs 64.8+) | TPF 2.28 / 2.52 / 2.24 / 2.62 vs native 1.33 / 1.46 / 1.07 / 1.13; no tok/s |
| ASRD (2606.16847) | training-free (anchor-guided generation + verification) | SDAR-8B, b32 | gen 512, A100 | 84.2 → 84.7 | 47.4 → 48.8 | 62.1 → 63.3 | 59.4 → 60.2 | wall-clock 1.9× / 1.7× / 1.5× / 1.3× over the native decoder |
| S2D2 config-B | training-free self-speculation | SDAR-8B, B=16 | H100 | 88.3 | — | 74.4 | 61.4 | 3.8× / 2.8× / 3.2× over AR (B=1) |
| Spiffy + dynamic | training-free, lossless | SDAR-8B, b32 | A100, T=1 | 0.88 | 0.47 | 0.71 | 0.25 | 8.25× / 5.46× / 6.96× / 5.29× TPS over 1 token/step |
| SimSD | training-free speculation, 1.7B drafts | SDAR-8B, B=4 | FP32, T=1 | 89.6 | — | — | 66.2 | **72.6** / 72.2 tok/s (2× RTX PRO 6000) |
| FreeDave+ | training-free, lossless | TraDo-8B, B=4 | L40S, T=1 | 93.0 | 79.0 | 78.7 | 63.2 | 42–52 tok/s (1.85–2.00 TPF) |
| HOPD (2609.33301) | on-policy distillation from TraDo-8B | SDAR-4B, B=4, τ=0.9 | bf16 | 88.9 → 90.7 | 66.1 → 72.5 | — | — | quality method; 11% more tokens/step than TOPD on MATH-500 |
| LoPA's SDAR row (2512.16229, Table 7) | baseline row, LMDeploy | SDAR-8B-Chat | hardware not stated for this row | 91.3 | — | — | 72.0 | 273.3 / 234.9 tok/s (TPF 2.10 / 1.60). Scores equal SDAR's published ones; treat as unverified |

Batched serving, not per-request latency:
- **JetEngine** (SDAR README): SDAR-4B at B=4 and batch 128 reaches 1800+ tok/s
  on A800 and 3700+ on H200.
- **HERALD**: 1,364 tok/s on SDAR-8B-Chat with 16K context at batch 90, against
  754 for dense. Source: the sota2.com leaderboard; the paper was not read.

Reading:
- **Highest batch-1 tok/s on an SDAR-8B-family model:** LightningRL, 336 tok/s
  on GSM8K at 90.3. It uses retrained weights at b32, on H100 with SGLang. The
  untouched SDAR-8B-b32 dynamic baseline gets 105.6 tok/s on the same stack.
- **Best training-free:**
  - SimSD reaches 72.6 tok/s at B=4, at −2.6 points on GSM8K, on a
    CUDA-graph FP32 stack.
  - In relative terms, Spiffy + dynamic reaches 5.3–8.3× over one token per
    step at b32.
- **Our absolute tok/s are engine-bound.** We get 25.7 (B=4 dynamic) and 38.4
  (b64 dynamic) on A4000s with HF eager, against SGLang's 105.6 on an H100 at
  b32. Comparing against these papers' absolute numbers needs our MI rule on
  the same engine (CUDA graphs at least). Until then, compare speedups over
  fixed:1 or dynamic.
- **The trained methods** (LightningRL, MBD-SDAR, HOPD, TraDo) change the
  weights. They are a different axis from a decoding rule, and the MI rule
  could run on top of any of them.
