# exp06_vgb: SDAR-8B-b64 scale-up

This experiment compares `JetLM/SDAR-8B-Chat-b64` with exp05's completed
`SDAR-1.7B-Chat-b64` fixed-k frontier on exactly the same sampled tasks.

## Scope

- Benchmarks: GSM8K, HumanEval+, and MBPP+.
- Sample: 100 tasks per benchmark, selected with seed `271828`.
- Methods: `fixed_k_{1,2,4,8}`.
- Canvases: GSM8K 256, HumanEval+ 512, MBPP+ 256.
- Block length: 64.
- Decode recipe: greedy (`temperature=0`), log-probability confidence,
  early stop at the first settled checkpoint stop token.

MATH-500 (canvas 1024) is included. exp05 deferred it, so its 1.7B reference
is ported from exp04 instead; see below.

## Ported exp05 reference

The complete exp05 SDAR baseline tree is copied verbatim to
`reference/sdar_1p7b_chat_b64/`. It contains raw records, scored EvalPlus
records, per-arm metadata, run metadata, and the aggregate comparison table.
The source and destination trees were checked with `diff -qr` after copying.

Useful entry points:

- `reference/sdar_1p7b_chat_b64/comparison/summary.csv`
- `reference/sdar_1p7b_chat_b64/gsm8k/`
- `reference/sdar_1p7b_chat_b64/humaneval_plus/`
- `reference/sdar_1p7b_chat_b64/mbpp_plus/`

The MATH-500 fixed-k arms are copied verbatim from exp04
(`outputs/baseline/sdar_math1024`) to `reference/sdar_1p7b_chat_b64_math500/`.
They use the same data file (SHA-256 match), the same seed, and the same 100
ordered task IDs as this run, and the current scorer reproduces their scores
exactly (0.43 / 0.42 / 0.26 / 0.02 for k = 1 / 2 / 4 / 8). exp04 ran without
early stop and did not stop at `<|im_end|>`. Because SDAR is block-causal, this
leaves every token before the stop unchanged. It can differ only when
`<|im_end|>` landed early and exp04 kept writing: exp05's cross-check on the
other three benchmarks found that in 0-6% of tasks at k >= 4 and never at
k <= 2. The timings are full-canvas and not comparable, so
`comparison/summary.exp06_vgb.json` marks them `timing_comparable: false` and
`compare_scale.py` prints them as n/a. To get comparable MATH timings, run
`REFERENCE_MATH=1 bash launch_sdar8b.sh`, which reruns 1.7B MATH after the 8B
phase.

The paths embedded in the historical metadata intentionally still name
`exp05_adaptive_re`; the artifacts are an immutable copy, not a rewritten run.

## Run status

The exp05 sampler, benchmark scripts, and tests are copied here unchanged
except for one addition: multi-GPU loading. The checkpoint is 16.4 GB in
BF16, and each local RTX A4000 has 16,376 MiB, so it cannot load on one card.

`--devices cuda:0 cuda:1` pipeline-splits the model (`shard_decoder` in
`adaptive_re/model.py`): embeddings and layers 0-17 on the first card, layers
18-35 plus the final norm and LM head on the second. Sampler state stays on the
second card. Each layer's K/V cache stays on that layer's card. Pre-forward
hooks move the mask, position IDs, and rotary tables to each card once per
forward. Hidden states cross between the cards once per forward.

- **Same function.** SDAR-1.7B-b64 split over two cards reproduced the
  single-card run and the exp05 records exactly for 4 GSM8K and 4 HumanEval+
  tasks at k=1 and k=4. The unit test is `tests/test_shard.py`.
- **Timing.** Every card in a replica is synchronized before and after each
  timed generation. Stage profiling uses the host-clock, sync-all-devices path.
  So `tokens_per_second` is wall-clock for the whole model.
  `peak_gpu_memory_bytes` is the sum over the replica's cards;
  `peak_gpu_memory_bytes_per_device` lists each card.
- **Memory.** SDAR-8B peaks at ~7.7 GB per card during block decoding. The
  full-canvas preflight worst case is ~9 GB.
- **Latency caveat.** The stages run one after another, so 8B latency
  includes a card-to-card hop and never uses both cards' compute at the same
  time. Measured on 1.7B, splitting costs 5-15% of wall time. 8B tokens/s is
  the latency of this deployment, not of 8B on one larger card.

Launch (two replicas, on the PIX pairs `(0,1)` and `(2,3)`):

    bash launch_sdar8b.sh                 # refuses if any card is already in use
    ALLOW_BUSY=1 bash launch_sdar8b.sh    # run anyway; contention is logged in RUNINFO.txt
    bash scripts/check_run.sh sdar8b      # progress

The launcher builds `../models/SDAR-8B-Chat-b64-local` if it is missing. It
then scores with EvalPlus, summarizes, and runs `scripts/compare_scale.py`.
That script refuses to compare unless every 8B arm's ordered task IDs match
the 1.7B fingerprints.

See [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md) for the official-code
comparison and the causality analysis.


## Density-ratio VGB (planning)

The second part of exp06 implements VGB (arXiv:2606.28301, Def. 4.2, r = 1)
with SDAR-1.7B as the reference model and SDAR-8B as the target, following
[vgb_density_ratio_decoding.tex](vgb_density_ratio_decoding.tex). Feasibility
analysis and the open design decisions are in [DESIGN.md](DESIGN.md). The
sampler is `adaptive_re/vgb.py`, with tests in `tests/test_vgb.py`. The first
GPU run to make is `scripts/probe_vgb.py --timing --walk`.

## exp05 re-run with SDAR-8B as the verifier (cross8b)

exp05's `xadaptive-remask` recipe with the Dream-7B verifier replaced by
SDAR-8B, to test whether an aligned verifier moves remask edits from style to
logic. Design: [DESIGN.md](DESIGN.md) §10. δ ∈ {0.1, 1}, each paired with its
no-remask counterfactual (`xadaptive`).

    bash launch_cross8b.sh                          # outputs/cross8b, logs/cross8b
    PYTHONPATH=. python scripts/analyze_cross8b.py  # style vs logic, exp05 vs cross8b
