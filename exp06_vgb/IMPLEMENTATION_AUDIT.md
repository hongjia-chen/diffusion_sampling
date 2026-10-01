# SDAR implementation audit for exp06_vgb

Checked 2026-09-24 against:

- Paper: <https://aclanthology.org/2026.findings-acl.1110/>
- Official repository: <https://github.com/JetAstra/SDAR>
- Official inference script:
  <https://github.com/JetAstra/SDAR/blob/main/generate.py>
- 8B-b64 checkpoint: <https://huggingface.co/JetLM/SDAR-8B-Chat-b64>

## b64 availability and host fit

`JetLM/SDAR-8B-Chat-b64` is an official released checkpoint. Its model tree is
16.4 GB and declares an 8B BF16 model. The host has four RTX A4000 GPUs with
16,376 MiB each. BF16 weights alone consume almost all of one card, before the
CUDA context and inference working memory, so the unmodified one-process-per-
GPU exp05 launcher is not feasible.

The model is feasible at BF16 if it is sharded over at least two cards. The
preferred topology is two replicas on the PIX-connected pairs `(0,1)` and
`(2,3)`. The exp05 provider must first be made device-map/tensor-parallel aware;
it currently moves the complete model to one device and creates all sampler
tensors there.

## Causality: exp05 matches SDAR

SDAR is not globally non-causal. It is causal between blocks and
bidirectional within the current block. The official inference script builds
a lower-triangular block mask and passes it to an attention kernel with
`is_causal=False`. The latter means “do not add a second, token-level triangular
mask”; it does not erase the explicit block mask.

Exp05 does the same thing:

- `adaptive_re.model.block_causal_mask` constructs the lower-triangular block
  mask.
- `adaptive_re.blockwise.generate_blockwise` supplies only the committed
  prefix plus the current block during decoding.
- The local SDAR attention uses the supplied mask with `is_causal=False`, so
  positions inside the current block can attend bidirectionally.

Therefore the apparent “causality off” setting is correct and should be kept.
Changing it to strict token-level causal attention would remove the feature
the checkpoint was trained for. Changing it to full-canvas bidirectional
attention would also be wrong: it would violate the checkpoint's inter-block
training mask, invalidate KV-cache reuse and the early-stop invariance, and
change the proposal probabilities and adaptive acceptance decisions. Either
change can materially alter quality and the algorithm's speed/quality curve.

## Differences from the official paper/code

| Area | exp05 | Official SDAR | Consequence |
| --- | --- | --- | --- |
| Block checkpoint | 1.7B-b64 | Main paper quality table uses the default b4 checkpoints; Study III varies 4/8/16/32/64 | The exp05 results are a block-size trade-off study, not directly comparable to the paper's headline scores. |
| Static schedule | fixed `k={1,2,4,8}` per forward | Static b64 uses 64 steps, i.e. one token per step; official dynamic decoding uses confidence thresholds | `fixed_k_1` is the paper-style b64 static ceiling. The k>1 arms are useful non-official latency/quality ablations. |
| Sampling | Greedy, temperature 0 | Paper evaluation is greedy; the standalone script defaults to temperature 1, top-k 0, top-p 1 | Matches the paper, but not the script's command-line defaults. |
| Stop timing | Stops as soon as the first stop token is settled | Reference script checks only after finishing a block | Same scored prefix without remasking; exp05 performs fewer forwards, so latency differs. |
| Output handling | Truncates at the first of `<|im_end|>` or `<|endoftext|>` and skips specials | Script returns the canvas and decodes with special tokens visible | Usually only trailing control text differs, but token counts and parsing policy are not identical. |
| Vocabulary | Restricts logits to shared IDs `<151665` for SDAR/Dream comparison | Uses all 151,936 SDAR logits | No exp05 greedy winner used a removed ID, but renormalized confidence/ranking and close adaptive thresholds can differ. |
| Benchmarks | 100-task seeded slices of GSM8K, HumanEval+, MBPP+ with local prompts/canvases | Paper uses its own full benchmark protocols; HumanEval/MBPP are not EvalPlus | Scores must not be compared directly to the paper's reported percentages. |
| Precision/kernel | BF16, patched FlashAttention-free PyTorch SDPA fallback, lean position-only LM-head projection | Official script defaults FP16; paper timing uses optimized infrastructure/JetEngine on other hardware | Expected function is the same, but borderline floating-point decisions and all wall-clock numbers can differ. |
| Engine | Python/Transformers harness on A4000 | Paper reports optimized engine results, including H200 measurements | Algorithmic forward counts are comparable; throughput is not. |
| Exp05 cross arm | SDAR proposes; Dream verifies; Dream remasks at most one token per block | Not part of official SDAR | This is the experiment's algorithm, so it should be labeled separately from an “official SDAR” baseline. |

## Interpretation for exp06_vgb

For a clean 1.7B-to-8B scale comparison, preserve the exp05 mask, prompts,
task IDs, canvases, stop policy, vocabulary cut, dtype, and fixed-k settings.
That answers whether model scale changes this experiment's frontier. Add a
separate official-fidelity arm only if needed: full vocabulary, b64 static
`k=1`, official block-end stopping, and no Dream verifier/remasking. Mixing
those changes into the scale-up would confound model size with implementation
policy.

